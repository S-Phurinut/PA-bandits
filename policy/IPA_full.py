import numpy as np
import scipy as sc
import math

class IPA_full():
    def __init__(self,type_arm,num_cost_learning=1,cost_alg='MultiBinSearch',**bandit_alg):
        self.type_arm=type_arm
        self.num_cost_learning=num_cost_learning
        self.cost_alg=cost_alg
        self.bandit_alg=bandit_alg

        if 'is_cost_known' in bandit_alg:
            self.is_cost_known=bandit_alg['is_cost_known']
        else:
            self.is_cost_known=False
        
        if 'is_reward_known' in bandit_alg:
            self.is_reward_known=bandit_alg['is_reward_known']
        else:
            self.is_reward_known=False
    
        self.reset=True

    def update_data(self,player):
        self.player=player

    def run(self,**info):
        if info['curr_round']==1 and self.reset:
            if self.is_cost_known:
                self.max_cost=np.array(self.player.cost)
                self.num_cost_learning=0
            else:
                self.max_cost=np.ones((self.player.num_agent,))

            self.min_cost=np.zeros((self.player.num_agent,))
            self.sum_reward=np.zeros((self.player.num_agent,))
            self.num_reward=np.zeros((self.player.num_agent,))
            self.best_incentive=np.zeros((self.player.num_agent,))
            self.est_reward=np.zeros((self.player.num_agent,))
            
            if self.num_cost_learning=='log2T': 
                self.num_cost_learning=math.ceil(np.log2(info['max_round']))
            elif self.num_cost_learning=='num_agent':
                self.num_cost_learning=int(self.player.num_agent)
            elif self.num_cost_learning=='T':
                self.num_cost_learning=int(info['max_round'])

            if self.bandit_alg['bandit_alg']=='TS' or self.bandit_alg['bandit_alg']=='UTS'  or self.bandit_alg['bandit_alg']=='incUTS' or self.bandit_alg['bandit_alg']=='concaveUTS'  or self.bandit_alg['bandit_alg']=='incTS' or self.bandit_alg['bandit_alg']=='concaveTS' and self.type_arm=='participation-based':
                if 'prior' in self.bandit_alg:
                    if self.bandit_alg['prior'] is not None:
                        if self.bandit_alg['prior'][0]=='beta':
                            self.alpha=np.array(self.bandit_alg['prior'][1])
                            self.beta=np.array(self.bandit_alg['prior'][2])
                else:
                    self.alpha=np.ones((self.player.num_agent,))
                    self.beta=np.ones((self.player.num_agent,))
                
                self.est_reward=self.alpha/(self.alpha+self.beta)

            if self.bandit_alg['bandit_alg']=='UTS'  or self.bandit_alg['bandit_alg']=='incUTS' or self.bandit_alg['bandit_alg']=='concaveUTS'  or self.bandit_alg['bandit_alg']=='OSUB':
                self.num_leader_count=np.zeros((self.player.num_agent,))

            self.reset=False
        else:

            #----------Update cost learning----------
            for n in range(self.player.num_agent):
                if info['previous_agent_response'][n]==0:
                    self.min_cost[n]=info['previous_incentive'][n]
                elif info['previous_agent_response'][n]==1:
                    self.max_cost[n]=info['previous_incentive'][n]
             #-----------Update Reward Data---------
            n=np.sum(info['previous_agent_response'])-1
            if n>=0:
                self.sum_reward[n]+=info['previous_reward']
                self.num_reward[n]+=1
                self.est_reward[n]= self.sum_reward[n]/self.num_reward[n]

                if self.bandit_alg['bandit_alg']=='TS' or self.bandit_alg['bandit_alg']=='UTS'  or self.bandit_alg['bandit_alg']=='incUTS' or self.bandit_alg['bandit_alg']=='concaveUTS'  or self.bandit_alg['bandit_alg']=='incTS' or self.bandit_alg['bandit_alg']=='concaveTS' :
                    if info['previous_reward']>0:
                        self.alpha[n]+=1
                    else:
                        self.beta[n]+=1
                    self.est_reward=self.alpha/(self.alpha+self.beta)*100000                

                    
                

            
        #====================Contracting Part============================
        #-------Phase1: Cost learning---------
        if info['curr_round']<=self.num_cost_learning :
            # print("max_cost=",self.max_cost)
            if self.cost_alg=='MultiBinSearch':
                incentive=(self.max_cost+self.min_cost)/2
            elif self.cost_alg=='leave-one-out_MultiBinSearch':
                incentive=(self.max_cost+self.min_cost)/2
                foreced_arm=int((info['curr_round']-1)%self.player.num_agent)
                incentive[foreced_arm]=self.max_cost[foreced_arm]
            elif self.cost_alg=='leave-cheapest-out_MultiBinSearch':
                incentive=(self.max_cost+self.min_cost)/2
                foreced_arm=int(np.argmin(self.max_cost))
                incentive[foreced_arm]=self.max_cost[foreced_arm]
            # elif self.cost_alg=='longEU':
            #     incentive=(self.max_cost+self.min_cost)/2
            elif self.cost_alg=='longEU_binomial':
                bnds=[(0.0, 1.0)]
                opt=sc.optimize.minimize(self.cal_longEU_binomial,x0=np.array([0.5]),bounds=bnds,args=(int(info['max_round']),int(info['curr_round']),self.est_reward))
                p=opt.x[0]
                print("100 offer=",-self.cal_longEU_binomial(1,int(info['max_round']),int(info['curr_round']),self.est_reward))
                print("longEU=",-opt.fun)
                print("offer cost at p=",p)
                incentive=p*(self.max_cost-self.min_cost)+self.min_cost
        
        #-------Phase2: Play Bandit Game-------
        else:
            
            if self.type_arm=='participation-based':
                if info['curr_round']== self.num_cost_learning+1:  
                    self.id_sorted_cost=np.argsort(self.max_cost)
                    sorted_cost=self.max_cost[self.id_sorted_cost]
                    self.cum_cost=np.cumsum(sorted_cost)
                    print("start bandit alg")
                    print("best_estimated_cost=",self.cum_cost)
                #---------------bandit alg-----------------
                if np.sum(self.num_reward<self.bandit_alg['min_sample_required'])>0:
                    pulled_arm=np.argmin(self.num_reward)
                else:
                    if self.is_reward_known:
                        self.est_reward=np.array(self.player.reward_generator.mean)
       
                    if self.bandit_alg['bandit_alg']=='UCB-lattimore':
                        pulled_arm=self.UCBlat_subroutine(info['max_round'])
                    elif self.bandit_alg['bandit_alg']=='UCB1':
                        pulled_arm=self.UCB1_subroutine(info['curr_round'])
                    elif self.bandit_alg['bandit_alg']=='TS':
                        pulled_arm=self.TS_subroutine()
                    elif self.bandit_alg['bandit_alg']=='incTS':
                        pulled_arm=self.incTS_subroutine(max_resampling_inc=self.bandit_alg['max_resampling_inc'])
                    elif self.bandit_alg['bandit_alg']=='concaveTS':
                        pulled_arm=self.concaveTS_subroutine(max_resampling_inc=self.bandit_alg['max_resampling_inc'],max_resampling_concave=self.bandit_alg['max_resampling_concave'])
                    elif self.bandit_alg['bandit_alg']=='UTS':
                        pulled_arm=self.UnimodalTS_subroutine()
                    elif self.bandit_alg['bandit_alg']=='incUTS':
                        pulled_arm=self.incUnimodalTS_subroutine(max_resampling_inc=self.bandit_alg['max_resampling_inc'])
                    elif self.bandit_alg['bandit_alg']=='concaveUTS':
                        pulled_arm=self.concaveUnimodalTS_subroutine(max_resampling_inc=self.bandit_alg['max_resampling_inc'],max_resampling_concave=self.bandit_alg['max_resampling_concave'])
                    elif self.bandit_alg['bandit_alg']=='OSUB':
                        pulled_arm=self.OSUB_subroutine(curr_round=info['curr_round'])
                    elif self.bandit_alg['bandit_alg']=='IMED-UB':
                        pulled_arm=self.IMED_UB_subroutine()
                    elif self.bandit_alg['bandit_alg']=='greedy':
                        pulled_arm=int(np.argmax(self.est_reward-self.cum_cost))
                        
                # print("At round=",info['curr_round']," pull arm=",pulled_arm)
                if info['curr_round']%1000==0: print("num reward=",self.num_reward)
                #------------Turn arm into incentive-----------
                if np.sum(self.max_cost)==0:
                    incentive=np.ones((self.player.num_agent,))*(-1)
                else:
                    incentive=np.zeros((self.player.num_agent,))
                for n in range(pulled_arm+1):
                    id_agent=self.id_sorted_cost[n]
                    incentive[id_agent]=self.max_cost[id_agent]

                #----------Predict best arm---------
                est_mean_reward=self.sum_reward/self.num_reward-self.cum_cost
                best_arm=0
                best_mean=-math.inf
                for arm in range(est_mean_reward.shape[0]):
                    if est_mean_reward[arm]<math.inf and est_mean_reward[arm]>best_mean:
                        best_mean=est_mean_reward[arm]
                        best_arm=arm
                
                self.best_incentive=np.zeros((self.player.num_agent,))
                if est_mean_reward[best_arm]<math.inf:
                    self.best_incentive=np.zeros((self.player.num_agent,))
                    for n in range(best_arm+1):
                        id_agent=self.id_sorted_cost[n]
                        self.best_incentive[id_agent]=self.max_cost[id_agent]
        if info['curr_round']==info['max_round']:
            print("final incentive=",incentive)
            print("final num reward=",self.num_reward)
        return incentive      

    def UCBlat_subroutine(self,max_round):
        UCB=np.zeros((self.player.num_agent,))
        for n in range(self.player.num_agent):
            if self.num_reward[n]==0:
                UCB[n]=1E6
            else:
                UCB[n]=self.sum_reward[n]/self.num_reward[n]+2*np.sqrt(np.log(max_round)/self.num_reward[n])-self.cum_cost[n]
        best_arm=int(np.argmax(UCB))
        return best_arm
    
    def UCB1_subroutine(self,round):
        UCB=np.zeros((self.player.num_agent,))
        for n in range(self.player.num_agent):
            if self.num_reward[n]==0:
                UCB[n]=1E6
            else:
                UCB[n]=self.sum_reward[n]/self.num_reward[n]+np.sqrt(2*np.log(round)/self.num_reward[n])-self.cum_cost[n]
        best_arm=int(np.argmax(UCB))
        return best_arm
    
    def TS_subroutine(self):
        sampled_thetas = []
        for n in range(self.player.num_agent):
            # Draw a sample from the Beta(alpha_i, beta_i) distribution
            sample = np.random.beta(self.alpha[n], self.beta[n])-self.cum_cost[n]
            sampled_thetas.append(sample)
        best_arm=np.argmax(sampled_thetas)
        return best_arm
    

    def incTS_subroutine(self,max_resampling_inc=1E7):
        if max_resampling_inc is None: max_resampling_inc=1E7
        resampling=True
        count_inc=-1
        while resampling:
            count_inc+=1
            sampled_thetas = []
            for n in range(self.player.num_agent):
                # Draw a sample from the Beta(alpha_i, beta_i) distribution
                sample = np.random.beta(self.alpha[n], self.beta[n])
                
                if n>0 and sample<sampled_thetas[-1] and count_inc<=max_resampling_inc: #if reward is not incresing, restart
                    resampling=True
                    break
                else:
                    resampling=False
                
                sampled_thetas.append(sample)

        best_arm=np.argmax(sampled_thetas-self.cum_cost)
        return best_arm
    
    def concaveTS_subroutine(self,max_resampling_inc=1E7,max_resampling_concave=1E6):
        if max_resampling_inc is None: max_resampling_inc=1E7
        if max_resampling_concave is None: max_resampling_concave=1E6
        resampling=True
        count_inc=-1
        count_cave=-1
        while resampling:
            count_inc+=1
            count_cave+=1
            sampled_thetas = []
            for n in range(self.player.num_agent):
                # Draw a sample from the Beta(alpha_i, beta_i) distribution
                sample = np.random.beta(self.alpha[n], self.beta[n])
                
                if n>0 and sample<sampled_thetas[-1] and count_inc<=max_resampling_inc: #if reward is not incresing, restart
                    resampling=True
                    break
                else:
                    if n>1 and sampled_thetas[-1]-sampled_thetas[-2]<sample-sampled_thetas[-1] and count_cave<=max_resampling_concave:  #if slope of reward is not decreasing, restart
                        resampling=True
                        break
                    elif n==1 and sampled_thetas[-1]<sample-sampled_thetas[-1] and count_cave<=max_resampling_concave:
                        resampling=True
                        break
                    else:
                        resampling=False
                
                sampled_thetas.append(sample)

        best_arm=np.argmax(sampled_thetas-self.cum_cost)
        return best_arm
    
    def UnimodalTS_subroutine(self):
        est_mean_reward=self.sum_reward/self.num_reward-self.cum_cost
        leader_arm=0
        best_mean=-math.inf
        num0_list=[]
        for arm in range(est_mean_reward.shape[0]):
            if self.num_reward[arm]==0:
                est_mean_reward[arm]=0
                num0_list.append(arm)
            
            if est_mean_reward[arm]>best_mean:
                best_mean=est_mean_reward[arm]
                leader_arm=arm
        
        if best_mean<=0 and len(num0_list)>0:
            leader_arm=int(np.random.choice(num0_list))
        
        if leader_arm==0:
            neighbor_arm=[leader_arm,1]
        elif leader_arm==self.player.num_agent-1:
            neighbor_arm=[leader_arm-1,leader_arm]
        else:
            neighbor_arm=[leader_arm-1,leader_arm,leader_arm+1]

        if self.num_leader_count[leader_arm] % len(neighbor_arm)==0:
            best_arm=int(leader_arm)
        else:   
            sampled_thetas = []
            for n in neighbor_arm:
                # Draw a sample from the Beta(alpha_i, beta_i) distribution
                sample = np.random.beta(self.alpha[n], self.beta[n])-self.cum_cost[n]
                sampled_thetas.append(sample)
            id_best_arm=np.argmax(sampled_thetas)
            best_arm=neighbor_arm[id_best_arm]

        self.num_leader_count[leader_arm]+=1
        return best_arm
    
    def incUnimodalTS_subroutine(self,max_resampling_inc=1E7):
        est_mean_reward=self.sum_reward/self.num_reward-self.cum_cost
        leader_arm=0
        best_mean=-math.inf
        num0_list=[]
        for arm in range(est_mean_reward.shape[0]):
            if self.num_reward[arm]==0:
                est_mean_reward[arm]=0
                num0_list.append(arm)
            
            if est_mean_reward[arm]>best_mean:
                best_mean=est_mean_reward[arm]
                leader_arm=arm
        
        if best_mean<=0 and len(num0_list)>0:
            leader_arm=int(np.random.choice(num0_list))
        
        if leader_arm==0:
            neighbor_arm=[leader_arm,1]
        elif leader_arm==self.player.num_agent-1:
            neighbor_arm=[leader_arm-1,leader_arm]
        else:
            neighbor_arm=[leader_arm-1,leader_arm,leader_arm+1]

        if self.num_leader_count[leader_arm] % len(neighbor_arm)==0:
            best_arm=int(leader_arm)
        else:  
            if max_resampling_inc is None: max_resampling_inc=1E7
            resampling=True
            count_inc=-1
            while resampling:
                count_inc+-1
                sampled_thetas = []
                for n in neighbor_arm:
                    # Draw a sample from the Beta(alpha_i, beta_i) distribution
                    sample = np.random.beta(self.alpha[n], self.beta[n])
                    
                    if len(sampled_thetas)>0 and sample<sampled_thetas[-1] and count_inc<=max_resampling_inc: #if reward is not incresing, restart
                        resampling=True
                        break
                    else:
                        resampling=False
                    
                    sampled_thetas.append(sample-self.cum_cost[n])

            id_best_arm=np.argmax(sampled_thetas)
            best_arm=neighbor_arm[id_best_arm]

        self.num_leader_count[leader_arm]+=1
        return best_arm
    
    def concaveUnimodalTS_subroutine(self,max_resampling_inc=1E7,max_resampling_concave=1E6):
        est_mean_reward=self.sum_reward/self.num_reward-self.cum_cost
        leader_arm=0
        best_mean=-math.inf
        num0_list=[]
        for arm in range(est_mean_reward.shape[0]):
            if self.num_reward[arm]==0:
                est_mean_reward[arm]=0
                num0_list.append(arm)
            
            if est_mean_reward[arm]>best_mean:
                best_mean=est_mean_reward[arm]
                leader_arm=arm
        
        if best_mean<=0 and len(num0_list)>0:
            leader_arm=int(np.random.choice(num0_list))
        
        if leader_arm==0:
            neighbor_arm=[leader_arm,1]
        elif leader_arm==self.player.num_agent-1:
            neighbor_arm=[leader_arm-1,leader_arm]
        else:
            neighbor_arm=[leader_arm-1,leader_arm,leader_arm+1]

        if self.num_leader_count[leader_arm] % len(neighbor_arm)==0:
            best_arm=int(leader_arm)
        else:  
            if max_resampling_inc is None: max_resampling_inc=1E7
            if max_resampling_concave is None: max_resampling_concave=1E6
            resampling=True
            count_inc=-1
            count_cave=-1
            while resampling:
                count_inc+=1
                count_cave+=1
                sampled_thetas = []
                for n in neighbor_arm:
                    # Draw a sample from the Beta(alpha_i, beta_i) distribution
                    sample = np.random.beta(self.alpha[n], self.beta[n])
                    
                    if len(sampled_thetas)>0 and sample<sampled_thetas[-1] and count_inc<=max_resampling_inc: #if reward is not incresing, restart
                        resampling=True
                        break
                    else:
                        if len(sampled_thetas)>1 and sampled_thetas[-1]-sampled_thetas[-2]<sample-sampled_thetas[-1] and count_cave<=max_resampling_concave:  #if slope of reward is not decreasing, restart
                            resampling=True
                            break
                        else:
                            resampling=False
                        
                    sampled_thetas.append(sample-self.cum_cost[n])

            id_best_arm=np.argmax(sampled_thetas)
            best_arm=neighbor_arm[id_best_arm]

        self.num_leader_count[leader_arm]+=1
        return best_arm
    
    def IMED_UB_subroutine(self):
        est_mean_reward=self.sum_reward/self.num_reward-self.cum_cost
        est_mean=self.sum_reward/self.num_reward
        leader_arm=0
        best_mean=-math.inf
        num0_list=[]
        for arm in range(est_mean_reward.shape[0]):
            if self.num_reward[arm]==0:
                est_mean_reward[arm]=0
                num0_list.append(arm)
            
            if est_mean_reward[arm]>best_mean:
                best_mean=est_mean_reward[arm]
                leader_arm=arm
        
        if best_mean<=0 and len(num0_list)>0:
            leader_arm=int(np.random.choice(num0_list))
        
        if leader_arm==0:
            neighbor_arm=[leader_arm,1]
        elif leader_arm==self.player.num_agent-1:
            neighbor_arm=[leader_arm-1,leader_arm]
        else:
            neighbor_arm=[leader_arm-1,leader_arm,leader_arm+1]

        IMED=math.inf
        for arm in neighbor_arm:
            if self.num_reward[arm]<=0:
                I=0
            else:
                I=self.num_reward[arm]*self.kl_bernoulli(est_mean[arm],est_mean[leader_arm])+math.log(self.num_reward[arm])

            if I<IMED:
                IMED=I
                best_arm=arm
        return best_arm
            
    def OSUB_subroutine(self,curr_round):
        est_mean_reward=self.sum_reward/self.num_reward-self.cum_cost
        est_mean=self.sum_reward/self.num_reward
        leader_arm=0
        best_mean=-math.inf
        num0_list=[]
        for arm in range(est_mean_reward.shape[0]):
            if self.num_reward[arm]==0:
                est_mean_reward[arm]=0
                est_mean[arm]=0
                num0_list.append(arm)
            
            if est_mean_reward[arm]>best_mean:
                best_mean=est_mean_reward[arm]
                leader_arm=arm
        
        if best_mean<=0 and len(num0_list)>0:
            leader_arm=int(np.random.choice(num0_list))
        
        if leader_arm==0:
            neighbor_arm=[leader_arm,1]
        elif leader_arm==self.player.num_agent-1:
            neighbor_arm=[leader_arm-1,leader_arm]
        else:
            neighbor_arm=[leader_arm-1,leader_arm,leader_arm+1]
        
        self.num_leader_count[leader_arm]+=1

        if (self.num_leader_count[leader_arm]-1) % len(neighbor_arm)==0:
            best_arm=int(leader_arm)
        else:
            q=0
            for arm in neighbor_arm:
                if self.num_reward[arm] == 0 or est_mean[arm]>=1:
                    low=1
                else:
                    # OSUB's F(p,s,n) function:
                    #  F(p,s,n) = sup{ q >= p : s * KL(p,q) <= log(n) + c log(log(n)) }

                    rhs = math.log(curr_round) + self.bandit_alg['bandit_alg_paras'] * math.log(math.log(curr_round))

                    # Binary search for q in [p,1]
                    low, high = est_mean[arm], 1.0
                    for _ in range(50):
                        mid = (low + high) / 2
                        if self.num_reward[arm] * self.kl_bernoulli(est_mean[arm], mid) <= rhs:
                            low = mid
                        else:
                            high = mid
                if low>q:
                    q=low
                    best_arm=arm
        return best_arm
            

    def kl_bernoulli(self,p, q):
        """KL divergence KL(p || q) for Bernoulli means p,q."""
        eps = 1e-12
        p = min(max(p, eps), 1 - eps)
        q = min(max(q, eps), 1 - eps)
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    
    #For longEU calculation 

    def cal_longEU_binomial(self,p,T,t,reward):
        # p=np.max((np.zeros(self.player.num_agent,),np.min((np.ones(self.player.num_agent,),(cost-self.min_cost)/(self.max_cost-self.min_cost)),axis=0)),axis=0)
        cost=p*(self.max_cost-self.min_cost)+self.min_cost
        # print("cost=",cost," at p=",p)
        id_sorted_cost=np.argsort(self.max_cost)
        sorted_cost=self.max_cost[id_sorted_cost]
        cum_cost=np.cumsum(sorted_cost)

        longEU=(T-t)*(1-p)*np.max(reward-cum_cost)

        for arm in range(1,self.player.num_agent+1):
            longEU+=(T-t+1)*sc.stats.binom.pmf(arm, self.player.num_agent, p)*(reward[arm-1]-np.sum(cost)*math.comb(self.player.num_agent-1, arm-1)/math.comb(self.player.num_agent, arm))
        # print("longEU=",longEU)
        return -longEU

    def poisson_binomial_pmf_full(self,p):
        dp = [0.0] * (self.player.num_agent + 1)
        dp[0] = 1.0
        for pi in p:
            for j in range(self.player.num_agent, 0, -1):
                dp[j] = dp[j] * (1 - pi) + dp[j - 1] * pi
            dp[0] *= (1 - pi)

        return dp
                