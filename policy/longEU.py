import numpy as np
import scipy as sc
import math
from poibin import PoiBin

class longEU():
    def __init__(self,type_arm,**bandit_alg):
        self.type_arm=type_arm
        self.bandit_alg=bandit_alg

        if 'is_reward_known' in bandit_alg:
            self.is_reward_known=bandit_alg['is_reward_known']
        else:
            self.is_reward_known=False


        if 'is_cost_known' in bandit_alg:
            self.is_cost_known=bandit_alg['is_cost_known']
        else:
            self.is_cost_known=False
        self.num_optimiser=bandit_alg['num_optimiser']
        self.reset=True

    def update_data(self,player):
        self.player=player

    def run(self,**info):
        if info['curr_round']==1 and self.reset:
            if self.is_cost_known:
                self.max_cost=np.array(self.player.cost)
                self.max_num_cost_learning=0
            else:
                self.max_cost=np.ones((self.player.num_agent,))

            self.previous_p=np.ones((self.player.num_agent,))*0.5
            self.min_cost=np.zeros((self.player.num_agent,))
            self.sum_reward=np.zeros((self.player.num_agent,))
            self.num_reward=np.zeros((self.player.num_agent,))
            self.best_incentive=np.zeros((self.player.num_agent,))
            
            if self.bandit_alg['est_reward']=='TS' or self.bandit_alg['est_reward']=='posterior-mean'  and self.type_arm=='participation-based':
                if self.bandit_alg['prior'] is not None:
                    if self.bandit_alg['prior'][0]=='beta':
                        if self.bandit_alg['prior'][1][0]=='fixed':
                            self.alpha=np.ones((self.player.num_agent,))*self.bandit_alg['prior'][1][1]
                        elif self.bandit_alg['prior'][1][0]=='linear':
                            self.alpha=np.linspace(self.bandit_alg['prior'][1][1],self.bandit_alg['prior'][1][2],num=self.player.num_agent)

                        if self.bandit_alg['prior'][2][0]=='fixed':
                            self.beta=np.ones((self.player.num_agent,))*self.bandit_alg['prior'][2][1] 
                        elif self.bandit_alg['prior'][2][0]=='linear':
                            self.alpha=np.linspace(self.bandit_alg['prior'][2][1],self.bandit_alg['prior'][2][2],num=self.player.num_agent)
                else:
                    self.alpha=np.ones((self.player.num_agent,))*1
                    self.beta=np.ones((self.player.num_agent,))

            self.reset=False
        else:

            #----------Update cost learning----------
            for n in range(self.player.num_agent):
                if info['previous_agent_response'][n]==0 and info['previous_incentive'][n]>=self.min_cost[n]:
                    self.min_cost[n]=info['previous_incentive'][n]
                elif info['previous_agent_response'][n]==1 and info['previous_incentive'][n]<=self.max_cost[n]:
                    self.max_cost[n]=info['previous_incentive'][n]
             #-----------Update Reward Data---------
            n=np.sum(info['previous_agent_response'])-1
            if n>=0:
                self.sum_reward[n]+=info['previous_reward']
                self.num_reward[n]+=1

                if self.bandit_alg['est_reward']=='TS' or self.bandit_alg['est_reward']=='posterior-mean' :
                    if info['previous_reward']>0:
                        self.alpha[n]+=1
                    else:
                        self.beta[n]+=1

            
        #====================Contracting Part============================
        if self.type_arm=='participation-based':
            self.id_sorted_cost=np.lexsort((np.random.rand(self.max_cost.shape[0]), self.max_cost))
            sorted_cost=self.max_cost[self.id_sorted_cost] 
            self.cum_cost=np.cumsum(sorted_cost) #max cum cost

            #---------------Reward estimator ----------------- 
            if self.is_reward_known:
                est_reward=np.array(self.player.reward_generator.mean)
            else:
                if self.bandit_alg['est_reward']=='UCB-lattimore':
                    est_reward=self.UCBlat_value(info['max_round'])
                elif self.bandit_alg['est_reward']=='UCB1':
                    est_reward=self.UCB1_value(info['curr_round'],info['max_round'])
                elif self.bandit_alg['est_reward']=='TS':
                    est_reward=self.TS_value()
                elif self.bandit_alg['est_reward']=='posterior-mean':
                    est_reward=self.alpha/(self.alpha+self.beta)

            eps = 0 #float(1E-1)
            bnds = [(float(0 - eps), float(1 + eps)) for _ in range(self.player.num_agent)]
            best_longEU=-math.inf
            incentive=np.array(self.max_cost)
            best_p=np.ones((self.player.num_agent,))

            for prob in range(0,self.num_optimiser):
                if prob==0:
                    x0=self.previous_p
                elif prob==1:
                    x0=np.ones((self.player.num_agent,))*0.5
                else:
                    x0=np.clip(np.random.rand(self.player.num_agent,),0.1,0.9)

                
                
                opt=sc.optimize.minimize(self.longEU_value,x0=x0,bounds=bnds,args=(info['max_round'],info['curr_round'],est_reward),tol=1E-12,method="SLSQP")
                p=np.clip(opt.x, 0, 1)
                longEU=-opt.fun
                if p is not None and longEU>best_longEU and -1E20<longEU<1E20 and np.sum(p>=1E-12)>0:
                    # print(p,longEU)
                    best_longEU=float(longEU)
                    best_p=np.array(p)

            #Check exploit arm        
            exploit_arm=np.argmax(est_reward-self.cum_cost)
            exploit_p=np.zeros((self.player.num_agent,))
            for n in range(exploit_arm+1):
                id_agent=self.id_sorted_cost[n]
                exploit_p[id_agent]=1
            exploit_longEU=-self.longEU_value(exploit_p,info['max_round'],info['curr_round'],est_reward)
            if exploit_longEU>best_longEU:
                best_p=exploit_p

            incentive=best_p*(self.max_cost-self.min_cost)+self.min_cost*(best_p>0)
            self.previous_p=np.array(best_p)

            if info['curr_round']<=10:
                print("est_reward=",np.round(est_reward,4))
                print("incentive=",np.round(incentive,4))
                print("with p=",np.round(best_p,4))
                print("p_arm =",np.round(np.clip(PoiBin(best_p).pmf(np.arange(self.player.num_agent+1)),0,1),4))
                print("longEU optimizer=",best_longEU)
                # true_p=np.zeros((self.player.num_agent,))
                # true_p[:3]=np.ones((3,))
                # print("longEU 0.       =",-self.longEU_value(np.ones((self.player.num_agent,))*0,info['max_round'],info['curr_round'],est_reward))
                # print("longEU 0.0001 at1=",-self.longEU_value(A,info['max_round'],info['curr_round'],est_reward))
                # print("longEU re-compute=",-self.longEU_value(B,info['max_round'],info['curr_round'],est_reward))
                print("longEU BinSearch=",-self.longEU_value(np.ones((self.player.num_agent,))*0.5,info['max_round'],info['curr_round'],est_reward))
                # print("longEU 1.       =",-self.longEU_value(np.ones((self.player.num_agent,))*1,info['max_round'],info['curr_round'],est_reward))
                print("\n")
            
            if info['curr_round']%1000==0: print("num reward=",self.num_reward)
            

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
            print("final incentive=",np.round(incentive,4))
            print(" with p=",np.round(best_p,4))
            print("final num reward=",self.num_reward)
            print("longEU optimizer=",best_longEU)
            true_p=np.zeros((self.player.num_agent,))
            true_p[:3]=np.ones((3,))
            print("longEU optimal",-self.longEU_value(self.max_cost,info['max_round'],info['curr_round'],est_reward))
        return incentive   

    def longEU_value(self,p,T,t,reward):
        # p=np.clip(p, 0, 1)
        cost=p*(self.max_cost-self.min_cost)+self.min_cost*(p>0)
        longEU=0

        if np.sum(p<=1E-4)>=self.player.num_agent:
            longEU+=-1E8
        elif np.sum(p>=1-1E-12)>=self.player.num_agent:
            longEU+=(T-t+1)*reward[self.player.num_agent-1]-(T-t+1)*np.dot(p,cost)
        else:
            pb = PoiBin(p)
            longEU+=(T-t)*np.clip(pb.pmf(0), 0, 1)*np.max(reward-self.cum_cost)
            num_offered_agent=int(np.sum(p>1E-6))
            num_guaranteed_agent=int(np.sum(p>=1-1E-6))
            for arm in range(max(num_guaranteed_agent,1),num_offered_agent+1): #math.comb is wrong
                longEU+=(T-t+1)*reward[arm-1]*np.clip(pb.pmf(arm), 0, 1)
            longEU+=-(T-t+1)*np.dot(p,cost)

        return -longEU   

    def UCBlat_value(self,max_round):
        UCB=np.zeros((self.player.num_agent,))
        for n in range(self.player.num_agent):
            if self.num_reward[n]==0:
                UCB[n]=1+2*np.sqrt(np.log(max_round))
            else:
                UCB[n]=self.sum_reward[n]/self.num_reward[n]+2*np.sqrt(np.log(max_round)/self.num_reward[n])
        return UCB
    
    def UCB1_value(self,round,max_round):
        UCB=np.zeros((self.player.num_agent,))
        for n in range(self.player.num_agent):
            if self.num_reward[n]==0:
                UCB[n]=1+2*np.sqrt(np.log(max_round))
            else:
                UCB[n]=self.sum_reward[n]/self.num_reward[n]+np.sqrt(2*np.log(round)/self.num_reward[n])
        return UCB
    
    def TS_value(self):
        sampled_thetas = []
        for n in range(self.player.num_agent):
            # Draw a sample from the Beta(alpha_i, beta_i) distribution
            sample = np.random.beta(self.alpha[n], self.beta[n])
            sampled_thetas.append(sample)
        return np.array(sampled_thetas)
    
            


                