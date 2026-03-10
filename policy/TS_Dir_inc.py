import numpy as np
from scipy.stats import beta as beta_dist
import scipy.signal
import scipy.signal.windows

# Patch older import name for PyMC compatibility
if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = scipy.signal.windows.gaussian
import pymc as pm
import math

class TS_Dirichlet_increment():
    def __init__(self,type_arm,num_cost_learning=1,cost_alg='MultiBinSearch',
                 dir_para=0.5,**bandit_alg):
        self.type_arm=type_arm
        self.num_cost_learning=num_cost_learning
        self.cost_alg=cost_alg

        if 'is_cost_known' in bandit_alg:
            self.is_cost_known=bandit_alg['is_cost_known']
        else:
            self.is_cost_known=False

        if 'is_reward_known' in bandit_alg:
            self.is_reward_known=bandit_alg['is_reward_known']
        else:
            self.is_reward_known=False
        
        self.bandit_alg=bandit_alg
        self.reset=True
        self.dir_para=dir_para

        

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
            if self.num_cost_learning=='log2T': self.num_cost_learning=math.ceil(np.log2(info['max_round']))
            self.alpha=np.ones((self.player.num_agent,))
            self.beta=np.ones((self.player.num_agent,))
            
            if type(self.dir_para)==str:
                if self.dir_para=="Global-Jeffreys":
                    self.dir_para=np.ones((self.player.num_agent+1,))/(self.player.num_agent+1)
            elif type(self.dir_para)==list:
                self.dir_para=np.array(self.dir_para)
            else:
                self.dir_para=np.ones((self.player.num_agent+1,))*self.dir_para
            self.fit_first_model=True
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

                if info['previous_reward']>0:
                    self.alpha[n]+=1
                else:
                    self.beta[n]+=1


            
        #====================Contracting Part============================
        #-------Phase1: Cost learning---------
        if info['curr_round']<=self.num_cost_learning :
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
        
        #-------Phase2: Play Bandit Game-------
        else:
            
            if self.type_arm=='participation-based':
                if info['curr_round']== self.num_cost_learning+1:  
                    self.id_sorted_cost=np.argsort(self.max_cost)
                    sorted_cost=self.max_cost[self.id_sorted_cost]
                    self.cum_cost=np.cumsum(sorted_cost)
                    print("start bandits alg")
                    print("best_estimated_cost=",self.cum_cost)

                #----------Monotone Concave regression---------
                if np.sum(self.num_reward<self.bandit_alg['min_sample_required'])>0:
                    pulled_arm=np.argmin(self.num_reward)
                else:
                    refit_model=True
                    if self.bandit_alg['refit_step']=="adaptive-log10":
                        refit_step=max(int(10**(math.floor(np.log10(info['curr_round'])))),1)
                        if info['curr_round']%min(refit_step,self.bandit_alg['refit_max_round_step'])==0  or info['curr_round']<=self.bandit_alg['refit_max_round_1step'] :
                            refit_model=True
                        else:
                            refit_model=False

                    if type(self.bandit_alg['refit_step'])==int:
                        if (info['curr_round']-1)%int(self.bandit_alg['refit_step'])==0 or info['curr_round']<=self.bandit_alg['refit_max_round_1step'] :
                            refit_model=True
                        else:
                            refit_model=False

                    if self.fit_first_model:
                        refit_model=True
                        self.fit_first_model=False

                    if refit_model:
                        with pm.Model() as model:
                            # Positive latent variables
                            g = pm.Gamma("g", alpha=self.dir_para, beta=1.0, shape=self.player.num_agent + 1)

                            # Normalize to simplex
                            p = pm.Deterministic("p", g / pm.math.sum(g))

                            # Monotone probabilities:
                            # f(1)=p[0], f(2)=p[0]+p[1], ..., f(N)=p[0]+...+p[N-1]
                            f = pm.Deterministic("f", pm.math.cumsum(p)[:-1])

                            # Binomial likelihood at each x
                            pm.Binomial("obs", n=self.alpha+self.beta-2, p=f, observed=self.alpha-1)

                            # NUTS
                            idata = pm.sample(
                                draws=self.bandit_alg['pymc_draws'],          # keep only one posterior draw
                                tune=self.bandit_alg['pymc_tune'],        # warmup / adaptation
                                chains=self.bandit_alg['pymc_chains'],
                                cores=self.bandit_alg['pymc_cores'],
                                target_accept=self.bandit_alg['pymc_target_accept'],
                            )

                        self.f_sample = idata.posterior["f"].values[0, :]

                    index=np.random.randint(0,self.f_sample.shape[0])
                    sample=self.f_sample[index]
                    

                    if info['curr_round']%500==0:
                        print("TS_sample=",sample)      
                        print("num reward=",self.num_reward)

                    #---------------Greedy alg-----------------
                    best_utility=-math.inf
                    pulled_arm=0
                    for arm in range(self.player.num_agent):
                        utility=sample[arm]-self.cum_cost[arm]
                        if utility>best_utility:
                            pulled_arm=arm
                            best_utility=utility
                    
                    if self.bandit_alg['include_arm0']:
                        if best_utility<0:
                            pulled_arm=-1
                
                #------------Turn arm into incentive-----------
                incentive=np.zeros((self.player.num_agent,))
                for n in range(pulled_arm+1):
                    id_agent=self.id_sorted_cost[n]
                    incentive[id_agent]=self.max_cost[id_agent]
                    
                #----------Predict best arm---------
                est_mean_reward=self.sum_reward/self.num_reward-self.cum_cost
                best_arm=-1
                best_mean=0
                for arm in range(est_mean_reward.shape[0]):
                    if est_mean_reward[arm]<math.inf and est_mean_reward[arm]>best_mean:
                        best_mean=est_mean_reward[arm]
                        best_arm=arm

                self.best_incentive=np.zeros((self.player.num_agent,))
                if best_arm>=0:
                    if est_mean_reward[best_arm]<math.inf:
                        self.best_incentive=np.zeros((self.player.num_agent,))
                        for n in range(best_arm+1):
                            id_agent=self.id_sorted_cost[n]
                            self.best_incentive[id_agent]=self.max_cost[id_agent]
                            
        if info['curr_round']==info['max_round']:
            print("final incentive=",incentive)
            print("final num reward=",self.num_reward)
            if self.bandit_alg['include_arm0']: print("num arm0=",info['max_round']-np.sum(self.num_reward))
        return incentive    

            
                