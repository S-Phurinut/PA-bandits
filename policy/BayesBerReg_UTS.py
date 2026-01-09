import numpy as np
import scipy as sc
import cvxpy as cp
import math

import scipy.signal
import scipy.signal.windows

# Patch older import name for PyMC compatibility
if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = scipy.signal.windows.gaussian
import pymc as pm
import pytensor.tensor as pt

class Bayesian_Bernoulli_Regression_UnimodalTS():
    def __init__(self,type_arm,num_cost_learning=1,cost_alg='MultiBinSearch',**bandit_alg):
        self.type_arm=type_arm
        self.num_cost_learning=num_cost_learning
        self.cost_alg=cost_alg

        if 'is_cost_known' in bandit_alg:
            self.is_cost_known=bandit_alg['is_cost_known']
        else:
            self.is_cost_known=False
    
        self.reset=True
        self.bandit_alg=bandit_alg

    def update_data(self,player):
        self.player=player
        self.N=self.player.num_agent

    def run(self,**info):
        if info['curr_round']==1 and self.reset:
            if self.is_cost_known:
                self.max_cost=np.array(self.player.cost)
                self.num_cost_learning=0
            else:
                self.max_cost=np.ones((self.player.num_agent,))
            self.min_cost=np.zeros((self.N,))
            self.sum_reward=np.zeros((self.N,))
            self.num_reward=np.zeros((self.N,))
            self.best_incentive=np.zeros((self.N,))
            self.num_leader_count=np.zeros((self.player.num_agent,))
            if self.num_cost_learning=='log2T': self.num_cost_learning=math.ceil(np.log2(info['max_round']))
            self.input=[]
            self.output=[]
            self.is_model_fitted=False

            self.X = np.zeros((self.N+1, self.N+1), dtype=np.float64)  # exclude first arm
            for k in range(0, self.N+1):
                for j in range(self.N+1):
                    if self.bandit_alg['constraint_model']=="increasing":
                        if k>=j:
                            self.X[k, j] = 1
                    elif self.bandit_alg['constraint_model']=="increasing-concave":
                        self.X[k, j] = min(k+1,j+1)
            self.reset=False
        else:

            #----------Update cost learning----------
            for n in range(self.N):
                if info['previous_agent_response'][n]==0:
                    self.min_cost[n]=info['previous_incentive'][n]
                elif info['previous_agent_response'][n]==1:
                    self.max_cost[n]=info['previous_incentive'][n]
             #-----------Update Reward Data---------
            n=np.sum(info['previous_agent_response'])-1
            if n>=0:
                self.sum_reward[n]+=info['previous_reward']
                self.num_reward[n]+=1

                self.input.append(n+1)
                self.output.append(info['previous_reward'])


            
        #====================Contracting Part============================
        #-------Phase1: Cost learning---------
        if info['curr_round']<=self.num_cost_learning :
            if self.cost_alg=='MultiBinSearch':
                incentive=(self.max_cost+self.min_cost)/2
            elif self.cost_alg=='leave-one-out_MultiBinSearch':
                incentive=(self.max_cost+self.min_cost)/2
                foreced_arm=int((info['curr_round']-1)%self.N)
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
                if np.sum(self.num_reward<=0)>0:
                    pulled_arm=np.argmin(self.num_reward)
                else:
                    if self.bandit_alg['refit_step']=="adaptive-log10":
                        refit_step=max(int(10**(math.floor(np.log10(info['curr_round']))-1)),1)
                    else:
                        refit_step=int(self.bandit_alg['refit_step'])

                    if info['curr_round']%refit_step==0  or info['curr_round']<=self.bandit_alg['max_round_1step'] or self.is_model_fitted==False:
                        self.is_model_fitted=True
                        # objective: least squares
                        with pm.Model() as model:
                            
                            
                            if self.bandit_alg['likelihood_model']== "cdf-exp":
                                # Bounded positive weights
                                w = pm.Uniform("w",lower=0, upper= 10, shape=self.N+1)
                                # Latent concave function
                                f_all = pt.dot(pt.as_tensor_variable(self.X, dtype="float64"), w)
                                # Hazard-style link for flexible probabilities
                                p_all = pm.Deterministic("p_all", 1 - pm.math.exp(-f_all))
                           
                           
                            # Likelihood
                            y = pm.Bernoulli("y", p=p_all[self.input], observed=self.output)

                            # Safe initialization
                            self.trace = pm.sample(draws=self.bandit_alg['pymc_draws'], tune=self.bandit_alg['pymc_tune'], target_accept=self.bandit_alg['pymc_target_accept'],
                                            initvals={"w": np.full(self.N+1, 0.01)}, cores=self.bandit_alg['pymc_cores'],chains=self.bandit_alg['pymc_chains'])
                
                    self.p_samples = self.trace.posterior["p_all"].stack(draws=("chain","draw")).values
                    p_mean=self.p_samples.mean(axis=1)[1:]

                    print("avg reward=",self.sum_reward/self.num_reward)
                    print("posterior mean=",p_mean)
                    print("posterior std=",self.p_samples.std(axis=1)[1:])
                    print("num reward=",self.num_reward)

                    leader_arm=np.argmax(p_mean-self.cum_cost)
                    self.num_leader_count[leader_arm]+=1
                    if leader_arm==0:
                        neighbor_arm=[leader_arm,1]
                    elif leader_arm==self.player.num_agent-1:
                        neighbor_arm=[leader_arm-1,leader_arm]
                    else:
                        neighbor_arm=[leader_arm-1,leader_arm,leader_arm+1]

                    if self.num_leader_count[leader_arm] % len(neighbor_arm)==0:
                        pulled_arm=int(leader_arm)
                    else:   
                        p_all=self.p_samples[1:,np.random.randint(0,self.p_samples.shape[1])] #exclude posterior of 0 agents
                        
                        #---------------Greedy alg-----------------
                        best_utility=-math.inf
                        pulled_arm=leader_arm
                        for arm in range(self.N):
                            utility=p_all[arm]-self.cum_cost[arm]
                            if utility>best_utility and arm in neighbor_arm:
                                pulled_arm=arm
                                best_utility=utility
                
                #------------Turn arm into incentive-----------
                incentive=np.zeros((self.N,))
                for n in range(pulled_arm+1):
                    id_agent=self.id_sorted_cost[n]
                    incentive[id_agent]=self.max_cost[id_agent]
                
                if self.is_model_fitted==False:
                    best_arm=0
                else:
                    best_arm=np.argmax(p_mean-self.cum_cost)
                self.best_incentive=np.zeros((self.N,))
                for n in range(best_arm+1):
                    id_agent=self.id_sorted_cost[n]
                    self.best_incentive[id_agent]=self.max_cost[id_agent]

        return incentive    

            
                