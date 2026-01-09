import numpy as np
import scipy as sc
import cvxpy as cp
import math

class TS_Monotone_Concave_LeastSquare():
    def __init__(self,type_arm,num_cost_learning=1,cost_alg='MultiBinSearch',**bandit_alg):
        self.type_arm=type_arm
        self.num_cost_learning=num_cost_learning
        self.cost_alg=cost_alg
        self.need_weighted_LS=bandit_alg['need_weighted_LS']
        if 'is_cost_known' in bandit_alg:
            self.is_cost_known=bandit_alg['is_cost_known']
        else:
            self.is_cost_known=False

        self.bandit_alg=bandit_alg
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
            if self.num_cost_learning=='log2T': self.num_cost_learning=math.ceil(np.log2(info['max_round']))
            self.alpha=np.ones((self.player.num_agent,))
            self.beta=np.ones((self.player.num_agent,))
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
                    if np.sum(self.num_reward<0)>0:
                        pulled_arm=np.argmin(self.num_reward)
                    else:
                        n=self.player.num_agent+1
                        f = cp.Variable(n)
                        # constraints
                        cons = []
                        cons += [ f[0]==0 ]
                        # isotonic: f[i+1] >= f[i]
                        cons += [f[i+1] - f[i] >= 0 for i in range(n-1)]
                        # concave: second differences <= 0
                        cons += [f[i+2] - 2*f[i+1] + f[i] <= 0 for i in range(n-2)]
                        # maximum prob <=1
                        cons += [f[n-1] <= 1]

                        # objective: least squares

                        TS_sample = np.zeros((self.player.num_agent+1,))
                        for n in range(self.player.num_agent):
                            # Draw a sample from the Beta(alpha_i, beta_i) distribution
                            TS_sample[n+1] = np.random.beta(self.alpha[n], self.beta[n])

                        if info['curr_round']%100==0:  
                            # print(TS_sample)
                            print("num reward=",self.num_reward)
                    
                        if self.need_weighted_LS:
                            obj = cp.Minimize(cp.sum_squares(cp.multiply(np.concatenate(([1],self.num_reward)),(TS_sample - f))))
                        else:
                            is_sample_exist= np.ones((self.player.num_agent))
                            for n in range(self.player.num_agent):
                                if self.num_reward[n]==0:
                                    is_sample_exist[n]=0
                            obj = cp.Minimize(cp.sum_squares(cp.multiply(np.concatenate(([1],is_sample_exist)),(TS_sample - f))))
                        
                        
                        prob = cp.Problem(obj, cons)
                        prob.solve(solver=cp.OSQP)
                        est_fun=lambda x : f.value[x]

                        # print("est success prob=",f.value)

                        #---------------Greedy alg-----------------
                        best_utility=-math.inf
                        pulled_arm=0
                        for arm in range(self.player.num_agent):
                            utility=est_fun(arm+1)-self.cum_cost[arm]
                            if utility>best_utility:
                                pulled_arm=arm
                                best_utility=utility
                
                #------------Turn arm into incentive-----------
                incentive=np.zeros((self.player.num_agent,))
                for n in range(pulled_arm+1):
                    id_agent=self.id_sorted_cost[n]
                    incentive[id_agent]=self.max_cost[id_agent]
                    
                self.best_incentive=np.array(incentive)
        return incentive    

            
                