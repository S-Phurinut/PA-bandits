import numpy as np
import scipy as sc
import cvxpy as cp
import math

class Monotone_Concave_LeastSquare_UnimodalTS():
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
            self.alpha=np.ones((self.player.num_agent,))
            self.beta=np.ones((self.player.num_agent,))
            self.num_leader_count=np.zeros((self.player.num_agent,))
            if self.num_cost_learning=='log2T': self.num_cost_learning=math.ceil(np.log2(info['max_round']))
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
                    print("start bandit alg")
                    print("best_estimated_cost=",self.cum_cost)

                #----------Monotone Concave regression---------
                if np.sum(self.num_reward<self.bandit_alg['min_sample_required'])>0:
                    pulled_arm=np.argmin(self.num_reward)
                    leader_arm=0
                else:

                    if self.bandit_alg['refit_step']=="adaptive-log10":
                        if self.bandit_alg['refit_step_constant'] is None : self.bandit_alg['refit_step_constant']=1
                        refit_step=max(int(10**(math.floor(np.log10(info['curr_round']))-self.bandit_alg['refit_step_constant'])),1)
                    else:
                        refit_step=int(self.bandit_alg['refit_step'])

                    if info['curr_round']%refit_step==0 or info['curr_round']<=self.bandit_alg['max_round_1step']:
                        n=self.player.num_agent+1
                        f = cp.Variable(n)
                        # constraints
                        cons = []
                        cons += [ f[0]==0 ]
                        
                        if self.bandit_alg['constraint_model']=="increasing":
                            cons += [f[i+1] - f[i] >= 0 for i in range(n-1)]  # isotonic: f[i+1] >= f[i]
                        
                        elif self.bandit_alg['constraint_model']=="increasing-concave":
                            cons += [f[i+1] - f[i] >= 0 for i in range(n-1)] # isotonic: f[i+1] >= f[i]
                            cons += [f[i+2] - 2*f[i+1] + f[i] <= 0 for i in range(n-2)] # concave: second differences <= 0
                        # maximum prob <=1
                        cons += [f[n-1] <= 1]

                        # objective: least squares
                        y=np.concatenate(([0],self.sum_reward/self.num_reward))
                        obj = cp.Minimize(cp.sum_squares(cp.multiply(np.concatenate(([1],self.num_reward)),(y - f))))
                        prob = cp.Problem(obj, cons)
                        prob.solve(solver=cp.OSQP)
                        self.est_fun=lambda x : f.value[x]

                    # print("est success prob=",f.value)
                    if info['curr_round']%1000==0: print("num reward=",self.num_reward)

                    #---------------leader selection-----------------
                    best_utility=-math.inf
                    pulled_arm=0
                    for arm in range(self.player.num_agent):
                        utility=self.est_fun(arm+1)-self.cum_cost[arm]
                        if utility>best_utility:
                            leader_arm=arm
                            best_utility=utility

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
                        sampled_thetas = []
                        for n in neighbor_arm:
                            # Draw a sample from the Beta(alpha_i, beta_i) distribution
                            sample = np.random.beta(self.alpha[n], self.beta[n])-self.cum_cost[n]
                            sampled_thetas.append(sample)
                        id_best_arm=np.argmax(sampled_thetas)
                        pulled_arm=neighbor_arm[id_best_arm]
                    
                #------------Turn arm into incentive-----------
                incentive=np.zeros((self.player.num_agent,))
                for n in range(pulled_arm+1):
                    id_agent=self.id_sorted_cost[n]
                    incentive[id_agent]=self.max_cost[id_agent]
                    
                self.best_incentive=np.zeros((self.player.num_agent,))
                for n in range(leader_arm+1):
                    id_agent=self.id_sorted_cost[n]
                    self.best_incentive[id_agent]=self.max_cost[id_agent]

        return incentive    

            
                