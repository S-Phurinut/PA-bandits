import numpy as np
import scipy as sc
from scipy.stats import beta as beta_dist
import cvxpy as cp
import math

class TS_Slice_Concave():
    def __init__(self,type_arm,num_cost_learning=1,cost_alg='MultiBinSearch',
                 num_sweeps=10,sweep_scan="linear",eps=1e-12,init_sweep_appr="once",**bandit_alg):
        self.type_arm=type_arm
        self.num_cost_learning=num_cost_learning
        self.cost_alg=cost_alg
        self.num_sweeps=num_sweeps #Number of full Gibbs sweeps
        self.sweep_scan=sweep_scan #If True, update indices in a random permutation each sweep
        self.eps=eps #Numerical safety margin for CDF inversion and interval clamping.
        self.init_sweep_appr=init_sweep_appr

        if 'is_cost_known' in bandit_alg:
            self.is_cost_known=bandit_alg['is_cost_known']
        else:
            self.is_cost_known=False

        if 'is_reward_known' in bandit_alg:
            self.is_reward_known=bandit_alg['is_reward_known']
        else:
            self.is_reward_known=False
        
        self.bandit_alg=bandit_alg
        self.init_Gibb_sample=True
        self.reset=True
        self.weighted_LS=bandit_alg['init_sweep_weighted_LS']

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
            self.init_Gibb_sample=True
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
                    if info['curr_round']%500==0:  
                        print("num reward=",self.num_reward)

                    # if self.init_sweep_appr=="T":
                    #     self.init_Gibb_sample=True

                    # #------------init Gibbs sample by Monotone regression-----------------
                    # if self.init_Gibb_sample:
                    #     n=self.player.num_agent
                    #     f = cp.Variable(n)
                    #     # constraints
                    #     cons = []
                    #     cons += [ f[0]>=0 ]
                    #     # isotonic: f[i+1] >= f[i]
                    #     cons += [f[i+1] - f[i] >= 0 for i in range(n-1)]
                    #     # concave: second differences <= 0
                    #     cons += [f[1] - 2*f[0]  <= 0 ]
                    #     cons += [f[i+2] - 2*f[i+1] + f[i] <= 0 for i in range(n-2)]
                    #     # maximum prob <=1
                    #     cons += [f[n-1] <= 1]

                    #     # objective: least squares
                    #     weights = np.zeros((n,))
                    #     # weights[0] = 1.0
                    #     TS_sample = np.zeros((n,))
                    #     for n in range(self.player.num_agent):
                    #         # Draw a sample from the Beta(alpha_i, beta_i) distribution
                    #         TS_sample[n] = np.random.beta(self.alpha[n], self.beta[n])

                    #         var = (self.alpha[n] * self.beta[n]) / (((self.alpha[n] + self.beta[n]) ** 2) * (self.alpha[n] + self.beta[n] + 1))

                    #         # inverse-variance weight
                    #         weights[n] = 1.0 / max(var, 1e-12)

                    #     # normalize weights for numerical stability
                    #     weights = weights / np.max(weights)
                    #     if self.weighted_LS=="variance":
                    #         obj = cp.Minimize(cp.sum(cp.multiply(weights, cp.square(TS_sample - f))))
                    
                    
                    #     prob = cp.Problem(obj, cons)
                    #     prob.solve(solver=cp.OSQP)

                    #     self.gibb_sample=np.array(f.value)
                    #     if info['curr_round']%500==0:   print("init_MCLS=",self.gibb_sample)
                    #     # print("init Gibb sample=",self.gibb_sample)
                    #     self.init_Gibb_sample=False

                    self.gibb_sample=sample_concave_once(self.alpha-1,self.alpha+self.beta-2)
                            


                    if info['curr_round']%500==0:   
                        print("s/m=",(self.alpha-1)/(self.alpha+self.beta-2))
                        print("m=",self.alpha+self.beta-2)
                        print("gibb_sample=",self.gibb_sample)

                    #---------------Greedy alg-----------------
                    best_utility=-math.inf
                    pulled_arm=0
                    for arm in range(self.player.num_agent):
                        utility=self.gibb_sample[arm]-self.cum_cost[arm]
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

            
def u_to_f(u):
    delta = np.cumsum(u[::-1])[::-1]
    return np.cumsum(delta)

def log_post(u, s, m, eps=1e-12):
    n = len(u)
    j = np.arange(1, n + 1)

    if np.any(u < 0):
        return -np.inf
    if np.dot(j, u) >= 1:
        return -np.inf

    f = u_to_f(u)
    if np.any(f <= 0) or np.any(f >= 1):
        return -np.inf

    f = np.clip(f, eps, 1 - eps)
    return np.sum(s * np.log(f) + (m - s) * np.log(1 - f))

def slice_update(u, idx, s, m, w=0.02):
    n = len(u)
    j = np.arange(1, n + 1)

    current = u[idx]
    logy = log_post(u, s, m) + np.log(np.random.rand())

    other_sum = np.dot(j, u) - (idx + 1) * current
    upper = max(0.0, (1.0 - other_sum) / (idx + 1))

    L = max(0.0, current - w * np.random.rand())
    R = min(upper, L + w)
    L = max(0.0, R - w)

    while True:
        x = np.random.uniform(L, R)
        u_try = u.copy()
        u_try[idx] = x

        if log_post(u_try, s, m) >= logy:
            return u_try

        if x < current:
            L = x
        else:
            R = x

def sample_concave_once(s, m, sweeps=50, u_init=None):
    n = len(s)

    if u_init is None:
        u = np.zeros(n)
        u[0] = 0.5 / n
    else:
        u = u_init.copy()

    for _ in range(sweeps):
        for idx in np.random.permutation(n):
            u = slice_update(u, idx, s, m)

    return u_to_f(u)