import numpy as np
import scipy.signal
import scipy.signal.windows
from scipy.stats import beta as beta_dist
import cvxpy as cp

# Patch older import name for PyMC compatibility
if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = scipy.signal.windows.gaussian
import pymc as pm
import pytensor.tensor as pt

import scipy as sc
import math
from poibin import PoiBin

class gEU_Gibbs_Monotone(): #EU with agent approx model
    def __init__(self,type_arm,
                 num_sweeps=10,random_scan=True,eps=1e-12,init_sweep_appr='once',**alg):
        self.type_arm=type_arm
        self.alg=alg
        self.num_cost_learning=self.alg['num_cost_learning']
        if "cost_alg" in self.alg :
            self.cost_alg=self.alg['cost_alg']
        else:
            self.cost_alg=None
            self.num_cost_learning=0

        if 'is_reward_known' in alg:
            self.is_reward_known=alg['is_reward_known']
        else:
            self.is_reward_known=False

        if 'is_model_known' in alg:
            self.is_model_known=alg['is_model_known']
        else:
            self.is_model_known=False

        self.num_optimiser=alg['num_optimiser']
        self.is_cost_learning_done=False
        self.is_model_training_done=False
        self.reset=True

        self.num_sweeps=num_sweeps #Number of full Gibbs sweeps
        self.random_scan=random_scan #If True, update indices in a random permutation each sweep
        self.eps=eps #Numerical safety margin for CDF inversion and interval clamping.
        self.init_sweep_appr=init_sweep_appr
        self.init_Gibb_sample=True

    def update_data(self,player):
        self.player=player

    def run(self,**info):
        if info['curr_round']==1 and self.reset:
            self.is_cost_learning_done=False
            self.is_model_training_done=False
            self.previous_c=np.ones((self.player.num_agent,))*0.5
            self.sum_reward=np.zeros((self.player.num_agent,))
            self.num_reward=np.zeros((self.player.num_agent,))
            self.init_Gibb_sample=True

            if self.num_cost_learning=='log2T': 
                self.num_cost_learning=math.ceil(np.log2(info['max_round']))
            elif self.num_cost_learning=='logT': 
                self.num_cost_learning=math.ceil(np.log(info['max_round']))
            elif self.num_cost_learning=='T1/2': 
                self.num_cost_learning=math.ceil(np.sqrt(info['max_round']))
            
            
            if self.cost_alg=="uniformly-space":
                self.cost_list=list(np.linspace(1E-12,1-(1E-12),int(self.num_cost_learning)))
            
            if self.alg['prior'] is not None:
                if self.alg['prior'][0]=='beta':
                    if self.alg['prior'][1][0]=='fixed':
                        self.alpha=np.ones((self.player.num_agent,))*self.alg['prior'][1][1]
                    elif self.alg['prior'][1][0]=='linear':
                        self.alpha=np.linspace(self.alg['prior'][1][1],self.alg['prior'][1][2],num=self.player.num_agent)

                    if self.alg['prior'][2][0]=='fixed':
                        self.beta=np.ones((self.player.num_agent,))*self.alg['prior'][2][1] 
                    elif self.alg['prior'][2][0]=='linear':
                        self.alpha=np.linspace(self.alg['prior'][2][1],self.alg['prior'][2][2],num=self.player.num_agent)
            else:
                self.alpha=np.ones((self.player.num_agent,))*1
                self.beta=np.ones((self.player.num_agent,))


            
        else:
            self.reset=False
            #----------Update agent model learning----------
            if info['curr_round']>1 and self.is_model_known==False:
                train_model=False
                if type(self.alg['model_training_appr'])==int:
                    if info['curr_round']-1%self.alg['model_training_appr']==0:
                        train_model=True
                else:
                    if self.alg['model_training_appr']=='once' and self.is_model_training_done==False : #and self.is_cost_learning_done==True:
                        train_model=True
                    elif self.alg['model_training_appr']=='T':
                        train_model=True

                if train_model:
                     self.alg['model'].fit(X=self.player.incentive_array[:(info['curr_round']-1),:],Y=self.player.agent_response_array[:(info['curr_round']-1),:])

                if info['curr_round']==self.num_cost_learning:
                    print("logit model_para=",self.alg['model'].para_loc,self.alg['model'].para_shape)
                    self.is_model_training_done=True

            #-----------Update Reward Data---------
            n=np.sum(info['previous_agent_response'])-1
            if n>=0:
                self.sum_reward[n]+=info['previous_reward']
                self.num_reward[n]+=1

                if info['previous_reward']>0:
                    self.alpha[n]+=1
                else:
                    self.beta[n]+=1


        if info['curr_round']<=self.num_cost_learning:
            if info['curr_round']==self.num_cost_learning:
                self.is_cost_learning_done=True

            if self.cost_alg=="uniformly-space":
                best_cost=np.ones((self.player.num_agent,))*self.cost_list[int(info['curr_round']-1)]
            elif self.cost_alg=="D-optimal":
                best_cost=np.clip(self.D_optimal(),0,1)
            elif self.cost_alg=="A-optimal":
                pass
        else:  
            self.is_cost_learning_done=True      
            #====================Contracting Part============================
            if self.type_arm=='participation-based':

                #---------------Reward estimator ----------------- 
                if self.is_reward_known:
                    est_reward=np.array(self.player.reward_generator.mean)
                else:
                    if self.init_sweep_appr=="T":
                        self.init_Gibb_sample=True

                    #------------init Gibbs sample by Monotone regression-----------------
                    if self.init_Gibb_sample:
                        n=self.player.num_agent+1
                        f = cp.Variable(n)
                        # constraints
                        cons = []
                        cons += [ f[0]==0 ]
                        # isotonic: f[i+1] >= f[i]
                        cons += [f[i+1] - f[i] >= 0 for i in range(n-1)]
                        # maximum prob <=1
                        cons += [f[n-1] <= 1]

                        # objective: least squares

                        TS_sample = np.zeros((self.player.num_agent+1,))
                        for n in range(self.player.num_agent):
                            # Draw a sample from the Beta(alpha_i, beta_i) distribution
                            TS_sample[n+1] = np.random.beta(self.alpha[n], self.beta[n])

                        is_sample_exist= np.ones((self.player.num_agent))
                        for n in range(self.player.num_agent):
                            if self.num_reward[n]==0:
                                is_sample_exist[n]=0
                        obj = cp.Minimize(cp.sum_squares(cp.multiply(np.concatenate(([1],is_sample_exist)),(TS_sample - f))))
                    
                    
                        prob = cp.Problem(obj, cons)
                        prob.solve(solver=cp.OSQP)

                        self.gibb_sample=np.array(f.value)[1:]
                        self.init_Gibb_sample=False


                    for _ in range(self.num_sweeps):
                        if self.random_scan:
                            order = np.random.permutation(self.player.num_agent)
                        else:
                            order = range(self.player.num_agent)
                        
                        for i in order:
                            L = 0.0 if i == 0 else self.gibb_sample[i - 1]
                            U = 1.0 if i == self.player.num_agent - 1 else self.gibb_sample[i + 1]

                            # Clamp interval into [0,1] and ensure nonempty numerically
                            L = float(np.clip(L, 0.0, 1.0))
                            U = float(np.clip(U, 0.0, 1.0))
                            if U < L:
                                # This should not happen if mu is monotone, but guard anyway.
                                L, U = U, L

                            # If interval is essentially a point, just set to midpoint
                            if U - L <=self.eps:
                                self.gibb_sample[i] = 0.5 * (L + U)
                                continue

                            a_i, b_i = float(self.alpha[i]), float(self.beta[i])

                            # Truncated Beta via inverse CDF sampling:
                            # u ~ Uniform(F(L), F(U)), mu_i = F^{-1}(u)
                            FL = beta_dist.cdf(L, a_i, b_i)
                            FU = beta_dist.cdf(U, a_i, b_i)

                            # Numerical safety: keep within [0,1] and avoid FL==FU issues
                            FL = float(np.clip(FL, 0.0, 1.0))
                            FU = float(np.clip(FU, 0.0, 1.0))

                            if FU - FL <=self.eps:
                                self.gibb_sample[i] = 0.5 * (L + U)
                                continue

                            u = np.random.uniform(FL +self.eps, FU -self.eps) if (FU - FL) > 2 *self.eps else np.random.uniform(FL, FU)
                            self.gibb_sample[i] = float(beta_dist.ppf(u, a_i, b_i))

                            # Final clamp (rarely needed) and enforce local monotonicity
                            if self.gibb_sample[i] < L:
                                self.gibb_sample[i] = L
                            elif self.gibb_sample[i] > U:
                                self.gibb_sample[i] = U
                    est_reward=np.array(self.gibb_sample) 

                eps = 0 #float(1E-1)
                bnds = [(float(0 - eps), float(1 + eps)) for _ in range(self.player.num_agent)]
                best_EU=-math.inf
                best_cost=np.ones((self.player.num_agent,))

                for i in range(0,self.num_optimiser):
                    if i==0:
                        x0=self.previous_c
                    else:
                        x0=np.clip(np.random.rand(self.player.num_agent,),0.1,0.9)

                    opt=sc.optimize.minimize(self.EU_value,x0=x0,bounds=bnds,args=(est_reward),tol=1E-12) #,method="SLSQP"
                    cost=opt.x
                    EU=-opt.fun
                    if cost is not None and EU>best_EU:
                        best_EU=float(EU)
                        best_cost=np.array(cost)

                self.previous_c=np.array(best_cost)
            if info['curr_round']%1000==0: print("num reward=",self.num_reward)
            
            if info['curr_round']==info['max_round']:
                print("final incentive=",np.round(best_cost,4))
                print("logit model_para=",self.alg['model'].para_loc,self.alg['model'].para_shape)

        return best_cost   

    def EU_value(self,cost,reward):
        if self.is_model_known:
            p=np.array(self.player.agent_policy.prob_accept(cost))
        else:
            p=np.array(self.alg['model'].prob_accept(cost))
        EU=0

        # if np.sum(np.array(cost)<=0)>=self.player.num_agent:
        #     EU+=0
        # else:
        if np.sum(p<=1E-4)>=self.player.num_agent:
            EU+=0
        elif np.sum(p>=1-1E-12)>=self.player.num_agent:
            EU+=reward[self.player.num_agent-1]-np.dot(p,cost)
        else:
            pb = PoiBin(p)
            num_offered_agent=int(np.sum(p>1E-6))
            num_guaranteed_agent=int(np.sum(p>=1-1E-6))
            for arm in range(max(num_guaranteed_agent,1),num_offered_agent+1): #math.comb is wrong
                EU+=reward[arm-1]*np.clip(pb.pmf(arm), 0, 1)
            EU+=-np.dot(p,cost)
        return -EU 

    def D_optimal(self):
        if self.reset:
            self.count=0

        if self.count%2==0:
            if self.alg['model'].name=="logit" :
                if self.alg['model'].para_type=="loc-shape":
                    self.x_mid=self.alg['model'].para_loc
                    self.b=1/self.alg['model'].para_shape
            elif self.alg['model'].name=="bayes-logit":
                if self.alg['model'].para_type=="loc-shape":
                    self.x_mid=self.alg['model'].u_mean
                    self.b=1/self.alg['model'].s_mean
            x=self.x_mid+(1.543/self.b)
        else:
            x=self.x_mid-(1.543/self.b)
        self.count+=1
        return x




    
    def A_optimal(self):
        return 1
    
            


                