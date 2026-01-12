import numpy as np
import scipy.signal
import scipy.signal.windows

# Patch older import name for PyMC compatibility
if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = scipy.signal.windows.gaussian
import pymc as pm
import pytensor.tensor as pt

import scipy as sc
import math
from poibin import PoiBin

import os
os.environ["PYTENSOR_FLAGS"] = "verbosity=low"


class TS_EU(): #EU with agent approx model
    def __init__(self,type_arm,**alg):
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

    def update_data(self,player):
        self.player=player

    def run(self,**info):
        if info['curr_round']==1 and self.reset:
            self.is_cost_learning_done=False
            self.is_model_training_done=False
            self.previous_c=np.ones((self.player.num_agent,))*0.5
            self.sum_reward=np.zeros((self.player.num_agent,))
            self.num_reward=np.zeros((self.player.num_agent,))

            if self.num_cost_learning=='log2T': 
                self.num_cost_learning=math.ceil(np.log2(info['max_round']))
            
            if self.cost_alg=="uniformly-space":
                self.cost_list=list(np.linspace(1E-12,1-(1E-12),int(self.num_cost_learning)))
            
            if self.alg['est_reward']=='TS' or self.alg['est_reward']=='incTS' or self.alg['est_reward']=='posterior-mean'  and self.type_arm=='participation-based':
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

            elif self.alg['est_reward']=="increasing-TS" or self.alg['est_reward']=="concave-TS":
                self.input=[]
                self.output=[]

            
        else:
            self.reset=False
            #----------Update agent model learning----------
            if info['curr_round']>1 and self.is_model_known==False:
                train_model=False
                if type(self.alg['model_training_appr'])==int:
                    if (info['curr_round']-1)%self.alg['model_training_appr']==0 or info['curr_round']<=self.alg['model_training_max_round_1step'] :
                        train_model=True
                else:
                    if self.alg['model_training_appr']=='once' and self.is_model_training_done==False : #and self.is_cost_learning_done==True:
                        train_model=True
                    elif self.alg['model_training_appr']=='T':
                        train_model=True

                if train_model:
                     self.alg['model'].fit(X=self.player.incentive_array[:(info['curr_round']-1),:],Y=self.player.agent_response_array[:(info['curr_round']-1),:])

                if info['curr_round']==self.num_cost_learning:
                    # print("logit model_para=",self.alg['model'].para_loc,self.alg['model'].para_shape)
                    self.is_model_training_done=True


            #-----------Update Reward Data---------
            n=np.sum(info['previous_agent_response'])-1
            if n>=0:
                self.sum_reward[n]+=info['previous_reward']
                self.num_reward[n]+=1

                if self.alg['est_reward']=='TS' or self.alg['est_reward']=='posterior-mean' or self.alg['est_reward']=='incTS':
                    if info['previous_reward']>0:
                        self.alpha[n]+=1
                    else:
                        self.beta[n]+=1

                elif self.alg['est_reward']=="increasing-TS" or self.alg['est_reward']=="concave-TS":
                    self.input.append(n+1)
                    self.output.append(info['previous_reward'])

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
                    if self.alg['est_reward']=='UCB-lattimore':
                        est_reward=self.UCBlat_value(info['max_round'])
                    elif self.alg['est_reward']=='UCB1':
                        est_reward=self.UCB1_value(info['curr_round'],info['max_round'])
                    elif self.alg['est_reward']=='TS':
                        est_reward=self.TS_value()
                    elif self.alg['est_reward']=='incTS':
                        est_reward=self.incTS_value(max_resampling_inc=self.alg['max_resampling_inc'])
                    elif self.alg['est_reward']=='posterior-mean':
                        est_reward=self.alpha/(self.alpha+self.beta)
                    elif self.alg['est_reward']=='increasing-TS':
                        refit_model=True
                        if self.alg['refit_step']=="adaptive-log10":
                            refit_step=max(int(10**(math.floor(np.log10(info['curr_round']))-1)),1)
                            if info['curr_round']%refit_step==0  or info['curr_round']<=self.alg['refit_max_round_1step'] :
                                refit_model=True
                            else:
                                refit_model=False
                        if type(self.alg['refit_step'])==int:
                            if (info['curr_round']-1)%int(self.alg['refit_step'])==0:
                                refit_model=True
                            else:
                                refit_model=False
                                
                        est_reward=self.structured_TS_value(type='increasing',refit_model=refit_model)
                    elif self.alg['est_reward']=='concave-TS':
                        refit_model=True
                        if self.alg['refit_step']=="adaptive-log10":
                            refit_step=max(int(10**(math.floor(np.log10(info['curr_round']))-1)),1)
                            if info['curr_round']%refit_step==0  or info['curr_round']<=self.alg['refit_max_round_1step'] :
                                refit_model=True
                            else:
                                refit_model=False
                        if type(self.alg['refit_step'])==int:
                            if (info['curr_round']-1)%int(self.alg['refit_step'])==0:
                                refit_model=True
                            else:
                                refit_model=False
                                
                        est_reward=self.structured_TS_value(type='increasing-concave',refit_model=refit_model)


                eps = 0 #float(1E-1)
                bnds = [(float(0 - eps), float(1 + eps)) for _ in range(self.player.num_agent)]
                best_EU=-math.inf
                best_cost=np.ones((self.player.num_agent,))
                
                self.model_para_sample=self.alg['model'].get_sample()
                print(self.model_para_sample)
                # print(np.array(self.alg['model'].prob_accept(np.array([0.32,0]),**self.model_para_sample)))
                # print(-self.EU_value(np.array([0.32,0]),est_reward))
                
                # print(np.array(self.alg['model'].prob_accept(np.array([0,0]),**self.model_para_sample)))
                # print(-self.EU_value(np.array([0,0]),est_reward))
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
                print("num reward=",self.num_reward)
                # print("logit model_para=",self.alg['model'].para_loc,self.alg['model'].para_shape)

        return best_cost   

    def EU_value(self,cost,reward):
        if self.is_model_known:
            p=np.array(self.player.agent_policy.prob_accept(cost))
        else:
            p=np.array(self.alg['model'].prob_accept(cost,**self.model_para_sample))
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
    
    def incTS_value(self,max_resampling_inc=1E7):
        if max_resampling_inc is None: max_resampling_inc=1E7
        resampling=True
        count_inc=-1
        while resampling:
            count_inc+-1
            sampled_thetas = []
            for n in range(self.player.num_agent):
                # Draw a sample from the Beta(alpha_i, beta_i) distribution
                sample = np.random.beta(self.alpha[n], self.beta[n])
                sampled_thetas.append(sample)

                if n>0 and sample<sampled_thetas[-1] and count_inc<=max_resampling_inc: #if reward is not incresing, restart
                    resampling=True
                    break
                else:
                    resampling=False

        return np.array(sampled_thetas)
    
    def structured_TS_value(self,type='increasing',refit_model=True):
        M = np.zeros((self.player.num_agent+1, self.player.num_agent+1), dtype=np.float64)  # exclude first arm
        for k in range(0, self.player.num_agent+1):
            for j in range(self.player.num_agent+1):
                if type=="increasing":
                    if k>=j:
                        M[k, j] = 1
                elif type=="increasing-concave":
                    M[k, j] = min(k+1,j+1)

        if refit_model:
            with pm.Model() as m:
                if self.alg['likelihood_model']== "cdf-exp":
                    w = pm.Uniform("w",lower=0, upper= 10, shape=self.player.num_agent+1)
                    # Latent concave function
                    f_all = pt.dot(pt.as_tensor_variable(M, dtype="float64"), w)
                    # Hazard-style link for flexible probabilities
                    p_all = pm.Deterministic("p_all", 1 - pm.math.exp(-f_all))
                    y = pm.Bernoulli("y", p=p_all[self.input], observed=self.output)

                    self.trace = pm.sample(draws=self.alg['pymc_draws'], tune=self.alg['pymc_tune'], target_accept=self.alg['pymc_target_accept'],
                                                    initvals={"w": np.full(self.player.num_agent+1, 0.01)}, cores=self.alg['pymc_cores'],chains=self.alg['pymc_chains'])
        self.p_samples = self.trace.posterior["p_all"].stack(draws=("chain","draw")).values
        p_all=self.p_samples[:,np.random.randint(0,self.p_samples.shape[1])][1:]

        return np.array(p_all)
    
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
    
            


                