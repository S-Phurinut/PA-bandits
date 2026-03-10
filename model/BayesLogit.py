import numpy as np
import scipy
import scipy.signal
import scipy.signal.windows
import math
from scipy.special import expit
from concurrent.futures import ProcessPoolExecutor, as_completed

# Patch older import name for PyMC compatibility
if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = scipy.signal.windows.gaussian

import pymc as pm
import os
os.environ["PYTENSOR_FLAGS"] = (
    "verbosity=low,"
    "exception_verbosity=high,"
    "linker=py,"
    "optimizer=fast_compile"
)

class BayesLogit:
    def __init__(self, num_agent, **model):
        self.num_agent = num_agent
        self.reset = True
        self.name = "bayes-logit"
        self.para_type = "loc-shape"
        self.u_mean=np.ones((self.num_agent,))*0.5
        self.s_mean=np.ones((self.num_agent,))*0.1
        
        self.u_sample=self.u_mean
        self.s_sample=self.s_mean
        self.model=model

        if 'fitting_appr' not in self.model:
            self.model['fitting_appr']='all_agents'
        if "buffer" not in self.model:
            self.buffer=0
        else:
            if self.model['buffer'] is None:
                self.buffer=0
            else:
                self.buffer=self.model['buffer']
        self.previous_data_id=0

    def update_data(self,X,Y):
        self.X=X
        self.Y=Y
        if self.reset:
            self.u_mean=np.ones((self.num_agent,))*0.5
            self.s_mean=np.ones((self.num_agent,))*0.1

            # if self.model['fitting_appr']=='one_agent':
            self.previous_data_id=0
            self.X_ind=[]
            self.Y_ind=[]
            for _ in range(self.num_agent):
                self.X_ind.append([])
                self.Y_ind.append([])
            self.reset = False
        
        for agent in range(self.num_agent):
            for i in range(self.previous_data_id,int(X.shape[0])):
                if X[i,agent]>=self.buffer:
                    self.X_ind[agent].append(X[i,agent])
                    self.Y_ind[agent].append(Y[i,agent])
        self.previous_data_id=int(X.shape[0])

    def fit(self):
        if self.model['fitting_appr']=='all_agents': #not consider cost buffer
            #Least Square for Warm-start
            self.u_LS=np.array(self.u_mean)
            self.s_LS=np.array(self.s_mean)
            for agent_id in range(self.num_agent):
                self.u_LS[agent_id],self.s_LS[agent_id]=self.get_MLE_estimator(agent_id)
            self.s_LS=np.clip(self.s_LS,1e-3,2)
            print("MLE est=\n",self.u_LS,"\n",self.s_LS)
            

            with pm.Model() as model:
                u = pm.Uniform("u", 0, 1, shape=self.num_agent)          # (N,) location parameter
                s = pm.Exponential("s", lam=self.model['shape_prior'][1], shape=self.num_agent)     # (N,) shape parameter

                z = (self.X - u) / s          # (T, N)
    
                p = pm.Deterministic("p", 1/(1+pm.math.exp(-z)))
                y = pm.Bernoulli("y", p=p, observed=self.Y)      # observed is (T, N)
 
                self.trace = pm.sample(draws=self.model['pymc_draws'], tune=self.model['pymc_tune'],
                                    chains=self.model['pymc_chains'], target_accept=self.model['pymc_target_accept'],
                                    cores=self.model['pymc_cores']) #,init="adapt_diag",initvals={"u": self.u_LS, "s": self.s_LS},)
            
            self.u_sample=self.trace.posterior["u"].stack(draws=("chain","draw")).values.T
            self.s_sample=self.trace.posterior["s"].stack(draws=("chain","draw")).values.T
            # self.s_sample=np.ones((self.num_agent,))*0.01 
        
            self.u_mean=self.u_sample.mean(axis=0) 
            self.s_mean=self.s_sample.mean(axis=0) 
            # self.s_mean=np.ones((self.num_agent,))*0.01 

            # print("\n new data=",X[-1,:],Y[-1,:])
            print("posterior_mean=",self.u_mean,self.s_mean)
            print("num_sample",self.u_sample.shape)
        
        elif type(self.model['fitting_appr'])==int: 
            #Least Square for Warm-start
            self.u_LS=np.array(self.u_mean)
            self.s_LS=np.array(self.s_mean)
            for agent_id in range(self.num_agent):
                self.u_LS[agent_id],self.s_LS[agent_id]=self.get_MLE_estimator(agent_id)
            self.s_LS=np.clip(self.s_LS,1e-3,2)
            print("MLE est=\n",self.u_LS,"\n",self.s_LS)
            
            count_init=0
            count_end=0
            self.u_sample=np.zeros((int(self.model['pymc_draws']*self.model['pymc_chains']),self.num_agent))
            self.s_sample=np.zeros((int(self.model['pymc_draws']*self.model['pymc_chains']),self.num_agent))
            for t in range(math.ceil(self.num_agent/self.model['fitting_appr'])):
                if t==0:
                    if int(self.num_agent%self.model['fitting_appr'])==0:
                        num_para=int(self.model['fitting_appr'])
                    else:
                        num_para=int(self.num_agent%self.model['fitting_appr'])
                else:
                    num_para=int(self.model['fitting_appr'])
                count_end+=num_para
                with pm.Model() as model:
                    u = pm.Uniform("u", 0, 1, shape=num_para)          # (N,) location parameter
                    s = pm.Exponential("s", lam=self.model['shape_prior'][1], shape=num_para)     # (N,) shape parameter

                    z = (self.X[:,count_init:count_end] - u) / s          # (T, N)
        
                    p = pm.Deterministic("p", 1/(1+pm.math.exp(-z)))
                    y = pm.Bernoulli("y", p=p, observed=self.Y[:,count_init:count_end])      # observed is (T, N)
    
                    self.trace = pm.sample(draws=self.model['pymc_draws'], tune=self.model['pymc_tune'],
                                        chains=self.model['pymc_chains'], target_accept=self.model['pymc_target_accept'],
                                        cores=self.model['pymc_cores']) #,init="adapt_diag",initvals={"u": self.u_LS, "s": self.s_LS},)
            
                self.u_sample[:,count_init:count_end]=self.trace.posterior["u"].stack(draws=("chain","draw")).values.T
                self.s_sample[:,count_init:count_end]=self.trace.posterior["s"].stack(draws=("chain","draw")).values.T
                # self.s_sample=np.ones((self.num_agent,))*0.01 
            
                self.u_mean[count_init:count_end]=self.u_sample[:,count_init:count_end].mean(axis=0) 
                self.s_mean[count_init:count_end]=self.s_sample[:,count_init:count_end].mean(axis=0) 
                # self.s_mean=np.ones((self.num_agent,))*0.01 
                count_init=count_init+num_para

            # print("\n new data=",X[-1,:],Y[-1,:])
            print("posterior_mean=",self.u_mean,self.s_mean)
            print("num_sample",self.u_sample.shape)

        elif self.model['fitting_appr']=='one_agent':
            self.u_sample = np.zeros((int(self.model['pymc_draws']*self.model['pymc_chains']),self.num_agent))
            self.s_sample = np.zeros((int(self.model['pymc_draws']*self.model['pymc_chains']),self.num_agent))

            with ProcessPoolExecutor(max_workers=self.model['max_workers']) as pool:
                futures = [
                    pool.submit(self.fit_one_agent, i, self.X_ind[i], self.Y_ind[i])
                    for i in range(self.num_agent)
                ]

                for fut in as_completed(futures):
                    agent_id, u_sample, s_sample = fut.result()
                    self.u_sample[:,agent_id] = u_sample.reshape(-1,)
                    self.s_sample[:,agent_id] = s_sample.reshape(-1,)
                    print(f"Finished agent {agent_id}")

            self.u_mean=self.u_sample.mean(axis=0) 
            self.s_mean=self.s_sample.mean(axis=0) 
            # self.s_mean=np.ones((self.num_agent,))*0.01 

            # print("\n new data=",self.X[-1,:],self.Y[-1,:])
            print("posterior_mean=",self.u_mean,self.s_mean)
            print("num_sample",self.u_sample.shape)

    def get_MLE_estimator(self,agent_id):
        if self.previous_data_id==0:
            return self.u_mean[agent_id],self.s_mean[agent_id]
        else:
            best_loss=math.inf
            u_LS=self.u_mean[agent_id]
            s_LS=self.s_mean[agent_id]

            for i in range(4):
                if i==0:
                    x0=np.array([u_LS,s_LS])
                else:
                    x0=np.random.random(2,)/10

                bnds = [(0,1),(-math.inf,math.inf)]
                opt=scipy.optimize.minimize(self.CE_loss,x0=x0,bounds=bnds,args=(np.array(self.X_ind[agent_id]),np.array(self.Y_ind[agent_id])),tol=1E-12)
                loss=opt.fun

                if opt.x is not None and loss<best_loss:
                    best_loss=loss
                    u_LS=opt.x[0]
                    s_LS=opt.x[1]

            u_LS=np.clip([u_LS], 1e-8, 1 - 1e-8)[0]
            s_LS=np.clip([s_LS], 1e-8,1E4)[0]
            return u_LS,s_LS



    def fit_one_agent(self,agent_id,X,Y):
        if len(Y)==0:
            u_sample=np.random.uniform(low=0,high=1,size=int(self.model['pymc_draws']*self.model['pymc_chains']))
            s_sample=np.random.exponential(scale=self.model['shape_prior'][1],size=int(self.model['pymc_draws']*self.model['pymc_chains']))
        else:

            #Least Square for Warm-start
            # u_LS, s_LS= self.get_MLE_estimator(agent_id)
            # print("warm-start",[agent_id,u_LS,s_LS])
            # # print(X)

            with pm.Model() as model:
                u = pm.Uniform("u", 0, 1)          # location parameter
                s = pm.Exponential("s", lam=self.model['shape_prior'][1])     # shape parameter

                z = (np.array(X) - u) / s          # (T, N)

                p = pm.Deterministic("p", 1/(1+pm.math.exp(-z)))
                y = pm.Bernoulli("y", p=p, observed=np.array(Y))      # observed is (T, N)

                trace = pm.sample(draws=self.model['pymc_draws'], tune=self.model['pymc_tune'],
                                chains=self.model['pymc_chains'], target_accept=self.model['pymc_target_accept'],
                                cores=self.model['pymc_cores'],) #init="adapt_diag",initvals={"u": u_LS, "s": s_LS},)
                u_sample=trace.posterior["u"].stack(draws=("chain","draw")).values.T
                s_sample=trace.posterior["s"].stack(draws=("chain","draw")).values.T
        return agent_id, u_sample, s_sample

             
    def get_sample(self):
        if self.reset==False:
            u=self.u_sample[np.random.randint(0,self.u_sample.shape[0]),:]
            s=self.s_sample[np.random.randint(0,self.s_sample.shape[0]),:]
        else:
            u=self.u_sample
            s=self.s_sample
            
        # s=np.ones((self.num_agent,))*0.01
        return {"loc":u,"shape":s}
    
    def prob_accept(self, incentive,**model_para):
        if "loc" in model_para:
            u=np.array(model_para['loc'])
        else: 
            u=self.u_sample
        
        if "shape" in model_para:
            s=np.array(model_para['shape'])
        else: 
            s=self.s_sample

        p_samples = (1/(1+np.exp(-(incentive - u) / s)))*(incentive>=self.buffer)

        if len(p_samples.shape)==2:
            return p_samples.mean(axis=0).reshape(-1,)
        else :
            return p_samples.reshape(-1,)

    def CE_loss(self,para,X,y):
        # Binary cross-entropy loss
        y_pred = expit((X-para[0])/para[1])
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return -loss 