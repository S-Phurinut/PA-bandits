import numpy as np
import scipy
import scipy.signal
import scipy.signal.windows
from scipy.special import expit

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

    def fit(self, X, Y):
        if self.reset:
            self.u_mean=np.ones((self.num_agent,))*0.5
            self.s_mean=np.ones((self.num_agent,))*0.1
            self.reset = False

        with pm.Model() as model:
            u = pm.Uniform("u", 0, 1, shape=self.num_agent)          # (N,) location parameter
            s = pm.Exponential("s", lam=10, shape=self.num_agent)     # (N,) shape parameter

            z = (X - u) / s          # (T, N)
 
            p = pm.Deterministic("p", 1/(1+pm.math.exp(-z)))
            y = pm.Bernoulli("y", p=p, observed=Y)      # observed is (T, N)

            self.trace = pm.sample(draws=self.model['pymc_draws'], tune=self.model['pymc_draws'], chains=self.model['pymc_chains'], target_accept=self.model['pymc_target_accept'],cores=self.model['pymc_cores'])
        
        self.u_sample=self.trace.posterior["u"].stack(draws=("chain","draw")).values.T
        self.s_sample=self.trace.posterior["s"].stack(draws=("chain","draw")).values.T
        # self.s_sample=np.ones((self.num_agent,))*0.01 
    
        self.u_mean=self.u_sample.mean(axis=0) 
        self.s_mean=self.s_sample.mean(axis=0) 
        # self.s_mean=np.ones((self.num_agent,))*0.01 

        print("\n new data=",X[-1,:],Y[-1,:])
        print("posterior_mean=",self.u_mean,self.s_mean)


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

        p_samples = 1/(1+np.exp(-(incentive - u) / s))

        if len(p_samples.shape)==2:
            return p_samples.mean(axis=0).reshape(-1,)
        else :
            return p_samples.reshape(-1,)

        