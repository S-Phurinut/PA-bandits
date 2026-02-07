import numpy as np
import scipy as sc
import math

class Logit():
    def __init__(self,num_agent,**model):
        self.num_agent=num_agent
        self.para_loc=np.ones((self.num_agent,))*0.5
        self.para_shape=np.ones((self.num_agent,))*0.001
        self.reset=True
        self.name="logit"
        self.para_type="loc-shape"
        
    def fit(self,X,Y):
        if self.reset:
            self.para_loc=np.ones((self.num_agent,))*0.5
            self.para_shape=np.ones((self.num_agent,))*0.001
            self.reset=False
        
        for agent in range(self.num_agent):
            best_loss=math.inf

            for i in range(4):
                if i==0:
                    x0=np.array([self.para_loc[agent],self.para_shape[agent]])
                    x0[0]=np.clip(x0[0],0,1)
                    x0[1]=np.clip(x0[1],0,10)
                else:
                    x0=np.random.random(2,)
            
                bnds = [(0,1),(0,10)]
                opt=sc.optimize.minimize(self.CE_loss,x0=x0,bounds=bnds,args=(X[:,agent].reshape(-1,),Y[:,agent].reshape(-1,)),tol=1E-12)
                loss=-opt.fun

                if opt.x is not None and loss<best_loss:
                    best_loss=loss
                    self.para_loc[agent]=opt.x[0]
                    self.para_shape[agent]=opt.x[1]

    def CE_loss(self,para,X,y):
        # Binary cross-entropy loss
        y_pred = 1/(1+np.exp(-(X-para[0])/para[1]))
        loss = np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return -loss

    def prob_accept(self,incentive): #prob of accept distribution 
        p=np.zeros((self.num_agent,))
        for agent in range(self.num_agent):
            p[agent]=1/(1+np.exp(-(incentive[agent]-self.para_loc[agent])/self.para_shape[agent]))
        return p