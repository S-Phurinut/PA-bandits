import numpy as np
import scipy as sc
import math
from scipy.special import expit

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
        arm=9
        for agent in range(self.num_agent):
            best_loss=math.inf

            for i in range(4):
                if i==0:
                    x0=np.array([self.para_loc[agent],self.para_shape[agent]])
                    
                else:
                    x0=np.random.random(2,)/10

                
                eps=1e-12
                x0=np.clip(x0,eps,1-eps)
                bnds = [(eps,1-eps),(eps,1.0)]
                opt=sc.optimize.minimize(self.CE_loss,x0=x0,bounds=bnds,args=(np.array(X[:,agent],dtype=float),np.array(Y[:,agent],dtype=float)))
                loss=opt.fun

                # if agent==arm : print(x0,opt.x,loss)

                if opt.x is not None and loss<best_loss:
                    best_loss=loss
                    self.para_loc[agent]=opt.x[0]
                    self.para_shape[agent]=opt.x[1]
            
        self.para_loc=np.clip(self.para_loc, 1e-8, 1 - 1e-8)
        self.para_shape=np.clip(self.para_shape, 1e-8, 1)

        # print("u_MLE=",np.round(self.para_loc,4))
        # print("s_MLE=",np.round(self.para_shape,4))

        # print("opt_loss_arm1=",self.CE_loss([self.para_loc[arm],self.para_shape[arm]],X[:,arm],Y[:,arm]))
        # print("true_loss_arm1=",self.CE_loss([0.01*(arm+1),0.01],X[:,arm],Y[:,arm]))

    def CE_loss(self,para,X,y):
        # Binary cross-entropy loss
        y_pred = expit((X-para[0])/para[1])
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return -loss 

    def prob_accept(self,incentive): #prob of accept distribution 
        p=np.zeros((self.num_agent,))
        for agent in range(self.num_agent):
            p[agent]=expit((incentive[agent]-self.para_loc[agent])/self.para_shape[agent])
        return p