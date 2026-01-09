import numpy as np
import scipy as sc
import math

class Logit():
    def __init__(self,**parameter):
        self.para_loc=parameter['para_loc']
        self.para_shape=parameter['para_shape']
        self.num_agent=parameter['num_agent']

        if self.para_shape is not list:
            self.para_shape=np.ones((self.num_agent,))*self.para_shape
        
    def prob_accept(self,incentive): #prob of accept distribution 
        p=np.zeros((self.num_agent,))
        for agent in range(self.num_agent):
            p[agent]=1/(1+math.exp(-(incentive[agent]-self.para_loc[agent])/self.para_shape[agent]))
        return p
    
    def return_response(self,incentive):
        p=self.prob_accept(incentive)
        x=np.random.rand(self.num_agent,)
        response=x<=p
        response.astype(int)
        return response

