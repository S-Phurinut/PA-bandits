import numpy as np
import scipy as sc
import math
import time

class Participation_based_Reward():
    def __init__(self,type_dist,num_agent,**dist_parameter):
        self.type_dist=type_dist
        self.dist_parameter=dist_parameter
        self.type_reward='participation-based'
        self.num_agent=num_agent

        if self.type_dist=='Bernoulli':
            self.r=self.dist_parameter['success_value']

            if self.dist_parameter['mean_prob']=='linear':
                self.contribution_level=self.dist_parameter['contribution_level']
                self.p= lambda x : x*self.contribution_level
                self.mean = np.arange(1,self.num_agent+1)*self.contribution_level*self.r
            elif self.dist_parameter['mean_prob']=='monotone-logistic':
                self.constant=self.dist_parameter['constant']
                self.p= lambda x : 1-math.exp(-self.constant*x)
                self.mean = (1-np.exp(-self.constant*np.arange(1,self.num_agent+1)))*self.r
            elif self.dist_parameter['mean_prob']=='polynomial':
                self.degree=self.dist_parameter['degree']
                if self.dist_parameter['constant'] is None:
                    self.constant=(1/self.num_agent)**self.degree
                else:
                    self.constant=self.dist_parameter['constant']
                self.p= lambda x : (x**self.degree)*self.constant
                self.mean = (np.arange(1,self.num_agent+1)**self.degree)*self.constant*self.r
            elif self.dist_parameter['mean_prob']=='bounded-log':
                self.base=self.dist_parameter['base']
                if self.dist_parameter['log_constant'] is not None:
                    self.a=self.dist_parameter['log_constant']
                else:
                    self.a=1

                if self.dist_parameter['curve_constant'] is not None:
                    self.b=self.dist_parameter['curve_constant']
                else:
                    self.b=1

                if self.base is None:
                    self.p= lambda x : self.a*np.log(x+1)/(self.b+self.a*np.log(x+1))
                    self.mean = (self.a*np.log(np.arange(1,self.num_agent+1)+1))/(self.b+self.a*np.log(np.arange(1,self.num_agent+1)+1))*self.r
                else:
                    self.p= lambda x : self.a*(np.log(x+1)/np.log(self.base))/(self.b+self.a*(np.log(x+1)/np.log(self.base)))
                    self.mean = (self.a*(np.log(np.arange(1,self.num_agent+1)+1)/np.log(self.base))/(self.b+self.a*(np.log(np.arange(1,self.num_agent+1)+1)/np.log(self.base))))*self.r
            else:
                if self.dist_parameter['mean_prob'] is not None:
                    self.p= lambda x : self.dist_parameter['mean_prob'][int(x-1)]
                    self.mean = np.array(self.dist_parameter['mean_prob'])*self.r
    
    def set_mean(self,mean=None,success_value=None):
        if mean is not None:
            m=mean
        else:
            m=self.dist_parameter['mean_prob']

        if success_value is not None:
            R=success_value
        else:
            R=self.dist_parameter['success_value']

        self.p= lambda x : m[int(x-1)]
        self.mean = np.array(m)*R


    def get_true_mean(self,incentive=None,agent_action=None):
        if agent_action is None:
            return 0,0
        else:
            if incentive is None: incentive=np.zeros((agent_action.shape[0],))
            cost=np.sum(np.array(incentive)*np.array(agent_action)) #spent cost
            num_agent=np.sum(agent_action)
            if num_agent==0:
                return 0,0
            else:
                return self.mean[num_agent-1],cost
            
    def get_reward(self,incentive=None,agent_action=None):
        if agent_action is None:
            return 0,0
        else:
            if incentive is None: incentive=np.zeros((agent_action.shape[0],))
            cost=np.sum(np.array(incentive)*np.array(agent_action)) #spent cost
            num_agent=np.sum(agent_action)
            if self.type_dist=='Bernoulli':
                if num_agent==0:
                    return 0,0
                else:
                    reward=np.random.binomial(n=1,p=self.p(num_agent),size=1)*self.r
                    return reward[0],cost




            








