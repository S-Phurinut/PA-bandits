import numpy as np
import scipy as sc
import math

class Sigmoid_greedy():
    def __init__(self,type_arm,num_cost_learning=1,cost_alg='MultiBinSearch',**bandit_alg):
        self.type_arm=type_arm
        self.num_cost_learning=num_cost_learning
        self.cost_alg=cost_alg
    
        if 'is_cost_known' in bandit_alg:
            self.is_cost_known=bandit_alg['is_cost_known']
        else:
            self.is_cost_known=False
    
        self.reset=True

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
            self.Y=[]
            self.X=[]
            self.x0=np.array([1])
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
            self.Y.append(info['previous_reward'])
            self.X.append(n+1)
            if n>=0:
                self.sum_reward[n]+=info['previous_reward']
                self.num_reward[n]+=1

            
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

                #----------Extended Sigmoid regression---------
                
                optimise=sc.optimize.minimize(self.sigmoid_loss_function,x0=self.x0,bounds=[(0,math.inf)])
                para=optimise.x
                print("est_para=",para)
                self.x0=optimise.x

                #---------------Greedy alg-----------------
                Sig=lambda x : (1-np.exp(-x*para[0]))/(1+np.exp(-x*para[0]))
                best_utility=-math.inf
                pulled_arm=0
                for arm in range(self.player.num_agent):
                    utility=Sig(arm+1)-self.cum_cost[arm]
                    if utility>best_utility:
                        pulled_arm=arm
                        best_utility=utility
                
                #------------Turn arm into incentive-----------
                incentive=np.zeros((self.player.num_agent,))
                for n in range(pulled_arm+1):
                    id_agent=self.id_sorted_cost[n]
                    incentive[id_agent]=self.max_cost[id_agent]
                    
                self.best_incentive=np.array(incentive)
        return incentive    
    
    def sigmoid_loss_function(self,para):
        sum_loss=0
        for id_data in range(len(self.Y)):
            loss=(self.Y[id_data]-(1-np.exp(-self.X[id_data]*para[0]))/(1+np.exp(-self.X[id_data]*para[0])))**2
            sum_loss+=loss
        return sum_loss
            
                