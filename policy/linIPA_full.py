import numpy as np
import scipy as sc
import math

class linIPA_full():
    def __init__(self,type_arm,num_cost_learning=1,cost_alg='MultiBinSearch',**bandit_alg):
        self.type_arm=type_arm
        self.num_cost_learning=num_cost_learning
        self.cost_alg=cost_alg
        self.bandit_alg=bandit_alg
    
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
            if self.num_cost_learning=='log2T': 
                self.num_cost_learning=math.ceil(np.log2(info['max_round']))
            elif self.num_cost_learning=='num_agent':
                self.num_cost_learning=int(self.player.num_agent)
            self.reset=False
        else:

            #----------Update cost learning----------
            for n in range(self.player.num_agent):
                if info['previous_agent_response'][n]==0:
                    self.min_cost[n]=info['previous_incentive'][n]
                elif info['previous_agent_response'][n]==1:
                    self.max_cost[n]=info['previous_incentive'][n]
             #-----------Update Reward Data---------
            k=np.sum(info['previous_agent_response'])-1
            if k>=0:
                for n in range(self.player.num_agent):
                    self.sum_reward[n]+=info['previous_reward']*(n+1)/(k+1)
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
                    print("start bandit alg")
                    print("best_estimated_cost=",self.cum_cost)

                #---------------bandit alg-----------------
                if self.bandit_alg['bandit_alg']=='UCB-lattimore':
                    pulled_arm=self.UCBlat_subroutine(info['max_round'])
                elif self.bandit_alg['bandit_alg']=='UCB1':
                    pulled_arm=self.UCB1_subroutine(info['curr_round'])
                
                #------------Turn arm into incentive-----------
                incentive=np.zeros((self.player.num_agent,))
                for n in range(pulled_arm+1):
                    id_agent=self.id_sorted_cost[n]
                    incentive[id_agent]=self.max_cost[id_agent]

                #----------Predict best arm---------
                est_mean_reward=self.sum_reward/self.num_reward
                est_mean_reward=np.min(np.concatenate((est_mean_reward.reshape(-1,1),np.ones((self.player.num_agent,1))),axis=0),axis=0)-self.cum_cost
                # print("\n est_mean_reward=",est_mean_reward)
                best_arm=np.argmax(est_mean_reward)
                self.best_incentive=np.zeros((self.player.num_agent,))
                if est_mean_reward[best_arm]<math.inf:
                    self.best_incentive=np.zeros((self.player.num_agent,))
                    for n in range(best_arm+1):
                        id_agent=self.id_sorted_cost[n]
                        self.best_incentive[id_agent]=self.max_cost[id_agent]
    
        return incentive      

    def UCBlat_subroutine(self,max_round):
        UCB=np.zeros((self.player.num_agent,))
        for n in range(self.player.num_agent):
            if self.num_reward[n]==0:
                UCB[n]=1E6
            else:
                UCB[n]=np.min([self.sum_reward[n]/self.num_reward[n],1])+2*np.sqrt(np.log(max_round)/self.num_reward[n])-self.cum_cost[n]
        # print("UCB=",UCB)
        best_arm=int(np.argmax(UCB))
        return best_arm
            
    def UCB1_subroutine(self,round):
        UCB=np.zeros((self.player.num_agent,))
        for n in range(self.player.num_agent):
            if self.num_reward[n]==0:
                UCB[n]=1E6
            else:
                UCB[n]=np.min([self.sum_reward[n]/self.num_reward[n],1])+np.sqrt(2*np.log(round)/self.num_reward[n])-self.cum_cost[n]
        best_arm=int(np.argmax(UCB))
        return best_arm