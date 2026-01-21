import numpy as np
import scipy as sc
from tqdm import tqdm
import math
from poibin import PoiBin

class MABA_model():
    """One-principal Multi-agent Binary-action Model"""
    def __init__(self,principal_policy,agent_cost,num_agent,Reward_generator,**kwargs):
        self.policy=principal_policy
        self.num_agent=num_agent
        if type(agent_cost)==list:
            self.cost=agent_cost
        else:
            self.cost=list(np.ones((self.num_agent,))*agent_cost)
        self.reward_generator=Reward_generator

    def run_fixed_budget(self,max_round=1):
        incentive_array=np.zeros((max_round,self.num_agent))
        agent_response_array=np.zeros((max_round,self.num_agent))
        reward_array=np.zeros((max_round,))
        pred_best_reward_array=np.zeros((max_round,))

        for t in tqdm(range(max_round)):
            
            self.policy.update_data(self)
            if t==0:
                incentive=self.policy.run(curr_round=t+1,max_round=max_round)
            else:
                incentive=self.policy.run(curr_round=t+1,max_round=max_round,previous_agent_response=agent_response,previous_incentive=incentive,previous_reward=reward)
            
            agent_response=np.array(incentive)>=np.array(self.cost)
            agent_response=agent_response.astype(int)
            
            incentive_array[t,:]=incentive
            agent_response_array[t,:]=agent_response
            if self.policy.is_reward_known:
                reward,cost=self.reward_generator.get_true_mean(incentive=incentive,agent_action=agent_response)
            else:
                reward,cost=self.reward_generator.get_reward(incentive=incentive,agent_action=agent_response)
            reward_array[t]=reward-cost

            best_incentive=self.policy.best_incentive
            best_agent_response=np.array(best_incentive)>=np.array(self.cost)
            best_agent_response=best_agent_response.astype(int)
            avg_reward,avg_cost=self.reward_generator.get_true_mean(incentive=best_incentive,agent_action=best_agent_response)
            pred_best_reward_array[t]=avg_reward-avg_cost

            # print("At round =",t+1)
            # print("offer incentive=",incentive)
            # print("Agents's response=",agent_response)
            # print("Reward=",reward-cost)
        return reward_array,agent_response_array,incentive_array,pred_best_reward_array


    def optimal_solution(self):
        #solve the optimal solution from the reward function
        if self.reward_generator.type_reward=='participation-based':
            index_sort=np.argsort(np.array(self.cost),axis=None)
            sorted_cost=np.array(self.cost)[index_sort]
            cum_cost=np.cumsum(sorted_cost)
            best_net_reward=0
            optimal_number=0
            for n in range(self.num_agent):
                net_reward=self.reward_generator.mean[n]-cum_cost[n]
                if net_reward>best_net_reward:
                    optimal_number=n+1
                    best_net_reward=net_reward
            
            optimal_incentive=np.zeros((self.num_agent,))
            for n in range(optimal_number):
                id_agent=int(index_sort[n])
                optimal_incentive[id_agent]=self.cost[id_agent]
            optimal_utility=best_net_reward
            print("true_net_reward=",self.reward_generator.mean-cum_cost)
            print("optimal_num_agent=",optimal_number," with incentive=",optimal_incentive)

            return optimal_incentive, optimal_utility
        

class MASA_model():
    """One-principal Multi-agent Stochastic-action Model"""
    def __init__(self,principal_policy,agent_policy,Reward_generator,**kwargs):
        self.policy=principal_policy
        self.agent_policy=agent_policy #stochastic model
        self.num_agent=agent_policy.num_agent
        self.reward_generator=Reward_generator

    def run_fixed_budget(self,max_round=1):
        self.incentive_array=np.zeros((max_round,self.num_agent))
        self.agent_response_array=np.zeros((max_round,self.num_agent))
        self.reward_array=np.zeros((max_round,))

        for t in tqdm(range(max_round)):
            self.policy.update_data(self)
            if t==0:
                incentive=self.policy.run(curr_round=t+1,max_round=max_round)
            else:
                incentive=self.policy.run(curr_round=t+1,max_round=max_round,previous_agent_response=agent_response,previous_incentive=incentive,previous_reward=reward)
            
            agent_response=self.agent_policy.return_response(incentive)
            
            self.incentive_array[t,:]=incentive
            self.agent_response_array[t,:]=agent_response
            reward,cost=self.reward_generator.get_reward(incentive=incentive,agent_action=agent_response)
            self.reward_array[t]=reward-cost

        return self.reward_array,self.agent_response_array,self.incentive_array


    def optimal_solution(self):
        #solve the optimal solution from the reward function
        if self.reward_generator.type_reward=='participation-based':
            optimal_incentive=np.zeros((self.num_agent,))

            best_EU=-math.inf
            eps = 0 #float(1E-1)
            bnds = [(float(0 - eps), float(1 + eps)) for _ in range(self.num_agent)]

            for _ in range(32):
                x0=np.random.rand(self.num_agent,)
                opt=sc.optimize.minimize(self.EU_value,x0=x0,bounds=bnds,tol=1E-12)
                
                cost=opt.x 
                EU=-opt.fun

                if cost is not None and EU>best_EU:
                    best_EU=float(EU)
                    optimal_incentive=opt.x
                    optimal_utility=EU

            print("optimal_incentive=",optimal_incentive," with EU=",optimal_utility)
            return optimal_incentive, optimal_utility
        
    def EU_value(self,cost):
        p=np.array(self.agent_policy.prob_accept(cost))
        EU=0

        # if np.sum(np.array(cost)<=0)>=self.num_agent:
        #     EU+=0
        # else:
        if np.sum(p<=1E-4)>=self.num_agent:
            EU+=0
        elif np.sum(p>=1-1E-12)>=self.num_agent:
            EU+=self.reward_generator.mean[self.num_agent-1]-np.dot(p,cost)
        else:
            pb = PoiBin(p)
            num_offered_agent=int(np.sum(p>1E-6))
            num_guaranteed_agent=int(np.sum(p>=1-1E-6))
            for arm in range(max(num_guaranteed_agent,1),num_offered_agent+1): #math.comb is wrong
                EU+=self.reward_generator.mean[arm-1]*np.clip(pb.pmf(arm), 0, 1)
            EU+=-np.dot(p,cost)
        return -EU