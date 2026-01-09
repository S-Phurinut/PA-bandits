import numpy as np
import scipy as sc
import math

class Greedy_Binary_Agent(): 
    """Agent reacting to the leader agent with the best-response action and change their decision if higher utility is offered"""
    def __init__(self,num_agent=1):
        self.num_agent=num_agent
        self.type_agent="greedy"

    def update_data(self,agent_cost):
        self.cost=agent_cost

    def choose_action(self,incentive):
        action_list=[]
        for i in range(self.num_agent):
            if incentive[i]>=self.cost[i]:
                action_list.append(1)
            else:
                action_list.append(0)
        return {'action_list':action_list}

    