import numpy as np
import scipy as sc
import math
import time

class MABA_Principal():
    def __init__(self,policy,**kwargs):
        self.policy=policy    
    
    def update_data(self,**info):
        if 'max_round' in info: self.T=info['max_round']
    
    def choose_action(self,round,previous_agent_response=None,previous_reward=None):
        if round==0:
            self.policy.reset=True
        
        self.policy.update_data(self) 
        incentive=self.policy.run(round=round,previous_agent_response=previous_agent_response,previous_reward=previous_reward)

        return {'incentive': incentive}