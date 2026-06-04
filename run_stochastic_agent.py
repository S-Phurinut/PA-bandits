import time
import wandb
import hydra
from omegaconf import OmegaConf
import pandas as pd
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings

warnings.warn("This warning will be hidden")

import os
os.environ["WANDB__SERVICE"] = "wandb-core"
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["PYTENSOR_FLAGS"] = (
    "compiledir=/tmp/pytensor_unique,"
    "verbosity=low"
    "linker=py,"
    "optimizer=fast_compile"
)


@hydra.main(version_base=None,config_path="config", config_name="main2")
def main(config):

    #Instantitate wandb and log info of configurations 
    log_config=pd.json_normalize(OmegaConf.to_object(config),sep='/')
    log_config=log_config.to_dict(orient='records')[0]
    wandb.init(project ="PA_bandits",config=log_config, mode=config.wandb_mode, entity="s_phurinut", settings=wandb.Settings(init_timeout=120))
    
    M=config.num_sim
    T=config.max_round
    N=config.agent.num_agent

    regret_array=np.zeros((M,T))
    EU_regret_array=np.zeros((M,T))
    # simple_regret_array=np.zeros((M,T))

    offered_incentive_array=np.zeros((M,T,N))
    l1_dist_incentive_array=np.zeros((M,T))
    l2_dist_incentive_array=np.zeros((M,T))
    linf_dist_incentive_array=np.zeros((M,T))

    total_incentive_array=np.zeros((M,T))
    l1_dist_total_incentive_array=np.zeros((M,T))

    l1_dist_para_loc_array=np.zeros((M,T))
    l2_dist_para_loc_array=np.zeros((M,T))
    linf_dist_para_loc_array=np.zeros((M,T))
    l1_dist_para_shape_array=np.zeros((M,T))
    l2_dist_para_shape_array=np.zeros((M,T))
    linf_dist_para_shape_array=np.zeros((M,T))

    min_var_loc_array=np.zeros((M,T))
    max_var_loc_array=np.zeros((M,T))
    avg_var_loc_array=np.zeros((M,T))
    min_var_shape_array=np.zeros((M,T))
    max_var_shape_array=np.zeros((M,T))
    avg_var_shape_array=np.zeros((M,T))

    try:
        #Set environment of the simulations from YAML files
        for sim in range(0,M):
            seed=sim+config.setting['init_seed_id']
            np.random.seed(seed=seed)
            print("sim=",sim,' game_seed=',seed)

            Reward_generator=hydra.utils.instantiate(config.reward_generator,num_agent=N)

            if config.reward_generator['type']=='random':
                if "uniform" in list(config.reward_generator['mean_prob_constraint']):
                    if "increasing" in list(config.reward_generator['mean_prob_constraint']):
                        if "concave" in list(config.reward_generator['mean_prob_constraint']):

                            g = np.random.dirichlet([1]*(N+1))[:-1]
                            slopes = np.sort(g)[::-1]
                            sampled_reward = np.cumsum(slopes)
                        else:
                            sampled_reward=np.random.rand(N,)
                            sampled_reward=np.sort(sampled_reward)
                    else:
                        sampled_reward=np.random.rand(N,)
                elif "dirichlet-gap" in list(config.reward_generator['mean_prob_constraint']):
                    g = np.random.dirichlet([config.reward_generator.alpha]*(N+1))     # gaps sum to 1
                    b = config.reward_generator.get('endpoint_bound',1)        
                    sampled_reward=np.cumsum(g[:-1]*b)  # f in [0,1], monotone
                elif "dirichlet-gap-endpoint" in list(config.reward_generator['mean_prob_constraint']):
                    g = np.random.dirichlet([config.reward_generator.alpha]*(N))     # gaps sum to 1       
                    if config.reward_generator['endpoint_dist'][0]=='Uniform':
                        E=np.random.uniform(low=config.reward_generator['endpoint_dist'][1],high=config.reward_generator['endpoint_dist'][2])
                    if config.reward_generator['endpoint_dist'][0]=='fixed':
                        E=config.reward_generator['endpoint_dist'][1]
                    sampled_reward=np.cumsum(g*E)  # f in [0,1], monotone

                sampled_reward=np.clip(sampled_reward,0,1)
                Reward_generator.set_mean(mean=list(sampled_reward))

            Agent=hydra.utils.instantiate(config.agent)

            if type(config.agent['para_loc'])==str:
                if config.agent['para_loc']=='random':
                    if config.agent['para_loc_dist'][0]=='Uniform':
                        low=np.ones(N,)*config.agent['para_loc_dist'][1]
                        high=np.ones(N,)*config.agent['para_loc_dist'][2]
                        para_loc=np.random.uniform(low=low,high=high)
                    elif config.agent['para_loc_dist'][0]=='Dirichlet':
                        alpha=[config.agent['para_loc_dist'][1]]*N
                        H = np.random.rand()
                        para_loc = H*np.random.dirichlet(alpha)
                    elif config.agent['para_loc_dist'][0]=='linear':
                        end_point= np.random.rand()*config.agent['para_loc_dist'][1]
                        para_loc = np.ones(N,)*end_point/N
                    elif config.agent['para_loc_dist'][0]=='Beta':
                        a=np.ones(N,)*config.agent['para_loc_dist'][1]
                        b=np.ones(N,)*config.agent['para_loc_dist'][2]
                        para_loc=np.random.beta(a,b)
                        print("para_loc=",para_loc)
                    
                    Agent.para_loc=para_loc
            
            if type(config.agent['para_shape'])==str:
                if config.agent['para_shape']=='random':
                    if config.agent['para_shape_dist'][0]=='Uniform':
                        low=np.ones(N,)*config.agent['para_shape_dist'][1]
                        high=np.ones(N,)*config.agent['para_shape_dist'][2]
                        para_shape=np.random.uniform(low=low,high=high)
                    elif config.agent['para_shape_dist'][0]=='Gamma':
                        shape=np.ones(N,)*config.agent['para_shape_dist'][1]
                        rate=np.ones(N,)*config.agent['para_shape_dist'][2]
                        para_shape=np.random.gamma(shape=shape,scale=1/rate)
                        print("para_shape=",para_shape)
                    elif config.agent['para_shape_dist'][0] == 'LogNormal':
                        median = config.agent['para_shape_dist'][1]
                        sigma = config.agent['para_shape_dist'][2]

                        mean = np.log(median)

                        para_shape = np.random.lognormal(
                            mean=mean,
                            sigma=sigma,
                            size=N)
                        print("para_shape=",para_shape)

                    Agent.para_shape=para_shape
            
            
            Agent_model=hydra.utils.instantiate(config.model,num_agent=N)
            Policy=hydra.utils.instantiate(config.policy,model=Agent_model)
            Setting=hydra.utils.instantiate(config.setting,
                                        principal_policy=Policy,
                                        agent_policy=Agent,
                                        Reward_generator=Reward_generator)
            
            

            #--------Store data for each run--------
            if config.reward_generator['type']=='random':
                print("Random Reward func=",sampled_reward)
                optimal_incentive, optimal_utility = Setting.optimal_solution()
            else:
                if sim==0:
                    optimal_incentive, optimal_utility = Setting.optimal_solution()
            wandb.log({"seed":seed,"optimal_utility": optimal_utility})
            

            if Agent_model.name=="bayes-logit":
                reward_array, _, incentive_array, EU_array, para_loc_array,para_shape_array,var_loc_array,var_shape_array = Setting.run_fixed_budget(max_round=T)
            else:
                reward_array, _, incentive_array, EU_array, para_loc_array,para_shape_array = Setting.run_fixed_budget(max_round=T)
            
            
            
            offered_incentive_array[sim,:,:]=incentive_array
            incentive_regret=optimal_incentive-incentive_array
            l1_dist_incentive_array[sim,:]=np.linalg.norm(incentive_regret,ord=1,axis=1)
            l2_dist_incentive_array[sim,:]=np.linalg.norm(incentive_regret,ord=2,axis=1)
            linf_dist_incentive_array[sim,:]=np.linalg.norm(incentive_regret,ord=np.inf,axis=1)

            total_incentive_array[sim,:]=np.sum(incentive_array,axis=1).reshape(-1,)
            l1_dist_total_incentive_array[sim,:]=np.abs(total_incentive_array[sim,:]-np.sum(optimal_incentive))

            regret_array[sim,:]=optimal_utility-reward_array
            EU_regret_array[sim,:]=optimal_utility-EU_array

            # simple_regret_array[sim,:]=optimal_utility-pred_best_reward_array
            para_loc_regret=Agent.para_loc-para_loc_array
            para_shape_regret=Agent.para_shape-para_shape_array
            l1_dist_para_loc_array[sim,:]=np.linalg.norm(para_loc_regret,ord=1,axis=1)
            l1_dist_para_shape_array[sim,:]=np.linalg.norm(para_shape_regret,ord=1,axis=1)
            l2_dist_para_loc_array[sim,:]=np.linalg.norm(para_loc_regret,ord=2,axis=1)
            l2_dist_para_shape_array[sim,:]=np.linalg.norm(para_shape_regret,ord=2,axis=1)
            linf_dist_para_loc_array[sim,:]=np.linalg.norm(para_loc_regret,ord=np.inf,axis=1)
            linf_dist_para_shape_array[sim,:]=np.linalg.norm(para_shape_regret,ord=np.inf,axis=1)

            if Agent_model.name=="bayes-logit":
                min_var_loc_array[sim,:]=np.min(var_loc_array,axis=1)
                max_var_loc_array[sim,:]=np.max(var_loc_array,axis=1)
                avg_var_loc_array[sim,:]=np.mean(var_loc_array,axis=1)
                min_var_shape_array[sim,:]=np.min(var_shape_array,axis=1)
                max_var_shape_array[sim,:]=np.max(var_shape_array,axis=1)
                avg_var_shape_array[sim,:]=np.mean(var_shape_array,axis=1)

        # #-------log mean and variance of data-------
        mean_offered_incentive_array=np.mean(offered_incentive_array,axis=0)
        sum_offered_incentive_array=np.sum(offered_incentive_array,axis=0)
        sqsum_offered_incentive_array=np.sum(offered_incentive_array**2,axis=0)
        mean_l1_dist_incentive_array=np.mean(l1_dist_incentive_array,axis=0)
        sum_l1_dist_incentive_array=np.sum(l1_dist_incentive_array,axis=0)
        sqsum_l1_dist_incentive_array=np.sum(l1_dist_incentive_array**2,axis=0)
        mean_l2_dist_incentive_array=np.mean(l2_dist_incentive_array,axis=0)
        sum_l2_dist_incentive_array=np.sum(l2_dist_incentive_array,axis=0)
        sqsum_l2_dist_incentive_array=np.sum(l2_dist_incentive_array**2,axis=0)
        mean_linf_dist_incentive_array=np.mean(linf_dist_incentive_array,axis=0)
        sum_linf_dist_incentive_array=np.sum(linf_dist_incentive_array,axis=0)
        sqsum_linf_dist_incentive_array=np.sum(linf_dist_incentive_array**2,axis=0)

        mean_total_incentive_array=np.mean(total_incentive_array,axis=0)
        sum_total_total_incentive_array=np.sum(total_incentive_array,axis=0)
        sqsum_total_incentive_array=np.sum(total_incentive_array**2,axis=0)
        mean_l1_dist_total_incentive_array=np.mean(l1_dist_total_incentive_array,axis=0)
        sum_l1_dist_total_incentive_array=np.sum(l1_dist_total_incentive_array,axis=0)
        sqsum_l1_dist_total_incentive_array=np.sum(l1_dist_total_incentive_array**2,axis=0)

        
        mean_regret_array=np.mean(regret_array,axis=0)
        sum_regret_array=np.sum(regret_array,axis=0)
        sqsum_regret_array=np.sum(regret_array**2,axis=0)
        cum_regret_array=np.cumsum(regret_array,axis=1)
        mean_cum_regret_array=np.mean(cum_regret_array,axis=0)
        sum_cum_regret_array=np.sum(cum_regret_array,axis=0)
        sqsum_cum_regret_array=np.sum(cum_regret_array**2,axis=0)

        mean_EU_regret_array=np.mean(EU_regret_array,axis=0)
        sum_EU_regret_array=np.sum(EU_regret_array,axis=0)
        sqsum_EU_regret_array=np.sum(EU_regret_array**2,axis=0)
        cum_EU_regret_array=np.cumsum(EU_regret_array,axis=1)
        mean_cum_EU_regret_array=np.mean(cum_EU_regret_array,axis=0)
        sum_cum_EU_regret_array=np.sum(cum_EU_regret_array,axis=0)
        sqsum_cum_EU_regret_array=np.sum(cum_EU_regret_array**2,axis=0)

        # mean_simple_regret_array=np.mean(simple_regret_array,axis=0)
        # sum_simple_regret_array=np.sum(simple_regret_array,axis=0)
        # sqsum_simple_regret_array=np.sum(simple_regret_array**2,axis=0)

        mean_l1_dist_para_loc_array=np.mean(l1_dist_para_loc_array,axis=0)
        sum_l1_dist_para_loc_array=np.sum(l1_dist_para_loc_array,axis=0)
        sqsum_l1_dist_para_loc_array=np.sum(l1_dist_para_loc_array**2,axis=0)
        mean_l2_dist_para_loc_array=np.mean(l2_dist_para_loc_array,axis=0)
        sum_l2_dist_para_loc_array=np.sum(l2_dist_para_loc_array,axis=0)
        sqsum_l2_dist_para_loc_array=np.sum(l2_dist_para_loc_array**2,axis=0)
        mean_linf_dist_para_loc_array=np.mean(linf_dist_para_loc_array,axis=0)
        sum_linf_dist_para_loc_array=np.sum(linf_dist_para_loc_array,axis=0)
        sqsum_linf_dist_para_loc_array=np.sum(linf_dist_para_loc_array**2,axis=0)

        mean_l1_dist_para_shape_array=np.mean(l1_dist_para_shape_array,axis=0)
        sum_l1_dist_para_shape_array=np.sum(l1_dist_para_shape_array,axis=0)
        sqsum_l1_dist_para_shape_array=np.sum(l1_dist_para_shape_array**2,axis=0)
        mean_l2_dist_para_shape_array=np.mean(l2_dist_para_shape_array,axis=0)
        sum_l2_dist_para_shape_array=np.sum(l2_dist_para_shape_array,axis=0)
        sqsum_l2_dist_para_shape_array=np.sum(l2_dist_para_shape_array**2,axis=0)
        mean_linf_dist_para_shape_array=np.mean(linf_dist_para_shape_array,axis=0)
        sum_linf_dist_para_shape_array=np.sum(linf_dist_para_shape_array,axis=0)
        sqsum_linf_dist_para_shape_array=np.sum(linf_dist_para_shape_array**2,axis=0)


        if Agent_model.name=="bayes-logit":
            mean_min_var_loc_array=np.mean(min_var_loc_array,axis=0)
            sum_min_var_loc_array=np.sum(min_var_loc_array,axis=0)
            mean_max_var_loc_array=np.mean(max_var_loc_array,axis=0)
            sum_max_var_loc_array=np.sum(max_var_loc_array,axis=0)
            mean_avg_var_loc_array=np.mean(avg_var_loc_array,axis=0)
            sum_avg_var_loc_array=np.sum(avg_var_loc_array,axis=0)

            mean_min_var_shape_array=np.mean(min_var_shape_array,axis=0)
            sum_min_var_shape_array=np.sum(min_var_shape_array,axis=0)
            mean_max_var_shape_array=np.mean(max_var_shape_array,axis=0)
            sum_max_var_shape_array=np.sum(max_var_shape_array,axis=0)
            mean_avg_var_shape_array=np.mean(avg_var_shape_array,axis=0)
            sum_avg_var_shape_array=np.sum(avg_var_shape_array,axis=0)

        if T>1000:
            log=10
        else:
            log=1
        for t in range(T):
            if t % log != 0 and t<T-1:
                continue

            if Agent_model.name=="bayes-logit":
                wandb.log({ "round":t,
                        "mean_offered_incentive":mean_offered_incentive_array[t],"sum_offered_incentive":sum_offered_incentive_array[t],"sqsum_offered_incentive":sqsum_offered_incentive_array[t],
                        "mean_l1_dist_incentive":mean_l1_dist_incentive_array[t],"sum_l1_dist_incentive":sum_l1_dist_incentive_array[t],"sqsum_l1_dist_incentive":sqsum_l1_dist_incentive_array[t],
                        "mean_l2_dist_incentive":mean_l2_dist_incentive_array[t],"sum_l2_dist_incentive":sum_l2_dist_incentive_array[t],"sqsum_l2_dist_incentive":sqsum_l2_dist_incentive_array[t],
                        "mean_linf_dist_incentive":mean_linf_dist_incentive_array[t],"sum_linf_dist_incentive":sum_linf_dist_incentive_array[t],"sqsum_linf_dist_incentive":sqsum_linf_dist_incentive_array[t],
                        "mean_total_incentive":mean_total_incentive_array[t],"sum_total_incentive":sum_total_total_incentive_array[t],"sqsum_total_incentive":sqsum_total_incentive_array[t],
                        "mean_l1_dist_total_incentive":mean_l1_dist_total_incentive_array[t],"sum_l1_dist_total_incentive":sum_l1_dist_total_incentive_array[t],"sqsum_l1_dist_total_incentive":sqsum_l1_dist_total_incentive_array[t],
                        "mean_regret":mean_regret_array[t],"sum_regret":sum_regret_array[t],"sqsum_regret":sqsum_regret_array[t],
                        "mean_cum_regret":mean_cum_regret_array[t],"sum_cum_regret":sum_cum_regret_array[t],"sqsum_cum_regret":sqsum_cum_regret_array[t],
                        "mean_EU_regret":mean_EU_regret_array[t],"sum_EU_regret":sum_EU_regret_array[t],"sqsum_EU_regret":sqsum_EU_regret_array[t],
                        "mean_cum_EU_regret":mean_cum_EU_regret_array[t],"sum_cum_EU_regret":sum_cum_EU_regret_array[t],"sqsum_cum_EU_regret":sqsum_cum_EU_regret_array[t],
                        # "mean_simple_regret":mean_simple_regret_array[t],"sum_simple_regret":sum_simple_regret_array[t],"sqsum_simple_regret":sqsum_simple_regret_array[t],
                        "mean_l1_dist_para_loc_array":mean_l1_dist_para_loc_array[t],"sum_l1_dist_para_loc_array":sum_l1_dist_para_loc_array[t],"sqsum_l1_dist_para_loc_array":sqsum_l1_dist_para_loc_array[t],
                        "mean_l2_dist_para_loc_array":mean_l2_dist_para_loc_array[t],"sum_l2_dist_para_loc_array":sum_l2_dist_para_loc_array[t],"sqsum_l2_dist_para_loc_array":sqsum_l2_dist_para_loc_array[t],
                        "mean_linf_dist_para_loc_array":mean_linf_dist_para_loc_array[t],"sum_linf_dist_para_loc_array":sum_linf_dist_para_loc_array[t],"sqsum_linf_dist_para_loc_array":sqsum_linf_dist_para_loc_array[t],
                        "mean_l1_dist_para_shape_array":mean_l1_dist_para_shape_array[t],"sum_l1_dist_para_shape_array":sum_l1_dist_para_shape_array[t],"sqsum_l1_dist_para_shape_array":sqsum_l1_dist_para_shape_array[t],
                        "mean_l2_dist_para_shape_array":mean_l2_dist_para_shape_array[t],"sum_l2_dist_para_shape_array":sum_l2_dist_para_shape_array[t],"sqsum_l2_dist_para_shape_array":sqsum_l2_dist_para_shape_array[t],
                        "mean_linf_dist_para_shape_array":mean_linf_dist_para_shape_array[t],"sum_linf_dist_para_shape_array":sum_linf_dist_para_shape_array[t],"sqsum_linf_dist_para_shape_array":sqsum_linf_dist_para_shape_array[t],
                        "mean_min_var_loc":mean_min_var_loc_array[t],"mean_max_var_loc":mean_max_var_loc_array[t],"mean_avg_var_loc":mean_avg_var_loc_array[t],
                        "mean_min_var_shape":mean_min_var_shape_array[t],"mean_max_var_shape":mean_max_var_shape_array[t],"mean_avg_var_shape":mean_avg_var_shape_array[t],
                        "sum_min_var_loc":sum_min_var_loc_array[t],"sum_max_var_loc":sum_max_var_loc_array[t],"sum_avg_var_loc":sum_avg_var_loc_array[t],
                        "sum_min_var_shape":sum_min_var_shape_array[t],"sum_max_var_shape":sum_max_var_shape_array[t],"sum_avg_var_shape":sum_avg_var_shape_array[t],
                        })
            else:
                wandb.log({"round":t,
                        "mean_offered_incentive":mean_offered_incentive_array[t],"sum_offered_incentive":sum_offered_incentive_array[t],"sqsum_offered_incentive":sqsum_offered_incentive_array[t],
                        "mean_l1_dist_incentive":mean_l1_dist_incentive_array[t],"sum_l1_dist_incentive":sum_l1_dist_incentive_array[t],"sqsum_l1_dist_incentive":sqsum_l1_dist_incentive_array[t],
                        "mean_l2_dist_incentive":mean_l2_dist_incentive_array[t],"sum_l2_dist_incentive":sum_l2_dist_incentive_array[t],"sqsum_l2_dist_incentive":sqsum_l2_dist_incentive_array[t],
                        "mean_linf_dist_incentive":mean_linf_dist_incentive_array[t],"sum_linf_dist_incentive":sum_linf_dist_incentive_array[t],"sqsum_linf_dist_incentive":sqsum_linf_dist_incentive_array[t],
                        "mean_total_incentive":mean_total_incentive_array[t],"sum_total_incentive":sum_total_total_incentive_array[t],"sqsum_total_incentive":sqsum_total_incentive_array[t],
                        "mean_l1_dist_total_incentive":mean_l1_dist_total_incentive_array[t],"sum_l1_dist_total_incentive":sum_l1_dist_total_incentive_array[t],"sqsum_l1_dist_total_incentive":sqsum_l1_dist_total_incentive_array[t],
                        "mean_regret":mean_regret_array[t],"sum_regret":sum_regret_array[t],"sqsum_regret":sqsum_regret_array[t],
                        "mean_cum_regret":mean_cum_regret_array[t],"sum_cum_regret":sum_cum_regret_array[t],"sqsum_cum_regret":sqsum_cum_regret_array[t],
                        "mean_EU_regret":mean_EU_regret_array[t],"sum_EU_regret":sum_EU_regret_array[t],"sqsum_EU_regret":sqsum_EU_regret_array[t],
                        "mean_cum_EU_regret":mean_cum_EU_regret_array[t],"sum_cum_EU_regret":sum_cum_EU_regret_array[t],"sqsum_cum_EU_regret":sqsum_cum_EU_regret_array[t],
                        # "mean_simple_regret":mean_simple_regret_array[t],"sum_simple_regret":sum_simple_regret_array[t],"sqsum_simple_regret":sqsum_simple_regret_array[t],
                        "mean_l1_dist_para_loc_array":mean_l1_dist_para_loc_array[t],"sum_l1_dist_para_loc_array":sum_l1_dist_para_loc_array[t],"sqsum_l1_dist_para_loc_array":sqsum_l1_dist_para_loc_array[t],
                        "mean_l2_dist_para_loc_array":mean_l2_dist_para_loc_array[t],"sum_l2_dist_para_loc_array":sum_l2_dist_para_loc_array[t],"sqsum_l2_dist_para_loc_array":sqsum_l2_dist_para_loc_array[t],
                        "mean_linf_dist_para_loc_array":mean_linf_dist_para_loc_array[t],"sum_linf_dist_para_loc_array":sum_linf_dist_para_loc_array[t],"sqsum_linf_dist_para_loc_array":sqsum_linf_dist_para_loc_array[t],
                        "mean_l1_dist_para_shape_array":mean_l1_dist_para_shape_array[t],"sum_l1_dist_para_shape_array":sum_l1_dist_para_shape_array[t],"sqsum_l1_dist_para_shape_array":sqsum_l1_dist_para_shape_array[t],
                        "mean_l2_dist_para_shape_array":mean_l2_dist_para_shape_array[t],"sum_l2_dist_para_shape_array":sum_l2_dist_para_shape_array[t],"sqsum_l2_dist_para_shape_array":sqsum_l2_dist_para_shape_array[t],
                        "mean_linf_dist_para_shape_array":mean_linf_dist_para_shape_array[t],"sum_linf_dist_para_shape_array":sum_linf_dist_para_shape_array[t],"sqsum_linf_dist_para_shape_array":sqsum_linf_dist_para_shape_array[t],
                    })
    finally:
        wandb.finish()

if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f" in {toc - tic:0.4f} seconds")
    sys.exit()

