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




@hydra.main(version_base=None,config_path="config", config_name="main2")
def main(config):

    #Instantitate wandb and log info of configurations 
    log_config=pd.json_normalize(OmegaConf.to_object(config),sep='/')
    log_config=log_config.to_dict(orient='records')[0]
    wandb.init(project ="PA_bandits",config=log_config, mode=config.wandb_mode, entity="s_phurinut",settings=wandb.Settings(start_method="thread"))
    
    M=config.num_sim
    T=config.max_round
    N=config.agent.num_agent

    regret_array=np.zeros((M,T))
    # simple_regret_array=np.zeros((M,T))

    offered_incentive_array=np.zeros((M,T,N))
    l1_dist_incentive_array=np.zeros((M,T))
    l2_dist_incentive_array=np.zeros((M,T))
    linf_dist_incentive_array=np.zeros((M,T))

    total_incentive_array=np.zeros((M,T))
    l1_dist_total_incentive_array=np.zeros((M,T))

    try:
        #Set environment of the simulations from YAML files
        if config.setting['type']=='pre-defined':
            for sim in range(0,M):
                seed=sim+config.setting['init_seed_id']
                np.random.seed(seed=seed)
                print("sim=",sim,' game_seed=',seed)

                Reward_generator=hydra.utils.instantiate(config.reward_generator,num_agent=N)
                Agent=hydra.utils.instantiate(config.agent)
                Agent_model=hydra.utils.instantiate(config.model,num_agent=N)
                Policy=hydra.utils.instantiate(config.policy,model=Agent_model)
                Setting=hydra.utils.instantiate(config.setting,
                                            principal_policy=Policy,
                                            agent_policy=Agent,
                                            Reward_generator=Reward_generator)
                reward_array, agent_response_array, incentive_array = Setting.run_fixed_budget(max_round=T)
                optimal_incentive, optimal_utility = Setting.optimal_solution()

                #--------Store data for each run--------
                wandb.log({"seed":seed,"optimal_utility": optimal_utility})
                
                offered_incentive_array[sim,:,:]=incentive_array
                incentive_regret=optimal_incentive-incentive_array
                l1_dist_incentive_array[sim,:]=np.linalg.norm(incentive_regret,ord=1,axis=1)
                l2_dist_incentive_array[sim,:]=np.linalg.norm(incentive_regret,ord=2,axis=1)
                linf_dist_incentive_array[sim,:]=np.linalg.norm(incentive_regret,ord=np.inf,axis=1)

                total_incentive_array[sim,:]=np.sum(incentive_array,axis=1).reshape(-1,)
                l1_dist_total_incentive_array[sim,:]=np.abs(total_incentive_array[sim,:]-np.sum(optimal_incentive))

                regret_array[sim,:]=optimal_utility-reward_array
                # simple_regret_array[sim,:]=optimal_utility-pred_best_reward_array

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

        # mean_simple_regret_array=np.mean(simple_regret_array,axis=0)
        # sum_simple_regret_array=np.sum(simple_regret_array,axis=0)
        # sqsum_simple_regret_array=np.sum(simple_regret_array**2,axis=0)


        if T>1000:
            log=10
        else:
            log=1
        for t in range(T):
            if t % log != 0 and t<T-1:
                continue

            wandb.log({"round":t,
                        "mean_offered_incentive":mean_offered_incentive_array[t],"sum_offered_incentive":sum_offered_incentive_array[t],"sqsum_offered_incentive":sqsum_offered_incentive_array[t],
                        "mean_l1_dist_incentive":mean_l1_dist_incentive_array[t],"sum_l1_dist_incentive":sum_l1_dist_incentive_array[t],"sqsum_l1_dist_incentive":sqsum_l1_dist_incentive_array[t],
                        "mean_l2_dist_incentive":mean_l2_dist_incentive_array[t],"sum_l2_dist_incentive":sum_l2_dist_incentive_array[t],"sqsum_l2_dist_incentive":sqsum_l2_dist_incentive_array[t],
                        "mean_linf_dist_incentive":mean_linf_dist_incentive_array[t],"sum_linf_dist_incentive":sum_linf_dist_incentive_array[t],"sqsum_linf_dist_incentive":sqsum_linf_dist_incentive_array[t],
                        "mean_total_incentive":mean_total_incentive_array[t],"sum_total_incentive":sum_total_total_incentive_array[t],"sqsum_total_incentive":sqsum_total_incentive_array[t],
                        "mean_l1_dist_total_incentive":mean_l1_dist_total_incentive_array[t],"sum_l1_dist_total_incentive":sum_l1_dist_total_incentive_array[t],"sqsum_l1_dist_total_incentive":sqsum_l1_dist_total_incentive_array[t],
                        "mean_regret":mean_regret_array[t],"sum_regret":sum_regret_array[t],"sqsum_regret":sqsum_regret_array[t],
                        "mean_cum_regret":mean_cum_regret_array[t],"sum_cum_regret":sum_cum_regret_array[t],"sqsum_cum_regret":sqsum_cum_regret_array[t],
                        # "mean_simple_regret":mean_simple_regret_array[t],"sum_simple_regret":sum_simple_regret_array[t],"sqsum_simple_regret":sqsum_simple_regret_array[t],
            })
    finally:
        wandb.finish()

if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f" in {toc - tic:0.4f} seconds")
    sys.exit()

