import subprocess
import numpy as np

num_sim=100
max_round=10000
for sim_id in range(1,5):
    init_game_seed_id=sim_id*num_sim
    game_seed="setting.init_seed_id="+str(init_game_seed_id)

    # process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
    #                         ,"setting=MABA_10IncUni","reward_generator=bernoulli_inc_rand","wandb_mode=offline","policy=TS_aDir_inc"])

    process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
                            ,"setting=MABA_10IncUni","reward_generator=bernoulli_inc_rand","wandb_mode=online","policy=TS_mGibb"])