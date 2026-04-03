import subprocess
import numpy as np

num_sim=10
max_round=10000
for sim_id in range(0,1):
    init_game_seed_id=sim_id*num_sim
    game_seed="setting.init_seed_id="+str(init_game_seed_id)
    # process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
    #                         ,"policy=TS","reward_generator=bernoulli_cave_rand","setting=MABA_10CaveUni",])
    # process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
    #                         ,"policy=UTS","reward_generator=bernoulli_cave_rand","setting=MABA_10CaveUni",])
    # process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
    #                         ,"policy=BayesUCB","reward_generator=bernoulli_cave_rand","setting=MABA_10CaveUni",])

    # process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
    #                         ,"policy=TS","reward_generator=bernoulli_cdire","setting=MABA_10CaveUni2","reward_generator.alpha=0.1",])
    # process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
    #                         ,"policy=UTS","reward_generator=bernoulli_cdire","setting=MABA_10CaveUni2","reward_generator.alpha=0.1"])
    # process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
    #                         ,"policy=BayesUCB","reward_generator=bernoulli_cdire","setting=MABA_10CaveUni2","reward_generator.alpha=0.1"])

    
    # process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
    #                         ,"policy=TS","reward_generator=bernoulli_cdire","setting=MABA_10CaveUni2","reward_generator.alpha=10",])
    # process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
    #                         ,"policy=UTS","reward_generator=bernoulli_cdire","setting=MABA_10CaveUni2","reward_generator.alpha=10"])
    # process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
    #                         ,"policy=BayesUCB","reward_generator=bernoulli_cdire","setting=MABA_10CaveUni2","reward_generator.alpha=10"])
    
    process=subprocess.run(["python","run_linear_game.py",game_seed,"num_sim="+str(num_sim),"max_round="+str(max_round)
                            ,"policy=TS_aDir_cave2","reward_generator=bernoulli_cdire","setting=MABA_10CaveUni2","reward_generator.alpha=10"])