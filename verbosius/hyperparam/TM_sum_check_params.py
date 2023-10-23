import pickle
import argparse
import os
import shutil

import optuna
import numpy as np
import green_tsetlin as gt
import config as config
import run_config as run_config

from chunking.stage_chunks import stage_chunks
from preprocessing.stage_preprocess import stage_preprocess
from trainingdata.stage_trainingdata import stage_trainingdata
from xai_transformer.stage_transformer import stage_transformer



os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_params_1(seed):

    print("Running params 1 with seed: ", seed)

    config.seed = seed
    # set parameters
    config.LITERAL_BUDGET = 6
    config.ERROR_LITERAL_BUDGET = 6

    # Hyperparameters to be optimized
    config.MAX_FEATURES = 1750
    config.MAX_DF = 0.7086319286046587
    config.MIN_DF = 22
    
    config.NUMBER_OF_CLAUSES = 4000
    config.S = 17.7
    config.T = 5000
    config.TM_EPOCHS = 15
    
    config.ERROR_MAX_FEATURES = 700
    config.ERROR_NUMBER_OF_CLAUSES = 3200
    config.ERROR_S = 25.61065
    config.ERROR_T = 1750
    config.ERROR_MAX_DF = 0.437663961421369
    config.ERROR_MIN_DF = 30

    config.SKB_score_func = "mutual_info_classif"


    total_count = stage_trainingdata(dataset = run_config.dataset,
                    input = run_config.input_preproc,
                    output = run_config.output_traindata,
                    chunkdist_n= run_config.chunkdist_n)


    os.system("cd ~ && ./projects/verbosius/make_env_HP.sh")

    return total_count

def test_params_2(seed):

    print("Running params 2 with seed: ", seed)
    
    config.seed = seed
    # set parameters
    config.LITERAL_BUDGET = 6
    config.ERROR_LITERAL_BUDGET = 6

    # Hyperparameters to be optimized
    config.MAX_FEATURES = 2250
    config.MAX_DF = 0.70
    config.MIN_DF = 21
    
    config.NUMBER_OF_CLAUSES = 4000
    config.S = 18
    config.T = 5000
    config.TM_EPOCHS = 15
    
    config.ERROR_MAX_FEATURES = 800
    config.ERROR_NUMBER_OF_CLAUSES = 3500
    config.ERROR_S = 25
    config.ERROR_T = 1750
    config.ERROR_MAX_DF = 0.45
    config.ERROR_MIN_DF = 25

    config.SKB_score_func = "mutual_info_classif"


    total_count = stage_trainingdata(dataset = run_config.dataset,
                    input = run_config.input_preproc,
                    output = run_config.output_traindata,
                    chunkdist_n= run_config.chunkdist_n)


    os.system("cd ~ && ./projects/verbosius/make_env_HP.sh")

    return total_count

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for IMDB dataset on complete pipeline")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials to run")
    
    run_config.chunkdist_n = 5555
    config.seed = 42

    user = os.environ.get('USER')

    if not os.path.exists(f"/home/{user}/data/verbosius/imdb/preprocess/imdb_chunkdist_5555/"):
        os.system("cd ~ && ./projects/verbosius/make_env.sh")
        stage_chunks(dataset = run_config.dataset,
                    chunk_size = 8000,
                    chunk_amount = 5,
                    input = run_config.input_raw,
                    output = run_config.output_chunk,
                    chunkdist_n = run_config.chunkdist_n)
        
        stage_preprocess(dataset = run_config.dataset,
                        input = run_config.input_chunk,
                        output = run_config.output_preproc,
                        chunkdist_n = run_config.chunkdist_n)
    else:
        os.system("cd ~ && ./projects/verbosius/make_env_HP.sh")
        print('yo')

    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    res_1 = []
    res_2 = []

    for seed in seeds:
        val1 = test_params_1(seed)
        print('val1: ', val1)
        val2 = test_params_2(seed)
        print('val2: ', val2)
        res_1.append(val1)
        res_2.append(val2)

    # write results to file
    with open(f"/home/{user}/projects/verbosius/hyperparam/params_1.pickle", "wb") as fp:
        pickle.dump(res_1, fp)

    with open(f"/home/{user}/projects/verbosius/hyperparam/params_2.pickle", "wb") as fp:
        pickle.dump(res_2, fp)
