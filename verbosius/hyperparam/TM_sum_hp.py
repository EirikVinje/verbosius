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

def objective(trial):

    print("Trial: ", trial.number)

    # set parameters
    config.LITERAL_BUDGET = 6
    config.ERROR_LITERAL_BUDGET = 6

    # Hyperparameters to be optimized
    config.MAX_FEATURES = trial.suggest_int("MAX_FEATURES", 1000, 10000, step=250)
    config.MAX_DF = trial.suggest_float("MAX_DF", 0.4, 0.9)
    config.MIN_DF = trial.suggest_int("MIN_DF", 1, 30) #increase from 20 to 30
    
    config.NUMBER_OF_CLAUSES = trial.suggest_int("NUMBER_OF_CLAUSES", 4000, 10000, step=250) # lower from 5k to 4k, step to 250
    config.S = trial.suggest_float("S", 2.0, 20.0, step=0.1)
    config.T = trial.suggest_int("T", 1000, 11000, step=500)
    config.TM_EPOCHS = trial.suggest_int("TM_EPOCHS", 4, 15) #increased from 12 to 15
    
    config.ERROR_MAX_FEATURES=trial.suggest_int("ERROR_MAX_FEATURES", 250, 3000, step=50)
    config.ERROR_NUMBER_OF_CLAUSES=trial.suggest_int("ERROR_NUMBER_OF_CLAUSES", 250, 4000, step=50) # increased from 3000 to 4000
    config.ERROR_S = trial.suggest_float("ERROR_S", 2.0, 27.5) # increased from 20.0 to 25.0 to 27.5
    config.ERROR_T = trial.suggest_int("ERROR_T", 500, 2500, step=250) # increased to 2500 from 2000
    config.ERROR_MAX_DF = trial.suggest_float("ERROR_MAX_DF", 0.4, 0.95) #increased from 0.9 to 0.95
    config.ERROR_MIN_DF = trial.suggest_int("ERROR_MIN_DF", 5, 30) #increase from 20 to 30, lower bound from 1 to 5

    config.SKB_score_func = trial.suggest_categorical("SKB_score_func", ["chi2", "f_classif", "mutual_info_classif"])


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
    # hprun_tot_tm_text_cs8000_cn5
    study = optuna.create_study(study_name="hprun_tot_tm_text_cs8000_cn5", direction="maximize", storage=f"sqlite:////home/{user}/projects/verbosius/sqlite3.db", load_if_exists=True)

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


    study.optimize(objective, n_trials=parser.parse_args().n_trials, show_progress_bar=True)
    
    print(study.best_params)
    print(study.best_value)