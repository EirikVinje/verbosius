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
from xai_validation.stage_validation import stage_validation



os.environ["TOKENIZERS_PARALLELISM"] = "false"

def objective(trial):

    print("Trial: ", trial.number)

    # set parameters
    

    config.learning_rate = 1e-5
    config.per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64, 128])
    config.per_device_eval_batch_size = trial.suggest_categorical("per_device_eval_batch_size", [8, 16, 32, 64, 128])
    config.num_train_epochs = 5
    # config.weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)
    # config.warmup_steps = trial.suggest_int("warmup_steps", 0, 1000)
    # config.eval_accumulation_steps = trial.suggest_categorical("eval_accumulation_steps", [1, 2, 4, 8, 16, 32])
    config.neutral_weight = trial.suggest_float("neutral_weight", 0, 0.05, step=0.0001)
    config.loss_weight = 5
    
    seq_acc = stage_transformer(dataset = run_config.dataset,
                    train_val_input = run_config.input_traindata,
                    test_input = run_config.input_testdata,
                    model_output = run_config.model_output,
                    chunkdist_n = run_config.chunkdist_n)
    
    log_res = stage_validation(model_path=run_config.input_xai_val_model,
                     model_name=run_config.model_name,
                     batch_size_pred=run_config.batch_size)

    os.system("cd ~ && ./projects/verbosius/make_env_HP.sh")

    return seq_acc, log_res

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for IMDB dataset on complete pipeline")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials to run")
    
    run_config.chunkdist_n = 5555
    config.seed = 42

    user = os.environ.get('USER')
    # hprun_tot_tm_text_cs8000_cn5
    study = optuna.create_study(study_name="new_params_2_hprun_fixed_tm_text_cs8000_cn5_test_transf", directions=["maximize", "maximize"], storage=f"sqlite:////home/{user}/projects/verbosius/sqlite3.db", load_if_exists=True)

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
        
    if not os.path.exists(f"/home/{user}/data/verbosius/imdb/trainingdata/imdb_chunkdist_5555/"):
        _ = stage_trainingdata(dataset = run_config.dataset,
                        input = run_config.input_preproc,
                        output = run_config.output_traindata,
                        chunkdist_n= run_config.chunkdist_n)
    else:
        os.system("cd ~ && ./projects/verbosius/make_env_HP.sh")
        print('yo')


    study.optimize(objective, n_trials=parser.parse_args().n_trials, show_progress_bar=True)
    
    print(study.best_params)
    print(study.best_value)