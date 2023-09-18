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

    config.MAX_FEATURES = trial.suggest_categorical("MAX_FEATURES", [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    config.MAX_DF = trial.suggest_float("MAX_DF", 0.4, 0.9)
    config.MIN_DF = trial.suggest_int("MIN_DF", 1, 20)
    
    config.NUMBER_OF_CLAUSES = trial.suggest_categorical("NUMBER_OF_CLAUSES", [5000, 6000, 7000, 8000, 9000, 10000])
    config.LITERAL_BUDGET = trial.suggest_int("LITERAL_BUDGET", 8, 16)
    config.S = trial.suggest_float("S", 2.0, 20.0)
    config.T = trial.suggest_categorical("T", [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000])
    config.TM_EPOCHS = trial.suggest_int("TM_EPOCHS", 4, 8)
    
    config.learning_rate = 1.3910662710078138e-05
    config.per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64, 128])
    config.per_device_eval_batch_size = trial.suggest_categorical("per_device_eval_batch_size", [8, 16, 32, 64, 128])
    config.num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
    config.weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)
    config.warmup_steps = trial.suggest_int("warmup_steps", 0, 1000)
    config.eval_accumulation_steps = trial.suggest_categorical("eval_accumulation_steps", [1, 2, 4, 8, 16, 32])
    config.neutral_weight = trial.suggest_float("neutral_weight", 0.0, 0.5)
    config.loss_weight = trial.suggest_float("loss_weight", 0.0, 0.5)

    stage_trainingdata(dataset = run_config.dataset,
                    input = run_config.input_preproc,
                    output = run_config.output_traindata,
                    chunkdist_n= run_config.chunkdist_n)


    seq_acc = stage_transformer(dataset = run_config.dataset,
                    train_val_input = run_config.input_traindata,
                    test_input = run_config.input_testdata,
                    model_output = run_config.model_output,
                    chunkdist_n = run_config.chunkdist_n)


    os.system("cd ~ && ./projects/verbosius/make_env.sh")

    return seq_acc

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for IMDB dataset on complete pipeline")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials to run")
    
    run_config.chunkdist_n = np.random.randint(0, 100000)
    config.seed = 42

    study = optuna.create_study(study_name="hprun_chunk25000_amount1_run2", direction="maximize", storage="sqlite:////home/bigtech/projects/verbosius/sqlite3.db", load_if_exists=True)

    stage_chunks(dataset = run_config.dataset,
                chunk_size = 25000,
                chunk_amount = 1,
                input = run_config.input_raw,
                output = run_config.output_chunk,
                chunkdist_n = run_config.chunkdist_n)
    
    stage_preprocess(dataset = run_config.dataset,
                    input = run_config.input_chunk,
                    output = run_config.output_preproc,
                    chunkdist_n = run_config.chunkdist_n)

    study.optimize(objective, n_trials=parser.parse_args().n_trials, show_progress_bar=True)
    
    print(study.best_params)
    print(study.best_value)