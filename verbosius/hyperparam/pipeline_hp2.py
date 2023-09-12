import optuna
import pickle
import argparse
import os
import shutil

import numpy as np
import green_tsetlin as gt
import config as config
import run_config as run_config

from chunking.stage_chunks import stage_chunks
from preprocessing.stage_preprocess import stage_preprocess
from trainingdata.stage_trainingdata import stage_trainingdata
from xai_transformer.stage_transformer import stage_transformer


def objective(trial):

    config.seed = trial.suggest_int("seed", 0, 100000)
    
    config.MAX_FEATURES = trial.suggest_categorical("MAX_FEATURES", [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    config.MAX_DF = trial.suggest_uniform("MAX_DF", 0.4, 0.9)
    config.MIN_DF = trial.suggest_int("MIN_DF", 1, 20)
    config.N_GRAM_RANGE = (1, 2)
    
    config.NUMBER_OF_CLAUSES = trial.suggest_categorical("NUMBER_OF_CLAUSES", [5000, 6000, 7000, 8000, 9000, 10000])
    config.LITERAL_BUDGET = trial.suggest_int("LITERAL_BUDGET", 5, 20)
    config.S = trial.suggest_uniform("S", 1, 20)
    config.T = trial.suggest_categorical("T", [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000])
    config.TM_EPOCHS = trial.suggest_int("TM_EPOCHS", 4, 8)
    
    config.learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    config.per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64, 128])
    config.per_device_eval_batch_size = trial.suggest_categorical("per_device_eval_batch_size", [8, 16, 32, 64, 128])
    config.num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
    config.weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    config.warmup_steps = trial.suggest_int("warmup_steps", 0, 1000)
    config.eval_accumulation_steps = trial.suggest_categorical("eval_accumulation_steps", [1, 2, 4, 8, 16, 32])
    config.neutral_weight = trial.suggest_uniform("neutral_weight", 0.0, 1.0)
    config.loss_weight = trial.suggest_uniform("loss_weight", 0.0, 1.0)

    #stage_chunks(dataset : str, chunk_size : int, chunk_amount : int, input : str, output : str, chunkdist_n : int)
    stage_chunks(dataset = run_config.dataset,
                chunk_size = 100,
                chunk_amount = 3,
                input = run_config.input_raw,
                output = run_config.output_chunk,
                chunkdist_n = run_config.chunkdist_n)
    
    # stage_preprocess(dataset:str, input:str, output:str, chunkdist_n : int)
    stage_preprocess(dataset = run_config.dataset,
                    input = run_config.input_chunk,
                    output = run_config.output_preproc,
                    chunkdist_n = run_config.chunkdist_n)
    
    # stage_traindata(dataset:str, input:str, output:str, chunkdist_n : int)
    stage_trainingdata(dataset = run_config.dataset,
                    input = run_config.input_preproc,
                    output = run_config.output_traindata,
                    chunkdist_n= run_config.chunkdist_n)

    # stage_transformer(dataset : str, train_val_input : str, test_input : str, model_output : str, save_model : str, chunkdist_n : int)
    stage_transformer(dataset = run_config.dataset,
                    train_val_input = run_config.input_traindata,
                    test_input = run_config.input_testdata,
                    model_output = run_config.model_output,
                    save_model = True,
                    chunkdist_n = run_config.chunkdist_n)

    os.system("cd ~ && ./projects/verbosius/make_env.sh")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for IMDB dataset on complete pipeline")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials to run")
    
    #shutil.rmtree(config.final_output_dir)
    #os.mkdir(config.final_output_dir)

    study = optuna.create_study(study_name="complete_pipeline_with_validation_actual", directions=["maximize", "maximize"], storage="sqlite:///imdb_tm_pipe.db", load_if_exists=True)
    study.optimize(objective, n_trials=parser.parse_args().n_trials, show_progress_bar=True)
    print(study.best_params)
    print(study.best_value)