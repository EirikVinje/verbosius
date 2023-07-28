import optuna
import pickle
import argparse
import os
import shutil

import numpy as np
import green_tsetlin as gt
import config as config

import preprocessing.stager as pr
import trainingdata.stage_trainingdata as td
import xai_transformer.stage_transformer as tf
import xai_validation.stage_validation as vf

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
from warnings import simplefilter

simplefilter(action='ignore', category=UserWarning)


def objective(trial):


    num_clauses = 10000  #trial.suggest_int("num_clauses", 7000, 11000)
    s = trial.suggest_float("s", 2.6, 6)
    n_gram_range = trial.suggest_categorical("n_gram_range", [(1,2), (1,3)])
    threshold = 9500 #trial.suggest_int("threshold", 9000, 12000)
    literal_budget = 8

    lr = trial.suggest_float("lr", 1e-5, 5e-5)
    neutral_w = trial.suggest_float("neutral_w", 0.1, 0.6)
    loss_w = trial.suggest_float("loss_w", 0.1, 0.6)

    batch_size_pred = trial.suggest_categorical("batch_size_pred", [1,2,4,8,16,32,64])
    
    config.NUMBER_OF_CLAUSES = num_clauses
    config.S = s
    config.THRESHOLD = threshold
    config.LITERAL_BUDGET = literal_budget
    config.learning_rate = lr
    config.neutral_weight = neutral_w
    config.loss_weight = loss_w
    config.N_GRAM_RANGE = n_gram_range
    config.batch_size_pred = batch_size_pred
    print(config.dataset)
    td.main(config.dataset, config.traindat_input_path, config.batchdist_n, config.traindat_output_path)
    seq_acc, tok_acc = tf.main(config.dataset, config.final_input_dir, config.final_output_dir, config.save_model, (0,1))
    #val_acc = vf.main(config.final_output_dir, "imdb_model_0", config.batch_size_pred)

    # os.system(f"rm -rf {config.final_output_dir}")
    # os.system(f"mkdir {config.final_output_dir}")
    shutil.rmtree(config.final_output_dir)
    os.mkdir(config.final_output_dir)

    return seq_acc, tok_acc

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for IMDB dataset on complete pipeline")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on, either 'cpu' or 'cuda'")
    
    config.device = int(parser.parse_args().device)

    config.N_JOBS = parser.parse_args().n_jobs
    config.batchdist_n = 0

    shutil.rmtree(config.final_output_dir)
    os.mkdir(config.final_output_dir)

    if len(os.listdir(config.preproc_input_path)) == 0:
        pr.main(dataset=config.dataset, batch_size=config.batch_size, batch_amount_per_mix=config.batch_amount, input=config.preproc_input_path, output=config.preproc_output_path, use_test_set=config.use_test_set, batch_size_test=config.batch_size_test, shuffle=config.shuffle, seed=config.seed, test=1, test_size=0.5, batch_amount_test=-1)
        

    study = optuna.create_study(study_name="pipline_no_validation", directions=["maximize", "maximize"], storage="sqlite:///SST5_tm_pipe.db", load_if_exists=True)
    study.optimize(objective, n_trials=parser.parse_args().n_trials, show_progress_bar=True)
    print(study.best_params)
    print(study.best_value)

    