import pickle
import argparse
import os
import shutil

import optuna
import numpy as np
import green_tsetlin as gt
import config as config

from chunking.stage_chunks import stage_chunks
from preprocessing.stage_preprocess import stage_preprocess
from trainingdata.stage_trainingdata import stage_trainingdata
from xai_transformer.stage_transformer import stage_transformer
from xai_validation.stage_validation import stage_validation



os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def objective(trial):

    print("Trial: ", trial.number)

    config.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    config.per_device_train_batch_size = 8
    config.per_device_eval_batch_size = 8
    config.num_train_epochs = 10
    config.neutral_weight = trial.suggest_float("neutral_weight", 0, 0.05, step=0.0001)
    config.loss_weight = trial.suggest_float("loss_weight", 1, 10, step=0.1)
    
    seq_acc = stage_transformer(dataset = config.dataset, chunkdist_n = config.chunkdist_n)

    os.system(f"rm -rf {config.root}/models/*")

    return seq_acc

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for IMDB dataset on complete pipeline")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials to run")
    
    user = config.user
    config.root = f"/home/{user}/data/verbosius/hpsearch_env/{config.dataset}"
    config.seed = 42

    config.chunkdist_n = 5555
    config.dataset = "amazon"
    config.chunk_size = 8000
    config.chunk_amount = 125
    
    study = optuna.create_study(study_name="transformer_params_hpsearch_cs8000_chunk_amount", direction="maximize", storage=f"sqlite:////home/{user}/projects/verbosius/sqlite3.db", load_if_exists=True)

    stage_chunks(dataset = config.dataset,
                 chunk_size = config.chunk_size,
                 chunk_amount = config.chunk_amount,
                 chunkdist_n = config.chunkdist_n)
    
    stage_preprocess(dataset = config.dataset,
                     chunkdist_n = config.chunkdist_n)
    
    
    stage_trainingdata(dataset = config.dataset,
                       chunkdist_n = config.chunkdist_n)


    study.optimize(objective, n_trials=parser.parse_args().n_trials, show_progress_bar=True)
    
    print(study.best_params)
    print(study.best_value)