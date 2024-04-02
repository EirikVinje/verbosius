import pickle
import argparse
import os
import shutil
import argparse

import optuna
import numpy as np
import green_tsetlin as gt
import config as config

from chunker import Chunker
from preprocess.preprocess import Preprocess
from weighter import Weighter
from train_eval_tokenize.trainingdata import Trainingdata

from xai_transformer.stage_transformer import stage_transformer
from xai_validation.stage_validation import stage_validation
from performance import model_accuracy


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def objective(trial):

    chunkdist_n = 6165
    dataset = "amazon"

    config.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    config.neutral_weight = trial.suggest_float("neutral_weight", 0, 0.05, step=0.0001)
    config.loss_weight = trial.suggest_float("loss_weight", 1, 10, step=0.1)
    
    stage_transformer(dataset, chunkdist_n)

    met = model_accuracy(dataset, chunkdist_n, "model_t", "big", False)

    model_path = os.path.join(config.root, 'models', f"{dataset}_chunkdist_{chunkdist_n}")
    os.system(f"rm -rf {model_path}")

    return met[0]


if __name__ == "__main__":
    
    study = optuna.create_study(study_name="new_transformer_search", direction="maximize", storage=f"sqlite:////home/{config.user}/projects/verbosius/sqlite3.db", load_if_exists=True)

    study.optimize(objective, n_trials=20, show_progress_bar=True)
    
    print(study.best_params)
    print(study.best_value)