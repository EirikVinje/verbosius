import pickle
import argparse
import os
import shutil
import argparse

import optuna
import numpy as np
import green_tsetlin as gt
import utils.config as config

from verbosius.chunker import Chunker
from verbosius.preprocess import Preprocess
from verbosius.weighter import Weighter
from verbosius.trainingdata import Trainingdata
from verbosius.transformer import Transformer
from verbosius.performance import ModelMetrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def objective(trial):


    config.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    config.neutral_weight = trial.suggest_float("neutral_weight", 0, 0.05, step=0.0001)
    config.loss_weight = trial.suggest_float("loss_weight", 1, 10, step=0.1)
    
    Transformer(part_n, "model_1", force_write=True).run()

    model_metrics = ModelMetrics("model_1", "big")
    model_metrics.load_test()
    model_metrics.set_model()
    model_metrics.get_metrics()

    # shutil.rmtree(os.path.join(config.root, 'models', "model_1"))
    
    return model_metrics.metrics["seq_acc"]


if __name__ == "__main__":
    
    config.root = f"/home/{config.user}/data/verbosius/hpsearch/"
    config.TM_EPOCHS = 50
    config.num_train_epochs = 5

    part_n = 202
    n_chunks = 25

    Chunker("big", part_n, n_chunks, progress_bar=True, force_write=True).run()
    Preprocess(part_n, progress_bar=True, force_write=True).run()
    Weighter(part_n, progress_bar=True, force_write=True).run()
    Trainingdata(part_n, progress_bar=True, force_write=True).run()

    study = optuna.create_study(study_name="transformersearch_02_04", direction="maximize", storage=f"sqlite:////home/{config.user}/projects/verbosius/sqlite3.db", load_if_exists=True)

    study.optimize(objective, n_trials=20, show_progress_bar=False)
    
    print(study.best_params)
    print(study.best_value)