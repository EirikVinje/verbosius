from gc import collect
import argparse
import shutil
import pickle
import os

from torch.cuda import empty_cache
import green_tsetlin as gt
import numpy as np
import optuna

from verbosius.trainingdata import Trainingdata
from verbosius.performance import ModelMetrics
from verbosius.transformer import Transformer
from verbosius.preprocess import Preprocess
from verbosius.weighter import Weighter
from verbosius.chunker import Chunker
import utils.config as config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def objective(trial):

    config.learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-2, log=True)
    config.neutral_weight = trial.suggest_float("neutral_weight", 0, 0.05, step=0.0001)
    config.loss_weight = trial.suggest_float("loss_weight", 1, 10, step=0.1)
    
    transformer = Transformer(part_n=part_n, 
                              model_name="model_1", 
                              force_write=True)
    transformer.run()
    del transformer

    model_metrics = ModelMetrics("model_1", "/home/bigtech/data/verbosius/amazon/nc_2")
    model_metrics.get_metrics()
    model_metrics.save_metrics()
    acc = model_metrics.metrics["seq_acc"]
    del model_metrics
    
    collect()
    empty_cache()

    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="transformer hpsearch")

    parser.add_argument("--n_chunks", type=int, help="Number of chunks")
    parser.add_argument("--part_n", type=int, help="partition id")
    parser.add_argument("--n_trials", type=int, help="Number of trials")
    parser.add_argument("--study_name", type=str, help="Study name")
    args = parser.parse_args()

    study_name = args.study_name
    n_chunks = args.n_chunks
    part_n = args.part_n
    n_trials = args.n_trials

    config.root = f"/home/bigtech/data/verbosius/hpsearch/"
    config.TM_EPOCHS = 50
    config.num_train_epochs = 10
    
    ds_path = f"/home/bigtech/data/verbosius/amazon/nc_{n_chunks}"

    
    preprocess = Preprocess(ds_path=ds_path, 
                            part_n=part_n, 
                            progress_bar=True, 
                            force_write=True)
    preprocess.run()
    del preprocess

    weighter = Weighter(part_n=part_n, 
                        progress_bar=True, 
                        force_write=True)
    weighter.run()
    del weighter

    trainingdata = Trainingdata(part_n=part_n, 
                                progress_bar=True, 
                                force_write=True)
    trainingdata.run()
    del trainingdata

    collect()    

    study = optuna.create_study(study_name=study_name, 
                                direction="maximize", 
                                storage=f"sqlite:////home/bigtech/projects/verbosius/sqlite3.db", 
                                load_if_exists=True)
    
    study.set_user_attr("part_n", part_n)
    study.set_user_attr("n_chunks", n_chunks)
    study.set_user_attr("TM_EPOCHS", config.TM_EPOCHS)
    study.set_user_attr("num_train_epochs", config.num_train_epochs)
    
    study.set_user_attr("lr_range", ["1e-8", "1e-2"])
    study.set_user_attr("loss_weight_range", [1, 10])
    study.set_user_attr("neutral_weight_range", [0, 0.05])

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(study.best_params)
    print(study.best_value)