from typing import Tuple
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
import utils.config as config


class HyperparamSearch:

    def __init__(self, 
                 n_chunks : int,
                 part_n : int,
                 n_trials : int,
                 study_name : str,
                 lr_range : Tuple[int, int] = (1e-8, 1e-2),
                 nw_range : Tuple[int, int] = (0, 0.05),
                 lw_range : Tuple[int, int] = (1, 10),
                 root : str = "/home/bigtech/data/verbosius/hpsearch/"):
        
        config.root = root
        config.TM_EPOCHS = 50
        config.num_train_epochs = 10
        
        self.n_trials = n_trials
        
        self.n_chunks = n_chunks
        self.part_n = part_n
        
        self.lr_range = lr_range
        self.nw_range = nw_range
        self.lw_range = lw_range

        self.study = optuna.create_study(study_name=study_name, 
                                         direction="maximize", 
                                         storage=f"sqlite:////home/bigtech/projects/verbosius/sqlite3.db", 
                                         load_if_exists=True)
        
        self.study.set_user_attr("part_n", self.part_n)
        self.study.set_user_attr("n_chunks", self.n_chunks)
        self.study.set_user_attr("TM_EPOCHS", config.TM_EPOCHS)
        self.study.set_user_attr("num_train_epochs", config.num_train_epochs)
        self.study.set_user_attr("lr_range", self.lr_range)
        self.study.set_user_attr("loss_weight_range", self.lw_range)
        self.study.set_user_attr("neutral_weight_range", self.nw_range)


    def prepare(self):

        ds_path = f"/home/bigtech/data/verbosius/amazon/nc_{self.n_chunks}"

        preprocess = Preprocess(ds_path=ds_path, part_n=self.part_n, progress_bar=True, force_write=True)
        preprocess.run()
        del preprocess

        weighter = Weighter(part_n=self.part_n, progress_bar=True, force_write=True)
        weighter.run()
        del weighter

        trainingdata = Trainingdata(part_n=self.part_n, progress_bar=True, force_write=True)
        trainingdata.run()
        del trainingdata

        collect()    


    def objective(self, trial):

        config.learning_rate = trial.suggest_float("learning_rate", self.lr_range[0], self.lr_range[1])
        config.neutral_weight = trial.suggest_float("neutral_weight", self.nw_range[0], self.nw_range[1])
        config.loss_weight = trial.suggest_float("loss_weight", self.lw_range[0], self.lw_range[1])
        
        transformer = Transformer(part_n=self.part_n, 
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
    

    def optimize(self):

        self.study.optimize(self.objective, n_trials=self.n_trials)

        print(self.study.best_params)
        print(self.study.best_value)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_chunks", type=int)
    parser.add_argument("--part_n", type=int)
    parser.add_argument("--n_trials", type=int)
    parser.add_argument("--study_name", type=str)
    args = parser.parse_args()

    n_chunks = args.n_chunks
    part_n = args.part_n
    n_trials = args.n_trials
    study_name = args.study_name

    hpsearch = HyperparamSearch(n_chunks=n_chunks,
                                part_n=part_n,
                                n_trials=n_trials,
                                study_name=study_name)
    
    hpsearch.prepare()

    hpsearch.optimize()
    