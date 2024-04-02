import os

import optuna
import numpy as np
import green_tsetlin as gt
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from chunking.stage_chunks import stage_chunks
from preprocess.stage_preprocess import stage_preprocess
from trainingdata.stage_trainingdata import stage_trainingdata
from xai_transformer.stage_transformer import stage_transformer
from xai_validation.stage_validation import stage_validation
from performance import model_accuracy
import utils.config as config


def objective(trial):

    print("Trial: ", trial.number)

    config.MAX_FEATURES = trial.suggest_int("MAX_FEATURES", 1000, 3000, step=100)
    config.MAX_DF = trial.suggest_float("MAX_DF", 0.4, 0.8, step=0.01)
    config.MIN_DF = trial.suggest_int("MIN_DF", 1, 10, step=1)
    config.NUMBER_OF_CLAUSES = trial.suggest_int("NUMBER_OF_CLAUSES", 1000, 7000, step=100)
    config.S = trial.suggest_float("S", 3, 30, step=0.1)
    config.T = trial.suggest_int("T", 1000, 7000, step=100)

    config.ERROR_MAX_FEATURES = trial.suggest_int("ERROR_MAX_FEATURES", 100, 2000, step=100)
    config.ERROR_NUMBER_OF_CLAUSES = trial.suggest_int("ERROR_NUMBER_OF_CLAUSES", 500, 5000, step=100)
    config.ERROR_S = trial.suggest_float("ERROR_S", 3, 30, step=0.1)
    config.ERROR_T = trial.suggest_int("ERROR_T", 500, 5000, step=100)
    config.ERROR_MAX_DF = trial.suggest_float("ERROR_MAX_DF", 0.4, 0.8, step=0.01)
    config.ERROR_MIN_DF = trial.suggest_int("ERROR_MIN_DF", 1, 10, step=1)
    SKB_score_func = trial.suggest_categorical("SKB_score_func", ["chi2", "f_classif", "mutual_info_classif"])

    config.SKB_score_func = eval(SKB_score_func)


    config.STOPWORDS = trial.suggest_categorical("STOPWORDS", [None, "english"])
    # config.n_badtexts = trial.suggest_int("n_badtexts", 1000, 5000, step=100)

    correct_x = stage_trainingdata(dataset=config.dataset,
                       chunkdist_n=config.chunkdist_n)

    acc = correct_x / (config.chunk_size * config.chunk_amount)
    
    os.system(f"rm -rf {config.root}/{config.dataset}/trainingdata/{config.dataset}_chunkdist_{config.chunkdist_n}/")

    return acc

if __name__ == "__main__":
    
    config.root = f"/home/{config.user}/data/verbosius/hpsearch_env/"
    config.seed = 42

    config.chunkdist_n = 19404
    config.dataset = "amazon"
    config.chunk_size = 8000
    config.chunk_amount = 25
    config.TM_EPOCHS = 50
    
    study = optuna.create_study(study_name="TM_param_search_official_cs8000_ca75_final", direction="maximize", storage=f"sqlite:////home/{config.user}/projects/verbosius/sqlite3.db", load_if_exists=True)
    
    # stage_chunks(dataset=config.dataset,
    #              chunk_size=config.chunk_size,
    #              chunk_amount=config.chunk_amount,
    #              chunkdist_n=config.chunkdist_n)
    
    # stage_preprocess(dataset=config.dataset,
    #                  chunkdist_n=config.chunkdist_n)

    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    print(study.best_params)
    print(study.best_value)