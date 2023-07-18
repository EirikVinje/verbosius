import optuna
import pickle
import argparse
import os

import numpy as np
import green_tsetlin as gt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
from warnings import simplefilter

simplefilter(action='ignore', category=UserWarning)

def objective(trial, train_x, train_y, test_x, test_y, n_jobs):

    train_x_store = train_x
    train_y_store = train_y
    test_x_store = test_x
    test_y_store = test_y
    


    tot_num_clauses = 5000
    n_base_clauses = 1000
    n_free_clauses = tot_num_clauses - 3*n_base_clauses
    threshold = 9225
    
    num_clauses1 = trial.suggest_int("num_clauses1", 0, n_free_clauses)
    num_clauses2 = trial.suggest_int("num_clauses2", 0, max(n_free_clauses-num_clauses1, 0))
    num_clauses3 = trial.set_user_attr("num_clauses3", [max(n_free_clauses - num_clauses1 - num_clauses2, 0)])
    num_clause_list = [num_clauses1, num_clauses2, trial.user_attrs["num_clauses3"][0]]
    print(trial.user_attrs)
    num_clause_list = sorted(num_clause_list)
    print(num_clause_list)

    s1 = trial.suggest_float("s1", 3, 10)
    s2 = trial.suggest_float("s2", 3, 10)
    s3 = trial.suggest_float("s3", 3, 10)
    s_list = [s1, s2, s3]

    literal_budget1 = 4
    literal_budget2 = 8
    literal_budget3 = 16
    literal_budget_list = [literal_budget1, literal_budget2, literal_budget3]

    max_df = 0.5263 #trial.suggest_float("max_df", 0.5, 0.9)
    min_df = 21 #trial.suggest_int("min_df", 10, 25)
    ngram_range = (1,2) #trial.suggest_categorical("ngram_range", [(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)])
    stop_words = None #trial.suggest_categorical("stop_words", [None, "english"])

    vectorizer = CountVectorizer(max_features=5000,
                                 max_df=max_df, 
                                 min_df=min_df,
                                 ngram_range=ngram_range,
                                 binary=True,
                                 dtype=np.uint8,
                                 stop_words = stop_words)
    
    train_x = vectorizer.fit_transform([" ".join(x) for x in train_x_store]).todense()
    test_x = vectorizer.transform([" ".join(x) for x in test_x_store]).todense()


    TMs = []
    for i in range(3):
        tm = gt.TsetlinMachine(n_literals=train_x.shape[1],
                                n_clauses=n_base_clauses+num_clause_list[i], 
                                n_classes=2,
                                s=s_list[i],
                                n_literal_budget=literal_budget_list[i])
        
        tm.set_train_data(train_x, train_y)
        tm.set_test_data(test_x, test_y)
        train_y = None
        test_y = None

        TMs.append(tm)


    trainer = gt.Trainer(threshold=threshold, 
                         n_epochs=7, # turned down from 10, as all the good results were found in the first 7 epochs in the first run
                         n_jobs=n_jobs,
                         early_exit_acc=1.0)
    
    output = trainer.train(TMs)

    trial.set_user_attr("best_test_epoch", output["best_test_epoch"])

    return output["best_test_score"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="IMDB", help="Which dataset to use")
    parser.add_argument("--batch_dist", type=int, default=0, help="Which batch number to use")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for hyperparam search")
    parser.add_argument("--n_jobs", type=int, default=2, help="Number of parallel jobs")
    parser.add_argument("--data_amount", type=float, default=.2, help="Fraction of data to use")

    args = parser.parse_args()
    
    path = f"/home/bigtech/data/verbosius/store_imdb_pickle/imdb_batchdist_{args.batch_dist}"

    train_x_t = []
    train_y_t = []
    test_x_t = []
    test_y_t = []
    for file in os.listdir(path):
        with open(f"{path}/{file}", "rb") as f:
            data = pickle.load(f)

        train_x_t.extend([instance["lemmas"] for instance in data["train"]])
        train_y_t.extend([instance["label"] for instance in data["train"]])
        test_x_t.extend([instance["lemmas"] for instance in data["test"]])
        test_y_t.extend([instance["label"] for instance in data["test"]])

    if args.data_amount < 1:
        train_x, _, train_y, _ = train_test_split(train_x_t, train_y_t, train_size=args.data_amount, stratify=train_y_t, random_state=42)
        test_x, _, test_y, _ = train_test_split(test_x_t, test_y_t, train_size=args.data_amount, stratify=test_y_t, random_state=42)
    else:
        train_x = train_x_t
        train_y = train_y_t
        test_x = test_x_t
        test_y = test_y_t

    train_y = np.array(train_y, dtype=np.uint32)
    test_y = np.array(test_y, dtype=np.uint32)

    
    obj_func = lambda trial: objective(trial, train_x, train_y, test_x, test_y, args.n_jobs)

    study = optuna.create_study(study_name="imdb_hp_3_tms", direction="maximize", storage="sqlite:///imdb_tm3.db", load_if_exists=True)
    study.optimize(obj_func, n_trials=args.n_trials, show_progress_bar=True)

    print(study.best_params)
    print(study.best_value)
    with open(f"/home/bigtech/data/verbosius/hp_studies/imdb_study_batchdist_{args.batch_dist}_ntrials_{args.n_trials}_acc_{int(study.best_value*100000)}.pkl", "wb") as f:
        pickle.dump(study, f)
