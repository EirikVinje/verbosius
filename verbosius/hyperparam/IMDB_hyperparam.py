import optuna
import pickle
import argparse
import os

import numpy as np
import green_tsetlin as gt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter

def objective(trial, train_x, train_y, test_x, test_y, n_jobs, ):
    num_clauses = trial.suggest_int("num_clauses", 4000, 12000)
    s = trial.suggest_int("s", 1, 100)
    threshold = trial.suggest_int("threshold", 100, 1000)
    literal_budget = trial.suggest_int("literal_budget", 1, 100)
    n_epochs = trial.suggest_int("n_epochs", 10, 25)

    tm = gt.TsetlinMachine(n_literals=train_x.shape[1],
                           n_clauses=num_clauses, 
                           n_classes=2,
                           s=s,
                           n_literal_budget=literal_budget)
    
    tm.set_train_data(train_x, train_y)
    tm.set_test_data(test_x, test_y)

    trainer = gt.Trainer(threshold=threshold, 
                         n_epochs=n_epochs,
                         n_jobs=n_jobs,
                         early_exit_acc=0.86)
    
    output = trainer.train(tm)

    return output["best_test_score"]

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dist", type=int, default=0, help="Which batch number to use")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for hyperparam search")
    parser.add_argument("--n_jobs", type=int, default=2, help="Number of parallel jobs")
    parser.add_argument("--data_amount", type=float, default=.2, help="Fraction of data to use")

    args = parser.parse_args()
    batch_dist = args.batch_dist

    path = f"/home/bigtech/data/verbosius/store_imdb_pickle/imdb_batchdist_{batch_dist}"
    
    train_x_t = []
    train_y_t = []
    test_x_t = []
    test_y_t = []
    for file in os.listdir(path):
        print("y")
        with open(f"{path}/{file}", "rb") as f:
            data = pickle.load(f)

        train_x_t.extend([instance["lemmas"] for instance in data["train"]])
        train_y_t.extend([instance["label"] for instance in data["train"]])
        test_x_t.extend([instance["lemmas"] for instance in data["test"]])
        test_y_t.extend([instance["label"] for instance in data["test"]])
    


    train_x, _, train_y, _ = train_test_split(train_x_t, train_y_t, train_size=args.data_amount, stratify=train_y_t)
    test_x, _, test_y, _ = train_test_split(test_x_t, test_y_t, train_size=args.data_amount, stratify=test_y_t)


    train_y = np.array(train_y, dtype=np.uint32)
    test_y = np.array(test_y, dtype=np.uint32)
    


    vectorizer = CountVectorizer(max_features=5000,
                                 max_df=0.7, 
                                 min_df=10,
                                 ngram_range=(1,2),
                                 binary=True,
                                 dtype=np.uint8,
                                 stop_words = 'english')
    
    train_x_bin = vectorizer.fit_transform([" ".join(x) for x in train_x]).todense()
    test_x_bin = vectorizer.transform([" ".join(x) for x in test_x]).todense()


    
    obj_func = lambda trial: objective(trial, train_x_bin, train_y, test_x_bin, test_y, args.n_jobs)

    study = optuna.create_study(direction="maximize")
    study.optimize(obj_func, n_trials=args.n_trials, show_progress_bar=True)
