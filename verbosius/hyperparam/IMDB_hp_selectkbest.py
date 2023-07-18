import optuna
import pickle
import argparse
import os

import numpy as np
import green_tsetlin as gt

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
from warnings import simplefilter

simplefilter(action='ignore', category=UserWarning)
best_model = None
best_vocabulary = None
best_result = 0.0




def objective(trial, train_x, train_y, test_x, test_y, n_jobs):

    train_x_store = train_x
    train_y_store = train_y
    test_x_store = test_x
    test_y_store = test_y
    
    
    num_clauses = trial.suggest_int("num_clauses", 4000, 10000)
    s = trial.suggest_float("s", 3, 10)
    threshold = 9225
    literal_budget = trial.suggest_int("literal_budget", 5, 20)
    max_df = 0.5232
    min_df = 21
    ngram_range = (1,2)
    stop_words = None

    vectorizer = CountVectorizer(max_features=30000,
                                 max_df=max_df, 
                                 min_df=min_df,
                                 ngram_range=ngram_range,
                                 binary=True,
                                 dtype=np.uint8,
                                 stop_words = stop_words)
    


    train_x = vectorizer.fit_transform([" ".join(x) for x in train_x_store]).todense()
    test_x = vectorizer.transform([" ".join(x) for x in test_x_store]).todense()


    SKB = SelectKBest(chi2, k=5000)
    SKB.fit(train_x, train_y)
    selected_features = SKB.get_support(indices=True)
    train_x = SKB.transform(train_x).toarray()
    test_x = SKB.transform(test_x).toarray()

    

    tm = gt.TsetlinMachine(n_literals=train_x.shape[1],
                           n_clauses=num_clauses, 
                           n_classes=2,
                           s=s,
                           n_literal_budget=literal_budget)
    
    tm.set_train_data(train_x, train_y)
    tm.set_test_data(test_x, test_y)

    trainer = gt.Trainer(threshold=threshold, 
                         n_epochs=7, # turned down from 10, as all the good results were found in the first 7 epochs in the first run
                         n_jobs=n_jobs,
                         early_exit_acc=1.0)
    
    output = trainer.train(tm)

    trial.set_user_attr("best_test_epoch", output["best_test_epoch"])

    return output["best_test_score"]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="IMDB", help="Which dataset to use")
    parser.add_argument("--batch_dist", type=int, default=0, help="Which batch number to use")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for hyperparam search")
    parser.add_argument("--n_jobs", type=int, default=2, help="Number of parallel jobs")
    parser.add_argument("--data_amount", type=float, default=1.0, help="Fraction of data to use")

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

    study = optuna.create_study(study_name="imdb_hp_1_tm_selectkbest", direction="maximize", storage="sqlite:///imdb_tm_skb.db", load_if_exists=True)
    study.optimize(obj_func, n_trials=args.n_trials, show_progress_bar=True)

    print(study.best_params)
    print(study.best_value)
