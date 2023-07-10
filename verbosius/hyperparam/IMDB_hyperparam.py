import optuna
import pickle

import numpy as np
import green_tsetlin as gt

from sklearn.feature_extraction.text import CountVectorizer

def objective(trial, train_x, train_y, test_x, test_y):
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
                         n_jobs=2,
                         early_exit_acc=0.865)
    
    output = trainer.train(tm)

    return output["best_test_score"]

if __name__ == '__main__':


    path = "/home/bigtech/data/verbosius/store_imdb_pickle/"
    batch_num = _
    data = pickle.load(open(f"{path}/batch_{batch_num}.pkl", "rb")) 

    train_x = [instance["lemmas"] for instance in data["train"]]
    train_y = [instance["label"] for instance in data["train"]]
    test_x = [instance["lemmas"] for instance in data["test"]]
    test_y = [instance["label"] for instance in data["test"]]
    
    train_y = np.array(train_y, dtype=np.uint32)
    test_y = np.array(test_y, dtype=np.uint32)

    vectorizer = CountVectorizer(max_features=5000,
                                 max_df=0.7, 
                                 min_df=10,
                                 ngram_range=(1,2),
                                 binary=True,
                                 dtype=np.uint8,
                                 stop_words = 'english')
    
    train_x_bin = vectorizer.fit_transform([" ".join(x) for x in train_x])
    test_x_bin = vectorizer.transform([" ".join(x) for x in test_x])


    
    obj_func = lambda trial: objective(trial, train_x_bin, train_y, test_x_bin, test_y)

    study = optuna.create_study(direction="maximize")
    study.optimize(obj_func, n_trials=2, show_progress_bar=True)