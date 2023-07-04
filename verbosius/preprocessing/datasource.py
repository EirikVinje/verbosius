import os
import random

import pandas as pd
import numpy as np


class IMDB:

    def __init__(self, two_cat : bool):
        
        self.two_cat = two_cat



    def load_data(self, path: str, batch: tuple, test: bool = False, shuffle: bool = False, seed: int = 42):

        root = os.path.expanduser('~')
        path = os.path.join(root, "projects", path)
        
        rng = np.random.default_rng(seed)

        if test:
            df_train = pd.read_csv(os.path.join(path, "imdb_train.csv")).reset_index(drop=True)
            df_test = pd.read_csv(os.path.join(path, "imdb_test.csv")).reset_index(drop=True)
            train_data = df_train.values.tolist()
            test_data = df_test.values.tolist()

            
            
            return train_data, test_data

        else:
            df_train = pd.read_csv(os.path.join(path, "imdb_train.csv")).reset_index(drop=True)
            train_data = df_train.values.tolist()


            return train_data
            

class RottenTomatoes:

    def __init__(self, two_cat : bool, batch : tuple):
        
        self.two_cat = two_cat
        self.batch = batch

    def load_data(self, path: str):

        return


class Amazon:

    def __init__(self, two_cat : bool, batch : tuple) -> None:
        
        self.two_cat = two_cat
        self.batch = batch

    def load_data(self, path: str):

        return 
    

def dataset(dataset : str):

    if dataset == "imdb":
        return IMDB
    elif dataset == "rottentomatoes":
        return RottenTomatoes
    elif dataset == "amazon":
        return Amazon
    else:
        raise ValueError("No such dataset exists")
    

def batch_data(dataset, n_batches_per_mix : int, batch_size : int, start_point:int, path : str, test : bool = False, shuffle : bool = False, seed : int = 42):
    total_mix_size = n_batches_per_mix * batch_size + start_point

    un_batched_mix = dataset.load_data(path, (start_point, total_mix_size), test=True, shuffle=shuffle, seed=seed)


    if test:
        train_x, train_y, test_x, test_y = un_batched_mix

        train_x_batched = []
        train_y_batched = []
        test_x_batched = []
        test_y_batched = []

        for i in range(n_batches_per_mix):
            train_x_batched.append(train_x[i*batch_size:(i+1)*batch_size])
            train_y_batched.append(train_y[i*batch_size:(i+1)*batch_size])
            test_x_batched.append(test_x[i*batch_size:(i+1)*batch_size])
            test_y_batched.append(test_y[i*batch_size:(i+1)*batch_size])

        return train_x_batched, train_y_batched, test_x_batched, test_y_batched


    else:
        train_x, train_y = un_batched_mix

        train_x_batched = []
        train_y_batched = []

        for i in range(n_batches_per_mix):
            train_x_batched.append(train_x[i*batch_size:(i+1)*batch_size])
            train_y_batched.append(train_y[i*batch_size:(i+1)*batch_size])

        return train_x_batched, train_y_batched