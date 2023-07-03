import os

import pandas as pd

from itertools import Counter

class IMDB:

    def __init__(self, two_cat : bool):
        
        self.two_cat = two_cat



    def load_data(self, path: str, batch: tuple, test: bool = False):

        root = os.path.expanduser('~')
        path = os.path.join(root, "projects", path)
        
        if test:
            df_train = pd.read_csv(os.path.join(path, "imdb_train.csv")).reset_index(drop=True)
            df_test = pd.read_csv(os.path.join(path, "imdb_test.csv")).reset_index(drop=True)
            train_data = df_train.values.tolist()
            test_data = df_test.values.tolist()

            train_x = [x[0] for x in train_data[batch[0]:batch[1]]]
            train_y = [x[1] for x in train_data[batch[0]:batch[1]]]
            test_x = [x[0] for x in test_data[batch[0]:batch[1]]]
            test_y = [x[1] for x in test_data[batch[0]:batch[1]]]
            
            return train_x, train_y, test_x, test_y

        else:
            df_train = pd.read_csv(os.path.join(path, "imdb_train.csv")).reset_index(drop=True)
            train_data = df_train.values.tolist()
            train_x = [x[0] for x in train_data[batch[0]:batch[1]]]
            train_y = [x[1] for x in train_data[batch[0]:batch[1]]]

            return train_x, train_y
            

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
    

def batch_data(dataset, n_batches_per_mix : int, batch_size : int, path : str, test : bool = False):
    total_mix_size = n_batches_per_mix * batch_size

    un_batched_mix = dataset.load_data(path, (0, total_mix_size), test=True)

    if test:
        train_x, train_y, test_x, test_y = un_batched_mix

        





    else:
        train_x, train_y = un_batched_mix
