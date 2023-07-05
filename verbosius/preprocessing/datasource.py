import os
import random

import pandas as pd
import numpy as np

from collections import Counter

class IMDB:

    def __init__(self, two_cat : bool):
        
        self.two_cat = two_cat

    def load_data(self, path: str):

        root = os.path.expanduser('~')
        path = os.path.join(root, "projects", path)
        
        df_train = pd.read_csv(os.path.join(path, "imdb_train.csv")).reset_index(drop=True)
        df_test = pd.read_csv(os.path.join(path, "imdb_test.csv")).reset_index(drop=True)
        train_data = np.array(df_train) # [[text, label], [text, label], ...]
        test_data = np.array(df_test) 

        return train_data, test_data

            
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

    un_batched_mix = dataset.load_data(path, test=True)

    train_data, test_data = un_batched_mix

    texts_train = train_data[:, 0]
    labels_train = train_data[:, 1]

    indices_class_0_train = np.where(labels_train==0)[0]
    indices_class_1_train = np.where(labels_train==1)[0]

    min_count = batch_size//2 #min(num_elements_0, num_elements_1)

    np.random.shuffle(indices_class_0_train)
    np.random.shuffle(indices_class_1_train)

    split_ind_0_train = np.array_split(indices_class_0_train[:min_count*n_batches_per_mix], n_batches_per_mix)
    split_ind_1_train = np.array_split(indices_class_1_train[:min_count*n_batches_per_mix], n_batches_per_mix)


    train_x = []
    train_y = []
    for i in range(n_batches_per_mix):
        split_ind = np.concatenate((split_ind_0_train[i], split_ind_1_train[i]))
        split_text = texts_train[split_ind]
        split_label = labels_train[split_ind]

        train_x.append(split_text)
        train_y.append(split_label)


    if test:
        texts_test = test_data[:, 0]
        labels_test = test_data[:, 1]
        indices_class_0_test = np.where(labels_test==0)[0]
        indices_class_1_test = np.where(labels_test==1)[0]

        np.random.shuffle(indices_class_0_test)
        np.random.shuffle(indices_class_1_test)

        split_ind_0_test = np.array_split(indices_class_0_test[:min_count*n_batches_per_mix], n_batches_per_mix)
        split_ind_1_test = np.array_split(indices_class_1_test[:min_count*n_batches_per_mix], n_batches_per_mix)
        test_x = []
        test_y = []
        for i in range(n_batches_per_mix):
            split_ind = np.concatenate((split_ind_0_test[i], split_ind_1_test[i]))
            split_text = texts_test[split_ind]
            split_label = labels_test[split_ind]

            test_x.append(split_text)
            test_y.append(split_label)

        return train_x, train_y, test_x, test_y
    return train_x, train_y, None, None