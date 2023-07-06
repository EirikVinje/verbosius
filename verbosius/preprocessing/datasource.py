import os
import random
import itertools

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from collections import Counter

class IMDB:

    def __init__(self, two_cat : bool):
        
        self.two_cat = two_cat



    def load_data(self, path: str, test: bool = False):

        root = os.path.expanduser('~')
        path = os.path.join(root, "projects", path)

        if test:
            df_train = pd.read_csv(os.path.join(path, "imdb_train.csv")).reset_index(drop=True)
            df_test = pd.read_csv(os.path.join(path, "imdb_test.csv")).reset_index(drop=True)
            train_data = np.array(df_train)
            test_data = np.array(df_test)

            return train_data, test_data

        else:
            df_train = pd.read_csv(os.path.join(path, "imdb_train.csv")).reset_index(drop=True)
            train_data = df_train.values.to_numpy()


            return train_data, None
            

class MNIST:

    def __init__(self, two_cat : bool):
        self.two_cat = two_cat

    def load_data(self, path: str, test: bool = False):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

        if test:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            x_train = np.where(x_train.reshape((x_train.shape[0], 28 * 28)) > 75, 1, 0)
            x_test = np.where(x_test.reshape((x_test.shape[0], 28 * 28)) > 75, 1, 0)
            x_train = x_train.astype(np.uint8)
            x_test = x_test.astype(np.uint8)
            y_train = y_train.astype(np.int32)
            y_test = y_test.astype(np.int32)

            train_data = np.column_stack((x_train, y_train))
            test_data = np.column_stack((x_test, y_test))
            print(y_train[0])
            print(train_data[0])
            return train_data, test_data
        
        return np.concatenate((X, y), axis=1), None


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
    elif dataset == "mnist":
        return MNIST
    else:
        raise ValueError("No such dataset exists")
    

def batch_data(dataset, n_batches_per_mix : int, batch_size : int, path : str, test : bool = False, shuffle : bool = True, seed : int = 42):
    """
    left for testing purposes, batch_data_multiclass is the update version that should be used. 
    """

    un_batched_mix = dataset.load_data(path, test=True)



    train_data, test_data = un_batched_mix

    texts_train = train_data[:, 0]
    labels_train = train_data[:, 1]

    indices_class_0_train = np.where(labels_train==0)[0]
    indices_class_1_train = np.where(labels_train==1)[0]

    min_count = batch_size//2 #min(num_elements_0, num_elements_1)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices_class_0_train)
        rng.shuffle(indices_class_1_train)

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

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices_class_0_test)
            rng.shuffle(indices_class_1_test)

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



def batch_data_multiclass(dataset, n_batches_per_mix : int, batch_size : int, path : str, test : bool = False, shuffle : bool = True, seed : int = 42):


    un_batched_mix = dataset.load_data(path, test=True)


    train_data, test_data = un_batched_mix

    texts_train = train_data[:, 0]
    labels_train = train_data[:, 1]
    n_classes = len(np.unique(labels_train))
    print(n_classes)

    indicies_class_train = []
    for i in range(n_classes):
        indicies_class_train.append(np.where(labels_train==i)[0])

    min_count = batch_size//n_classes #min(num_elements_0, num_elements_1)

    if shuffle:
        rng = np.random.default_rng(seed)
        for indicies in indicies_class_train:
            rng.shuffle(indicies)

    split_ind_train = []
    for i in range(n_classes):
        split_ind_train.append(np.array_split(indicies_class_train[i][:min_count*n_batches_per_mix], n_batches_per_mix))

    train_x = []
    train_y = []
    for i in range(n_batches_per_mix):
        split_ind = np.array([], dtype=int)
        for index in range(n_classes - 1):
            temp = np.concatenate((split_ind_train[index][i], split_ind_train[index+1][i]))
            split_ind = np.concatenate((split_ind, temp))

        split_text = texts_train[split_ind]
        split_label = labels_train[split_ind]

        train_x.append(split_text)
        train_y.append(split_label)

    print(train_y)
    if test:
        texts_test = test_data[:, 0]
        labels_test = test_data[:, 1]


        indicies_class_test = []
        for i in range(n_classes):
            indicies_class_test.append(np.where(labels_test==i)[0])

        min_count = batch_size//n_classes #min(num_elements_0, num_elements_1)

        if shuffle:
            rng = np.random.default_rng(seed)
            for indicies in indicies_class_test:
                rng.shuffle(indicies)

        split_ind_test = []
        for i in range(n_classes):
            split_ind_test.append(np.array_split(indicies_class_test[i][:min_count*n_batches_per_mix], n_batches_per_mix))


        test_x = []
        test_y = []
        for i in range(n_batches_per_mix):
            
            split_ind = np.array([], dtype=int)
            for index in range(n_classes - 1):
                temp = np.concatenate((split_ind_test[index][i], split_ind_test[index+1][i]))
                split_ind = np.concatenate((split_ind, temp))
            split_text = texts_test[split_ind]
            split_label = labels_test[split_ind]

            test_x.append(split_text)
            test_y.append(split_label)
        
        print(Counter(train_y[0]))
        assert False
        return train_x, train_y, test_x, test_y, n_classes

    return train_x, train_y, None, None, n_classes