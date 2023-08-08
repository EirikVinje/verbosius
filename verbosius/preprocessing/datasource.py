import os
import random
import itertools

import datasets as ds
import pandas as pd
import numpy as np

from time import perf_counter
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from collections import Counter

class IMDB:

    def __init__(self, two_cat : bool):
        
        self.two_cat = two_cat
        self.exists_test_set = True
        self.exists_validation_set = False
        self.n_classes = 2


    def load_data(self, path: str, test: bool = False, test_size: float = 0.2):

        """root = os.path.expanduser('~')
        path = os.path.join(root, "projects", path)
        
        df_train = pd.read_csv(os.path.join(path, "imdb_train.csv")).reset_index(drop=True)
        df_test = pd.read_csv(os.path.join(path, "imdb_test.csv")).reset_index(drop=True)
        train_data = np.array(df_train) # [[text, label], [text, label], ...]
        test_data = np.array(df_test) """
        dataset = ds.load_dataset("imdb")
        train_data = dataset["train"]
        test_data = dataset["test"]

        train_x, train_y, test_x, test_y = [], [], [], []
        for i in range(len(train_data)):
            train_x.append(train_data[i]["text"])
            train_y.append(train_data[i]["label"])
        for i in range(len(test_data)):
            test_x.append(test_data[i]["text"])
            test_y.append(test_data[i]["label"])
        
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)

        train_x = train_x.astype(object)
        test_x = test_x.astype(object)
        train_y = train_y.astype(np.uint8)
        test_y = test_y.astype(np.uint8)

        train_data = np.column_stack((train_x, train_y))
        test_data = np.column_stack((test_x, test_y))

        return train_data, test_data, None

            

class MNIST:
    """
    Exists for testing purposes, not intended for use in the pipeline, will crash if used because not text data.
    """
    def __init__(self, two_cat : bool):
        self.two_cat = two_cat
        self.exists_test_set = False
        self.exists_validation_set = False
        self.n_classes = 10

    def load_data(self, path: str, test: bool = False, test_size: float = 0.2):
        t0 = perf_counter()
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, )

        if test:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            x_train = np.where(x_train.reshape((x_train.shape[0], 28 * 28)) > 75, 1, 0)
            x_test = np.where(x_test.reshape((x_test.shape[0], 28 * 28)) > 75, 1, 0)
            x_train = x_train.astype(np.uint8)
            x_test = x_test.astype(np.uint8)
            y_train = y_train.astype(np.int32)
            y_test = y_test.astype(np.int32)

            # train_data = np.column_stack((x_train, y_train))
            train_data = []
            for i in range(len(x_train)):
                train_data.append([x_train[i].tolist(), y_train[i]])
            train_data = np.asarray(train_data, dtype=object)

            test_data = []
            for i in range(len(x_test)):
                test_data.append([x_test[i].tolist(), y_test[i]])
            test_data = np.asarray(test_data, dtype=object)

            t1 = perf_counter()
            print(f"Time to load MNIST: {t1 - t0:.2f} seconds")
            return train_data, test_data

        return np.concatenate((X, y), axis=1), None, None


class RottenTomatoes:

    def __init__(self, two_cat : bool):
        
        self.two_cat = two_cat
        self.exists_test_set = True
        self.exists_validation_set = True
        self.n_classes = 2

    def load_data(self, path: str,test: bool = False, test_size: float = 0.2):

        rt = ds.load_dataset("rotten_tomatoes")
        train_data = np.array(rt["train"])
        test_data = np.array(rt["test"])
        val_data = np.array(rt["validation"])


        train_x, train_y, test_x, test_y, val_x, val_y = [], [], [], [], [], []
        for i in range(len(train_data)):
            train_x.append(train_data[i]["text"])
            train_y.append(int(train_data[i]["label"]))
        for i in range(len(test_data)):
            test_x.append(test_data[i]["text"])
            test_y.append(int(test_data[i]["label"]))
        for i in range(len(val_data)):
            val_x.append(val_data[i]["text"])
            val_y.append(int(val_data[i]["label"]))

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)
        val_x = np.asarray(val_x)
        val_y = np.asarray(val_y)


        train_x = train_x.astype(object)
        test_x = test_x.astype(object)
        val_x = val_x.astype(object)
        train_y = train_y.astype(np.uint8)
        test_y = test_y.astype(np.uint8)
        val_y = val_y.astype(np.uint8)

        train_data = np.column_stack((train_x, train_y))
        test_data = np.column_stack((test_x, test_y))
        val_data = np.column_stack((val_x, val_y))

        return train_data, test_data, val_data

class SST5:

    def __init__(self, two_cat : bool):
        
        self.two_cat = two_cat
        self.exists_test_set = True
        self.exists_validation_set = True
        self.n_classes = 5

    def load_data(self, path: str, test: bool = False, test_size: float = 0.2):
        dataset = ds.load_dataset("SetFit/sst5")
        train_data = np.array(dataset["train"])
        test_data = np.array(dataset["test"])
        val_data = np.array(dataset["validation"])
        
        train_x, train_y, test_x, test_y, val_x, val_y = [], [], [], [], [], []
        train_all_labels, test_all_labels, val_all_labels = [], [], []
        combine_labels = {
            0: 0,
            1: 0,
            2: 1,
            3: 2,
            4: 2
        }
        for i in range(len(train_data)):
            train_x.append(train_data[i]["text"])
            train_y.append(combine_labels[int(train_data[i]["label"])])
            train_all_labels.append(int(train_data[i]["label"]))
        for i in range(len(test_data)):
            test_x.append(test_data[i]["text"])
            test_y.append(combine_labels[int(test_data[i]["label"])])
            test_all_labels.append(int(test_data[i]["label"]))
        for i in range(len(val_data)):
            val_x.append(val_data[i]["text"])
            val_y.append(combine_labels[int(val_data[i]["label"])])
            val_all_labels.append(int(val_data[i]["label"]))
        

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        train_all_labels = np.asarray(train_all_labels)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)
        test_all_labels = np.asarray(test_all_labels)
        val_x = np.asarray(val_x)
        val_y = np.asarray(val_y)
        val_all_labels = np.asarray(val_all_labels)

        train_x = train_x.astype(object)
        test_x = test_x.astype(object)
        val_x = val_x.astype(object)
        train_y = train_y.astype(np.uint8)
        test_y = test_y.astype(np.uint8)
        val_y = val_y.astype(np.uint8)

        self.train_all_labels = train_all_labels.astype(np.uint8)
        self.test_all_labels = test_all_labels.astype(np.uint8)
        self.val_all_labels = val_all_labels.astype(np.uint8)

        train_data = np.column_stack((train_x, train_y))
        test_data = np.column_stack((test_x, test_y))
        val_data = np.column_stack((val_x, val_y))
        

        return train_data, test_data, val_data


class Amazon:

    def __init__(self, two_cat : bool) -> None:
        
        self.two_cat = two_cat
        self.exists_test_set = False
        self.exists_validation_set = False

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
    elif dataset =="sst5":
        return SST5
    else:
        raise ValueError("No such dataset exists")
    




def shuffle_unison(a: list, seed : int = 42):
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(a[0]))
    if len(a) == 3:
        return a[0][p], a[1][p], a[2][p]
    return a[0][p], a[0][p]

def chunk_data(n_chunks_per_mix, n_classes, split_ind_input, texts, labels, dataset, seed):
    train_x = []
    train_y = []
    train_y_orig = []
    for i in range(n_chunks_per_mix):
        split_ind = np.array([], dtype=int)
        split_ind = np.concatenate((split_ind, split_ind_input[0][i]))
        for index in range(1, n_classes):
            temp = split_ind_input[index][i]
            split_ind = np.concatenate((split_ind, temp))

        split_text = texts[split_ind]
        split_label = labels[split_ind]
        
        if dataset.n_classes > 3:
            split_label_orig = dataset.train_all_labels[split_ind]
            split_text, split_label, split_label_orig = shuffle_unison([split_text, split_label, split_label_orig], seed)
            
            train_x.append(split_text)
            train_y.append(split_label)
            train_y_orig.append(split_label_orig)

        else:
            split_text, split_label = shuffle_unison([split_text, split_label], seed)

            train_x.append(split_text)
            train_y.append(split_label)


    if dataset.n_classes > 3:
        return train_x, train_y, train_y_orig
    return train_x, train_y, None



def chunk_data_multiclass(dataset, 
                          n_chunks_per_mix : int, 
                          chunk_size : int, 
                          path : str, 
                          validation : bool = True,
                          test_chunk_size: int = -1,
                          test_size: float = 0.2, 
                          val_chunk_size: int = -1,
                          val_size: float = 0.5, shuffle : bool = True, seed : int = 42): # n_chunks_per_mix: int = -1, n_chunks_per_mix: int = -1, 
    
    """
    dataset : dataset class
    n_chunks_per_mix : number of chunks per mix
    chunk_size : number of samples per chunk
    path : path to dataset
    test : whether to return test data
    dataset.exists_test_set : whether to use test set for training
    test_chunk_size : number of samples per test chunk, default is same as chunk_size but can be changed if you want different chunk size for the test data 
    
    test_size : size of test data, percentage of total data
    dataset.exists_validation_set : whether to use validation set for training
    val_chunk_size : number of samples per validation chunk, default is same as chunk_size but can be changed if you want different chunk size for the validation data
    
    val_size : size of validation data if no set is provided and the sizes aren't specified. Using this variable will split the TEST DATA into two parts, one for validation and one for testing
    shuffle : whether to shuffle data
    seed : random seed

    returns:
    train_x : training data
    train_y : training labels
    train_y_orig : original training labels, None if no original labels
    test_x : test data
    test_y : test labels
    test_y_orig : original test labels, None if no original labels
    val_x : validation data, None if no val data
    val_y : validation labels, None if no val data
    val_y_orig : original validation labels, None if no original labels
    """



    orig_chunk_size = chunk_size

    if not dataset.exists_test_set and test_chunk_size == -1:        
        test_chunk_size = int(orig_chunk_size*test_size)
        chunk_size = chunk_size - test_chunk_size
    elif test_chunk_size == -1:
        test_chunk_size = chunk_size



    if not dataset.exists_validation_set and val_chunk_size == -1:
        val_chunk_size = int(orig_chunk_size*val_size)

    elif val_chunk_size == -1:
        val_chunk_size = test_chunk_size


    un_chunked_mix = dataset.load_data(path, test_size=test_size)
    train_data, test_data, val_data = un_chunked_mix


    # TRAIN DATA vvvvvv
    texts_train = train_data[:, 0]
    labels_train = train_data[:, 1]

    unique_classes = np.unique(labels_train)
    n_classes = len(unique_classes)


    indicies_class_train = []    
    for i in unique_classes:
        indicies_class_train.append(np.where(labels_train==i)[0])

    # min_count = chunk_size//n_classes #min(num_elements_0, num_elements_1)
    # print('min_count:',min_count)
    # min_count = [3310, 1624, 3610]
    _bal_count = chunk_size//n_classes
    _min_count = Counter(labels_train)
    
    if _bal_count < _min_count[min(_min_count)]:
        #print('yo')
        for i in unique_classes:
            _min_count[i] = _bal_count
    min_count = _min_count
    
    if shuffle:
        rng = np.random.default_rng(seed)
        for indicies in indicies_class_train:
            rng.shuffle(indicies)

    split_ind_train = []
    for index, elem in enumerate(unique_classes):
        split_ind_train.append(np.array_split(indicies_class_train[index][:min_count[elem]*n_chunks_per_mix], n_chunks_per_mix))



    train_x, train_y, train_y_orig = chunk_data(n_chunks_per_mix, n_classes, split_ind_train, texts_train, labels_train, dataset, seed)
    


    # TEST DATA  vvvvvv
    if dataset.exists_test_set:
        texts_test = test_data[:, 0]
        labels_test = test_data[:, 1]


        indicies_class_test = []
        for i in unique_classes:
            indicies_class_test.append(np.where(labels_test==i)[0])


        _bal_count = chunk_size//n_classes
        _min_count = Counter(labels_test)
        if _bal_count < _min_count[min(_min_count)]:
            for i in unique_classes:
                _min_count[i] = _bal_count
        min_count = _min_count
        #print(min_count)
        

        if shuffle:
            rng = np.random.default_rng(seed)
            for indicies in indicies_class_test:
                rng.shuffle(indicies)

        split_ind_test = []
        for index,  elem in enumerate(unique_classes):
            split_ind_test.append(np.array_split(indicies_class_test[index][:min_count[elem]*n_chunks_per_mix], n_chunks_per_mix))

        test_x, test_y, test_y_orig = chunk_data(n_chunks_per_mix, n_classes, split_ind_test, texts_test, labels_test, dataset, seed)
        
    else:
        train_x_split, train_y_split, test_x_split, test_y_split = [], [], [], []
        for i in range(n_chunks_per_mix):
            train_x_temp, train_y_temp, test_x_temp, test_y_temp = train_test_split(train_x[i], train_y[i], test_size=test_size, random_state=seed)
            train_x_split.append(train_x_temp)
            train_y_split.append(train_y_temp)
            test_x_split.append(test_x_temp)
            test_y_split.append(test_y_temp)
        
        train_x = train_x_split
        train_y = train_y_split
        test_x = test_x_split
        test_y = test_y_split
        test_y_orig = None

        
    if dataset.exists_validation_set and validation:
        texts_val = val_data[:, 0]
        labels_val = val_data[:, 1]

        indicies_class_val = []
        for i in unique_classes:
            indicies_class_val.append(np.where(labels_val==i)[0])

        _bal_count = chunk_size//n_classes
        _min_count = Counter(labels_val)
        if _bal_count < _min_count[min(_min_count)]:
            for i in unique_classes:
                _min_count[i] = _bal_count
        min_count = _min_count


        if shuffle:
            rng = np.random.default_rng(seed)
            for indicies in indicies_class_val:
                rng.shuffle(indicies)

        split_ind_val = []
        for index,  elem in enumerate(unique_classes):
            split_ind_val.append(np.array_split(indicies_class_val[i][:min_count[elem]*n_chunks_per_mix], n_chunks_per_mix))


        val_x, val_y, val_y_orig = chunk_data(n_chunks_per_mix, n_classes, split_ind_val, texts_val, labels_val, dataset, seed)
        


    elif validation:
        test_x_split, test_y_split, val_x_split, val_y_split = [], [], [], []
        for i in range(n_chunks_per_mix):
            test_x_temp, val_x_temp, test_y_temp, val_y_temp = train_test_split(test_x[i], test_y[i], test_size=val_size, random_state=seed)
            test_x_split.append(test_x_temp)
            test_y_split.append(test_y_temp)
            val_x_split.append(val_x_temp)
            val_y_split.append(val_y_temp)
        
        val_x = val_x_split
        val_y = val_y_split
        val_y_orig = None

    
    else:
        val_x, val_y, val_y_orig = None, None, None
        

    return train_x, train_y, test_x, test_y, val_x, val_y, train_y_orig, test_y_orig, val_y_orig, n_classes


















# train_x = []
    # train_y = []
    # train_y_orig = []
    # for i in range(n_chunks_per_mix):
    #     split_ind = np.array([], dtype=int)
    #     split_ind = np.concatenate((split_ind, split_ind_train[0][i]))
    #     for index in range(1, n_classes):
    #         temp = split_ind_train[index][i]
    #         split_ind = np.concatenate((split_ind, temp))

    #     split_text = texts_train[split_ind]
    #     split_label = labels_train[split_ind]
        
    #     if dataset.n_classes > 3:
    #         split_label_orig = dataset.train_all_labels[split_ind]
    #         split_text, split_label, split_label_orig = shuffle_unison([split_text, split_label, split_label_orig], seed)
    #         train_x.append(split_text)
    #         train_y.append(split_label)
    #         train_y_orig.append(split_label_orig)

    #     else:
    #         split_text, split_label = shuffle_unison([split_text, split_label], seed)

    #         train_x.append(split_text)
    #         train_y.append(split_label)

# test_x = []
        # test_y = []
        # for i in range(n_chunks_per_mix):
        #     split_ind = np.array([], dtype=int)
        #     split_ind = np.concatenate((split_ind, split_ind_test[0][i]))
        #     for index in range(1, n_classes):
        #         temp = split_ind_test[index][i]
        #         split_ind = np.concatenate((split_ind, temp))
        #     split_text = texts_test[split_ind]
        #     split_label = labels_test[split_ind]

        #     split_text, split_label = shuffle_unison(split_text, split_label, seed)

        #     test_x.append(split_text)
        #     test_y.append(split_label)


# val_x = []
        # val_y = []
        # for i in range(n_chunks_per_mix):
        #     split_ind = np.array([], dtype=int)
        #     split_ind = np.concatenate((split_ind, split_ind_val[0][i]))
        #     for index in range(1, n_classes):
        #         temp = split_ind_val[index][i]
        #         split_ind = np.concatenate((split_ind, temp))
        #     split_text = texts_val[split_ind]
        #     split_label = labels_val[split_ind]

        #     split_text, split_label = shuffle_unison(split_text, split_label, seed)

        #     val_x.append(split_text)
        #     val_y.append(split_label)


"""
def chunk_data(dataset, n_chunks_per_mix : int, chunk_size : int, path : str, test : bool = False, shuffle : bool = True, seed : int = 42):


    un_chunked_mix = dataset.load_data(path)

    train_data, test_data = un_chunked_mix

    texts_train = train_data[:, 0]
    labels_train = train_data[:, 1]

    indices_class_0_train = np.where(labels_train==0)[0]
    indices_class_1_train = np.where(labels_train==1)[0]

    min_count = chunk_size//2 #min(num_elements_0, num_elements_1)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices_class_0_train)
        rng.shuffle(indices_class_1_train)

    split_ind_0_train = np.array_split(indices_class_0_train[:min_count*n_chunks_per_mix], n_chunks_per_mix)
    split_ind_1_train = np.array_split(indices_class_1_train[:min_count*n_chunks_per_mix], n_chunks_per_mix)


    train_x = []
    train_y = []
    for i in range(n_chunks_per_mix):
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

        split_ind_0_test = np.array_split(indices_class_0_test[:min_count*n_chunks_per_mix], n_chunks_per_mix)
        split_ind_1_test = np.array_split(indices_class_1_test[:min_count*n_chunks_per_mix], n_chunks_per_mix)
        test_x = []
        test_y = []
        for i in range(n_chunks_per_mix):
            split_ind = np.concatenate((split_ind_0_test[i], split_ind_1_test[i]))
            split_text = texts_test[split_ind]
            split_label = labels_test[split_ind]

            test_x.append(split_text)
            test_y.append(split_label)

        return train_x, train_y, test_x, test_y
    return train_x, train_y, None, None
"""