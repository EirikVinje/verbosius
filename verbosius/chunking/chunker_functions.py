import os
import datasets as ds
import pickle
import json
from time import perf_counter

import pandas as pd
import numpy as np
import chunking.get_data as gd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from collections import Counter



    

def shuffle_unison(a: list, seed : int = 42):
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(a[0]))
    if len(a) == 3:
        return a[0][p], a[1][p], a[2][p]
    return a[0][p], a[1][p]


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


def supersample_chunk(split_ind, n_classes, n_chunks, bal_count, rng):
    for c in range(n_classes):
        for i in range(n_chunks):
            if len(split_ind[c][i]) < bal_count:
                temp = list(range(n_chunks))
                temp.remove(i)
                sample_index = rng.choice(temp)
                split_ind[c][i] = np.concatenate((split_ind[c][i], rng.choice(split_ind[c][sample_index], bal_count-len(split_ind[c][i]))))
    return split_ind


def chunk_data_multiclass(dataset, 
                          n_chunks_per_mix : int, 
                          chunk_size : int, 
                          path : str, 
                          validation : bool = True,
                          test_chunk_size: int = -1,
                          test_size: float = 0.2, 
                          val_size: float = 0.5, 
                          shuffle : bool = True, 
                          seed : int = 42,
                          val_chunk_size : int = -1):  
    
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
    train_data, val_data = un_chunked_mix


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
        train_x_split, train_y_split, val_x_split, val_y_split = [], [], [], []
        for i in range(n_chunks_per_mix):
            train_x_temp, val_x_temp, train_y_temp, val_y_temp = train_test_split(train_x[i], train_y[i], test_size=val_size, random_state=seed)
            train_x_split.append(train_x_temp)
            train_y_split.append(train_y_temp)
            val_x_split.append(val_x_temp)
            val_y_split.append(val_y_temp)
        
        train_x = train_x_split
        train_y = train_y_split
        val_x = val_x_split
        val_y = val_y_split
        val_y_orig = None

    
    else:
        val_x, val_y, val_y_orig = None, None, None
        

    return train_x, train_y, val_x, val_y, train_y_orig, val_y_orig, n_classes


def chunk_data_multiclass_supersample(dataset, 
                                        n_chunks_per_mix : int, 
                                        chunk_size : int, 
                                        path : str, 
                                        validation : bool = True,
                                        test_chunk_size: int = -1,
                                        test_size: float = 0.2, 
                                        val_size: float = 0.5, 
                                        shuffle : bool = True, 
                                        seed : int = 42,
                                        val_chunk_size : int = -1):  
    
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
    rng = np.random.default_rng(seed)

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
    train_data, val_data = un_chunked_mix


    # TRAIN DATA vvvvvv
    texts_train = train_data[:, 0]
    labels_train = train_data[:, 1]

    unique_classes = np.unique(labels_train)
    n_classes = len(unique_classes)


    indicies_class_train = []    
    for i in unique_classes:
        indicies_class_train.append(np.where(labels_train==i)[0])

 
    _bal_count = chunk_size//n_classes
    _min_count = Counter(labels_train)

    
    if _bal_count < _min_count[min(_min_count)]:
        for i in unique_classes:
            _min_count[i] = _bal_count
    min_count = _min_count
    

    
    if shuffle:
        
        for indicies in indicies_class_train:
            rng.shuffle(indicies)

    split_ind_train = []
    for index, elem in enumerate(unique_classes):
        split_ind_train.append(np.array_split(indicies_class_train[index][:min_count[elem]*n_chunks_per_mix], n_chunks_per_mix))


    # Do supersample, if needed
    split_ind_train = supersample_chunk(split_ind_train, n_classes, n_chunks_per_mix, _bal_count, rng)

    train_x, train_y, train_y_orig = chunk_data(n_chunks_per_mix, n_classes, split_ind_train, texts_train, labels_train, dataset, seed)
    
        

        
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

        split_ind_val = supersample_chunk(split_ind_val, n_classes, n_chunks_per_mix, _bal_count, rng)
        

        val_x, val_y, val_y_orig = chunk_data(n_chunks_per_mix, n_classes, split_ind_val, texts_val, labels_val, dataset, seed)
        


    elif validation:
        train_x_split, train_y_split, val_x_split, val_y_split = [], [], [], []
        for i in range(n_chunks_per_mix):
            train_x_temp, val_x_temp, train_y_temp, val_y_temp = train_test_split(train_x[i], train_y[i], test_size=val_size, random_state=seed)
            train_x_split.append(train_x_temp)
            train_y_split.append(train_y_temp)
            val_x_split.append(val_x_temp)
            val_y_split.append(val_y_temp)
        
        train_x = train_x_split
        train_y = train_y_split
        val_x = val_x_split
        val_y = val_y_split
        val_y_orig = None

    
    else:
        val_x, val_y, val_y_orig = None, None, None
        

    return train_x, train_y, val_x, val_y, train_y_orig, val_y_orig, n_classes




def write_chunks(output, data, test : bool = False):
    
    if test:

        output = os.path.join(output, "test")

        if not os.path.exists(output):
            os.mkdir(output)
        
        dir = os.listdir(output)

        n = len(dir)

        with open(f"{output}/test_chunk_{n}.pkl", "wb") as f:
            pickle.dump(data, f)

    else:

        output = os.path.join(output, "train_val")

        if not os.path.exists(output):
            os.mkdir(output)
        
        dir = os.listdir(output)

        n = len(dir)

        with open(f"{output}/train_val_chunk_{n}.pkl", "wb") as f:
            pickle.dump(data, f)


def write_meta_chunks(output, train_length, validation_length, test_length, dataset, n_classes, seed, shuffle, chunk_amount):

    meta = {"train_length": train_length,
            "validation_length": validation_length,
            "test_length": test_length,
            "n_classes": n_classes,
            "dataset": dataset,
            "seed": seed,
            "shuffle": shuffle,
            "chunk_amount": chunk_amount}
    
    with open(f"{output}/meta.json", "w") as f:
         json.dump(meta, f)










# TEST DATA  vvvvvv
    # if dataset.exists_test_set:
    #     texts_test = test_data[:, 0]
    #     labels_test = test_data[:, 1]


    #     indicies_class_test = []
    #     for i in unique_classes:
    #         indicies_class_test.append(np.where(labels_test==i)[0])


    #     _bal_count = chunk_size//n_classes
    #     _min_count = Counter(labels_test)
    #     if _bal_count < _min_count[min(_min_count)]:
    #         for i in unique_classes:
    #             _min_count[i] = _bal_count
    #     min_count = _min_count

    #     if shuffle:
    #         rng = np.random.default_rng(seed)
    #         for indicies in indicies_class_test:
    #             rng.shuffle(indicies)

    #     split_ind_test = []
    #     for index,  elem in enumerate(unique_classes):
    #         split_ind_test.append(np.array_split(indicies_class_test[index][:min_count[elem]*n_chunks_per_mix], n_chunks_per_mix))

    #     test_x, test_y, test_y_orig = chunk_data(n_chunks_per_mix, n_classes, split_ind_test, texts_test, labels_test, dataset, seed)
        
    # else:
    #     train_x_split, train_y_split, test_x_split, test_y_split = [], [], [], []
    #     for i in range(n_chunks_per_mix):
    #         train_x_temp, train_y_temp, test_x_temp, test_y_temp = train_test_split(train_x[i], train_y[i], test_size=test_size, random_state=seed)
    #         train_x_split.append(train_x_temp)
    #         train_y_split.append(train_y_temp)
    #         test_x_split.append(test_x_temp)
    #         test_y_split.append(test_y_temp)
        
    #     train_x = train_x_split
    #     train_y = train_y_split
    #     test_x = test_x_split
    #     test_y = test_y_split
    #     test_y_orig = None


# for c in range(n_classes):
    #     for i in range(n_chunks_per_mix):
    #         if len(split_ind_train[c][i]) < _bal_count:
    #             print('yo')
    #             split_ind_train[c][i] = np.concatenate((split_ind_train[c][i], rng.choice(split_ind_train[c][i], _bal_count-len(split_ind_train[c][i]))))