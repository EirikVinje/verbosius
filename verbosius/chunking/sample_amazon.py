import gzip
import json
import pickle
import os
import argparse

import numpy as np
from tqdm import tqdm


def raw_amazon_iterator(data_path):
    
    k = 0
    with gzip.open(data_path, mode="rt") as zp:
        for line in zp:
            
            try:
                d = json.loads(line)
            
            except json.decoder.JSONDecodeError:
                print("ok")
                print(f"Skipped line {k}, len: {len(line)}")
                k+=1
                continue
            
            k += 1
            


            yield d


def distribute_rest(counts, rest, rng):
    """Randomly assign the rest such that the sum of counts is equal to data_size

    Args:
        counts (dict): dict of counts
        rest (int): rest sum of counts
        rng (generator): rng generator object

    Returns:
        dict: counts dict with rest added
    """
    keys = list(counts.keys())
    rest_keys = rng.choice(keys, size=rest, replace=True)

    for key in rest_keys:
        counts[key] += 1
    
    return counts


def split_train_orig_test(temp_data, counts, test_counts, rng):

    class_lookup = {
        1: 1,
        2: 1,
        3: 0,
        4: 2,
        5: 2
    }

    train_data = []
    orig_labels = []
    test_data = []


    for index, text, label in temp_data:
        if counts[label] > 0:
            train_data.append([text, int(class_lookup[label])])
            orig_labels.append([index, label-1])
            counts[label] -= 1


    train_data = np.array(train_data, dtype=object)
    orig_labels = np.array(orig_labels, dtype=object)
    
    remove_ind = set(orig_labels[:, 0])    
    temp_data = [d for d in temp_data if d[0] not in remove_ind]
    rng.shuffle(temp_data)

    for index, text, label in temp_data:
        if test_counts[label] > 0:
            test_data.append([text, label-1])
            test_counts[label] -= 1


    test_data = np.array(test_data, dtype=object)

    return train_data, orig_labels, test_data


def sample_amazon(path, rng, data_size: int, test_size: int, load_size: int):
    
    temp_data = []
    
    for index, d in enumerate(tqdm(raw_amazon_iterator(path))):
        
        try:

            if len(d["reviewText"].split(" ")) > 400:
                 continue    
                                
            temp_data.append([index, str(d["reviewText"]), int(d["overall"])])

            if len(temp_data) == load_size:
                break
        
        except:
            continue


    max_count_middle = data_size // 3
    max_count_rest = max_count_middle // 2

    max_count_test = test_size // 5

    counts = {
        1: max_count_rest,
        2: max_count_rest,
        3: max_count_middle,
        4: max_count_rest,
        5: max_count_rest
    }

    test_counts = {
        1: max_count_test,
        2: max_count_test,
        3: max_count_test,
        4: max_count_test,
        5: max_count_test
    }

    rest = data_size - sum(counts.values())
    test_rest = test_size - sum(test_counts.values())
    if rest > 0:
        counts = distribute_rest(counts, rest, rng)
    if test_rest > 0:
        test_counts = distribute_rest(test_counts, test_rest, rng)

    rng.shuffle(temp_data)

    train_data, orig_labels, test_data = split_train_orig_test(temp_data, counts, test_counts, rng)

    return train_data, orig_labels, test_data


def save_to_pickle(train_data, train_orig_labels, test_data, store_dir):
    
    train_data_path = os.path.join(store_dir, "train_data.pkl")
    train_orig_labels_path = os.path.join(store_dir, "train_orig_labels.pkl")
    test_data_path = os.path.join(store_dir, "test_data.pkl")

    with open(train_data_path, "wb") as f:
        pickle.dump(train_data, f)
    
    with open(train_orig_labels_path, "wb") as f:
        pickle.dump(train_orig_labels, f)
    
    with open(test_data_path, "wb") as f:
        pickle.dump(test_data, f)


if __name__ == "__main__":
    
    rng = np.random.default_rng(42)
    user = os.environ["USER"]
    store_path = f"/home/{user}/data/verbosius/amazon/pre_chunking/"
    
    name = "small"
    
    store_dir = os.path.join(store_path, name)

    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    else:
        assert False, "Store dir already exists"

    datapath = "/home/bigtech/aggressive_dedup.json.gz"

    train_data, train_orig_labels, test_data = sample_amazon(datapath, 
                                                             rng=rng, 
                                                             data_size=16_000, 
                                                             test_size=1600, 
                                                             load_size=500_000)
                                                             
    save_to_pickle(train_data, train_orig_labels, test_data, store_dir)

