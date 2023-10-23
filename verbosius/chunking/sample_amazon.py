import gzip
import json
import pickle

import numpy as np




def raw_amazon_iterator(data_path):
    with gzip.open(data_path, mode="rt") as zp:
        for line in zp:
            d = json.loads(line)
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

def split_train_orig_test(temp_data, counts, test_counts):

    class_lookup = {
        1: 0,
        2: 0,
        3: 1,
        4: 2,
        5: 2
    }

    train_data = []
    orig_labels = []
    test_data = []


    for index, text, label in temp_data:
        if counts[label] > 0:
            train_data.append([text, class_lookup[label]])
            orig_labels.append([index, label])
            counts[label] -= 1

    train_data = np.array(train_data)
    orig_labels = np.array(orig_labels)

    for index, text, label in temp_data:
        if test_counts[label] > 0 and index not in orig_labels[:, 0]:
            test_data.append([text, label])
            test_counts[label] -= 1


    
    test_data = np.array(test_data)

    return train_data, orig_labels, test_data


def sample_amazon(path, rng, data_size: int  = 1000, test_size: int = 200, load_size: int = 90000000, max_text_len: int = 400):
    
    temp_data = []
    

    # define counts for each class, to ensure class balance
    counts = np.zeros(5, dtype=np.uint32)

    max_count = data_size // 3
    max_count_combined_classes = max_count // 2

    for index, d in enumerate(raw_amazon_iterator(path)):
        if len(d["reviewText"]) > max_text_len:
            continue

        # orig_labels.append([index, int(d["overall"])])
        temp_data.append([index, str(d["reviewText"]), int(d["overall"])])

        if len(temp_data) == load_size:
            break

    
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

    # shuffle the data
    rng.shuffle(temp_data)

    train_data, orig_labels, test_data = split_train_orig_test(temp_data, counts, test_counts)

    return train_data, orig_labels, test_data


def save_to_pickle(train_data, train_orig_labels, test_data, store_dir):
    
    with open(f"{store_dir}train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    
    with open(f"{store_dir}train_orig_labels.pkl", "wb") as f:
        pickle.dump(train_orig_labels, f)
    
    with open(f"{store_dir}test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

    print("pickled data")
    

if __name__ == "__main__":
    store_dir = "/home/tobxtra/data/verbosius/amazon/pre_chunking/small/"
    path = "/home/bigtech/aggressive_dedup.json.gz"

    rng = np.random.default_rng(42)
    
    train_data, train_orig_labels, test_data = sample_amazon(path, rng=rng, data_size=8000, test_size=2000, load_size=100000)
    save_to_pickle(train_data, train_orig_labels, test_data, store_dir)

