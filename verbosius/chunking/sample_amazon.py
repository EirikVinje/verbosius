import gzip
import json
import pickle

import numpy as np

from tqdm import tqdm



def raw_amazon_iterator(data_path):
    k = 0
    with gzip.open(data_path, mode="rt") as zp:
        for line in zp:
            try:
                d = json.loads(line)
            except json.decoder.JSONDecodeError:
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
    
    # remove the indices that are in the train data
    remove_ind = set(orig_labels[:, 0])    
    temp_data = [d for d in temp_data if d[0] not in remove_ind]
    rng.shuffle(temp_data)

    for index, text, label in temp_data:
        if test_counts[label] > 0:
            test_data.append([text, label-1])
            test_counts[label] -= 1


    test_data = np.array(test_data, dtype=object)

    return train_data, orig_labels, test_data


def sample_amazon(path, rng, data_size: int  = 1000, test_size: int = 200, load_size: int = 90000000, max_text_len: int = 300):
    
    temp_data = []
    
    for index, d in enumerate(tqdm(raw_amazon_iterator(path))):
        try: 
            if len(d["reviewText"].split(" ")) > max_text_len:
                continue
        except:
            print(f"KeyError + {index}")
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

    train_data, orig_labels, test_data = split_train_orig_test(temp_data, counts, test_counts, rng)

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
    
    store_dir = "/home/bigtech/data/verbosius/amazon/pre_chunking/big/"
    
    path = "/home/bigtech/aggressive_dedup.json.gz"

    rng = np.random.default_rng(42)
    
    train_data, train_orig_labels, test_data = sample_amazon(path, rng=rng, data_size=2000000, test_size=400000, load_size=20000000)
    save_to_pickle(train_data, train_orig_labels, test_data, store_dir)

