import chunking.sample_amazon as sa
import numpy as np
import sys

from collections import Counter

DATA_PATH = "/home/bigtech/aggressive_dedup.json.gz"

def test_read_amazon():
    path = DATA_PATH
    data_size = 100
    test_size = 20

    rng = np.random.default_rng(42)

    train_data, orig_labels, test_data = sa.sample_amazon(path, rng=rng, data_size=data_size, test_size=test_size, load_size=1000)

    assert len(train_data) == data_size, len(train_data)
    assert train_data.shape[1] == 2, train_data.shape[1]
    assert len(orig_labels) == data_size, len(orig_labels)
    assert orig_labels.shape[1] == 2, orig_labels.shape[1]
    assert len(test_data) == test_size, len(test_data)
    assert test_data.shape[1] == 2, test_data.shape[1]


def test_convert_classes_5_to_3():
    path = DATA_PATH
    data_size = 100
    test_size = 20

    rng = np.random.default_rng(42)

    train_data, orig_labels, test_data = sa.sample_amazon(path, rng=rng, data_size=data_size, test_size = test_size, load_size=1000)



    # assert that number of classes is 3, not 5
    assert len(np.unique(train_data[:, 1])) == 3, len(np.unique(train_data[:, 1]))
    assert len(np.unique(orig_labels[:, 1])) == 5, len(np.unique(orig_labels))
    assert len(np.unique(test_data[:, 1])) == 5, len(np.unique(test_data[:, 1]))

def test_large_data_size():
    path = DATA_PATH
    data_size = 10000
    test_size = 2000
    load_size = 1000000

    rng = np.random.default_rng(42)

    train_data, orig_labels, test_data = sa.sample_amazon(path, rng=rng, data_size=data_size, test_size=test_size, load_size=load_size)


    assert len(train_data) == data_size, len(train_data)
    assert train_data.shape[1] == 2, train_data.shape[1]
    assert len(orig_labels) == data_size, len(orig_labels)
    assert len(np.unique(orig_labels[:, 1])) == 5, len(np.unique(orig_labels))
    assert orig_labels.shape[1] == 2, orig_labels.shape[1]
    assert len(test_data) == test_size, len(test_data)
    assert test_data.shape[1] == 2, test_data.shape[1]
    



if __name__ == "__main__":
    test_read_amazon()
    test_convert_classes_5_to_3()
    test_large_data_size()
    print(f"All tests passed in {__file__}")


