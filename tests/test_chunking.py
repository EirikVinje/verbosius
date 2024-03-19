
import numpy as np

from chunking.chunker_functions import chunk_data_multiclass_supersample
from chunking.chunker_functions import chunk_data_multiclass
from chunking.chunker_functions import datachunker_amazon
from chunking.get_data import dataset


def test_orig_labels():

    amazon = dataset("amazon")(two_cat=True, size="small")
    
    train = amazon.load_data()

    labels, counts = np.unique([s[1] for s in train], return_counts=True)

    orig_labels = amazon.load_orig_labels()

    class_lookup = {
        0: 1,
        1: 1,
        2: 0,
        3: 2,
        4: 2
    }

    for origlabel, label in zip(orig_labels[:, 1], train[:, 1]):
        assert class_lookup[origlabel] == label, f"{origlabel} and {label} not corresponding"


def test_size():

    amazon = dataset("amazon")(two_cat=True, size="huge")
    
    data = amazon.load_data()

    orig_labels = amazon.load_orig_labels()

    chunk_size = 8000
    chunk_amount = 1250
    seed = 42

    chunks = datachunker_amazon(data, chunk_size, chunk_amount, orig_labels, seed)

    assert len(chunks) == 1125
    

def test_indexing():

    data = np.array([["nr 10 text with label 0", 0],                       
                     ["nr 11 text with label 0", 0],
                     ["nr 12 text with label 0", 0],
                     ["nr 13 text with label 0", 0],
                     ["nr 14 text with label 0", 0],
                     ["nr 15 text with label 0", 0],
                     ["nr 16 text with label 1", 1],
                     ["nr 17 text with label 1", 1],
                     ["nr 18 text with label 1", 1],
                     ["nr 19 text with label 1", 1],
                     ["nr 20 text with label 1", 1],
                     ["nr 21 text with label 1", 1],
                     ["nr 22 text with label 2", 2],
                     ["nr 23 text with label 2", 2],
                     ["nr 24 text with label 2", 2],
                     ["nr 25 text with label 2", 2],
                     ["nr 26 text with label 2", 2],
                     ["nr 27 text with label 2", 2]], dtype=object)

    orig_labels = np.array([[9999, 0],
                            [9999, 0],
                            [9999, 0],
                            [9999, 0],
                            [9999, 0],
                            [9999, 0],
                            [9999, 1],
                            [9999, 1],
                            [9999, 1],
                            [9999, 1],
                            [9999, 1],
                            [9999, 1],
                            [9999, 2],
                            [9999, 2],
                            [9999, 2],
                            [9999, 2],
                            [9999, 2],
                            [9999, 2]])


    chunks = datachunker_amazon(data, chunk_size=6, chunk_amount=3, orig_y=orig_labels, seed=42)

    for chunk in chunks:
        assert np.array_equal(np.array([int(s[-1]) for s in chunk[:, 0]]), chunk[:, 1]), f"{chunk[:, 0]} and {chunk[:, 1]} not equal"
        assert np.array_equal(np.array([int(s[-1]) for s in chunk[:, 0]]), chunk[:, 2]), f"{chunk[:, 0]} and {chunk[:, 1]} not equal"
        assert np.array_equal(chunk[:, 2], chunk[:, 2]), f"{chunk[:, 0]} and {chunk[:, 1]} not equal"



if __name__ == "__main__":

    # test_load_dataset()
    # test_chunk_data_multliclass_supersample()
    test_size()
    # test_indexing()