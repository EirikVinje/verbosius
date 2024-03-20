
import numpy as np

from chunking.chunker import Chunker

def test_orig_labels():

    chunker = Chunker("small", 0, 1)
    
    chunker.load_amazon()

    train = chunker.train_data
    orig_labels = chunker.train_orig_y

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

    Chunker("small", 0, 1)
    Chunker("big", 0, 1)
    Chunker("huge", 0, 1)
    Chunker("test", 0, 1)
    

def test_indexing():

    chunker = Chunker("test", part_n=0, n_chunks=3, chunk_size=6)
    chunker.load_amazon()

    for chunk in chunker.chunk_data(return_chunks=True):
        
        assert np.array_equal(np.array([int(s[-1]) for s in chunk[:, 0]]), chunk[:, 1]), f"{chunk[:, 0]} and {chunk[:, 1]} not equal"
        assert np.array_equal(np.array([int(s[-1]) for s in chunk[:, 0]]), chunk[:, 2]), f"{chunk[:, 0]} and {chunk[:, 1]} not equal"
        assert np.array_equal(chunk[:, 2], chunk[:, 2]), f"{chunk[:, 0]} and {chunk[:, 1]} not equal"


if __name__ == "__main__":

    # test_orig_labels()
    # test_size()
    test_indexing()
    print("<done tests:", __file__, ">")
