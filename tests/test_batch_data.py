import numpy as np
import pytest
from verbosius.preprocessing import datasource
from pathlib import Path

from time import perf_counter



def test_batch_data_with_one_batch():
    """
    Simple test to check that loading data with one batch gives the same as simply loading the data.
    """
    path = (((Path(__file__).parent).parent).parent).parent / "data" / "verbosius" / "imdb"


    ds = datasource.dataset('imdb')
    ds = ds(two_cat=True)
    load_train, load_test = ds.load_data(path=path, test=True)

    # doesnt work becuase of tons of newline characters between the inputs here
    texts_train = load_train[:, 0]
    labels_train = load_train[:, 1]
    texts_test = load_test[:, 0]
    labels_test = load_test[:, 1]



    batch_data = datasource.batch_data(dataset = ds, 
                                       n_batches_per_mix = 1, 
                                       batch_size = 25000,
                                       path = path,
                                       shuffle=False, 
                                       test = True)
    
    load_data = [texts_train, labels_train, texts_test, labels_test]
    assert (load_data[0] == batch_data[0][0]).all(), "train_x"
    assert load_data[1] == batch_data[1][0], "train_y"
    assert load_data[2] == batch_data[2][0], "test_x"
    assert load_data[3] == batch_data[3][0], "test_y"

def test_batch_data_with_one_batch_with_shuffle():
    """
    Simple test to check that loading data with one batch gives the same as simply loading the data.
    """
    path = (((Path(__file__).parent).parent).parent).parent / "data" / "verbosius" / "imdb"


    ds = datasource.dataset('imdb')
    ds = ds(two_cat=True)
    load_data = ds.load_data(path=path, test=True, shuffle=True)

    batch_data = datasource.batch_data(dataset = ds, 
                                       n_batches_per_mix = 1, 
                                       batch_size = 25000,
                                       path = path, 
                                       test = True,
                                       shuffle = True)
    

    assert load_data[0] == batch_data[0][0], "train_x"
    assert load_data[1] == batch_data[1][0], "train_y"
    assert load_data[2] == batch_data[2][0], "test_x"
    assert load_data[3] == batch_data[3][0], "test_y"

def test_batch_data_with_two_batches():

    path = (((Path(__file__).parent).parent).parent).parent / "data" / "verbosius" / "imdb"


    ds = datasource.dataset('imdb')
    ds = ds(two_cat=True)
    load_data = ds.load_data(path=path, test=True)

    batch_data = datasource.batch_data(dataset = ds,
                                        n_batches_per_mix = 2,
                                        batch_size = 12500,
                                        start_point = 0,
                                        path = path,
                                        test = True)
    
    assert load_data[0][0:12500:] == batch_data[0][0], "train_x 1"
    assert load_data[0][12500::] == batch_data[0][1], "train_x 2"
    assert load_data[1][0:12500:] == batch_data[1][0], "train_y 1"
    assert load_data[1][12500::] == batch_data[1][1], "train_y 2"
    assert load_data[2][0:12500:] == batch_data[2][0], "test_x 1"
    assert load_data[2][12500::] == batch_data[2][1], "test_x 2"
    assert load_data[3][0:12500:] == batch_data[3][0], "test_y 1"
    assert load_data[3][12500::] == batch_data[3][1], "test_y 2"

def test_batch_data_mix_size_greater_than_data_size():

    path = (((Path(__file__).parent).parent).parent).parent / "data" / "verbosius" / "imdb"


    ds = datasource.dataset('imdb')
    ds = ds(two_cat=True)
    load_data = ds.load_data(path=path, test=True)


    batch_size = 20000
    batch_data = datasource.batch_data(dataset = ds,
                                        n_batches_per_mix = 2,
                                        batch_size = batch_size,
                                        start_point = 0,
                                        path = path,
                                        test = True)
    
    assert load_data[0][0:batch_size:] == batch_data[0][0], "train_x 1"
    assert load_data[0][batch_size::] == batch_data[0][1], "train_x 2"
    assert load_data[1][0:batch_size:] == batch_data[1][0], "train_y 1"
    assert load_data[1][batch_size::] == batch_data[1][1], "train_y 2"
    assert load_data[2][0:batch_size:] == batch_data[2][0], "test_x 1"
    assert load_data[2][batch_size::] == batch_data[2][1], "test_x 2"
    assert load_data[3][0:batch_size:] == batch_data[3][0], "test_y 1"
    assert load_data[3][batch_size::] == batch_data[3][1], "test_y 2"

def test_batch_data_two_rounds():

    path = (((Path(__file__).parent).parent).parent).parent / "data" / "verbosius" / "imdb"


    ds = datasource.dataset('imdb')
    ds = ds(two_cat=True)
    load_data = ds.load_data(path=path, test=True)

    rounds = 2
    batch_size = 25000//(rounds*2)
    curr_mid_point = -batch_size
    curr_end_point = 0
    start_point = 0
    for round in range(rounds):
        batch_data = datasource.batch_data(dataset = ds,
                                            n_batches_per_mix=2,
                                            batch_size=batch_size,
                                            start_point=start_point,
                                            path=path,
                                            test=True)
        curr_start_point = start_point
        curr_mid_point += batch_size*2
        curr_end_point += batch_size*2

        start_point += batch_size*2

        assert load_data[0][curr_start_point:curr_mid_point:] == batch_data[0][0], "train_x 1"
        assert load_data[0][curr_mid_point:curr_end_point:] == batch_data[0][1], "train_x 2"
        assert load_data[1][curr_start_point:curr_mid_point:] == batch_data[1][0], "train_y 1"
        assert load_data[1][curr_mid_point:curr_end_point:] == batch_data[1][1], "train_y 2"
        assert load_data[2][curr_start_point:curr_mid_point:] == batch_data[2][0], "test_x 1"
        assert load_data[2][curr_mid_point:curr_end_point:] == batch_data[2][1], "test_x 2"
        assert load_data[3][curr_start_point:curr_mid_point:] == batch_data[3][0], "test_y 1"
        assert load_data[3][curr_mid_point:curr_end_point:] == batch_data[3][1], "test_y 2"


def test_batch_data_multiple_rounds():
    path = (((Path(__file__).parent).parent).parent).parent / "data" / "verbosius" / "imdb"

    ds = datasource.dataset('imdb')
    ds = ds(two_cat=True)
    load_data = ds.load_data(path=path, test=True)

    rounds = 16
    batch_size = 25000//(rounds*2)
    curr_mid_point = -batch_size
    curr_end_point = 0
    start_point = 0
    for round in range(rounds):
        batch_data = datasource.batch_data(dataset = ds,
                                            n_batches_per_mix=2,
                                            batch_size=batch_size,
                                            start_point=start_point,
                                            path=path,
                                            test=True)
        curr_start_point = start_point
        curr_mid_point += batch_size*2
        curr_end_point += batch_size*2

        start_point += batch_size*2

        assert load_data[0][curr_start_point:curr_mid_point:] == batch_data[0][0], "train_x 1"
        assert load_data[0][curr_mid_point:curr_end_point:] == batch_data[0][1], "train_x 2"
        assert load_data[1][curr_start_point:curr_mid_point:] == batch_data[1][0], "train_y 1"
        assert load_data[1][curr_mid_point:curr_end_point:] == batch_data[1][1], "train_y 2"
        assert load_data[2][curr_start_point:curr_mid_point:] == batch_data[2][0], "test_x 1"
        assert load_data[2][curr_mid_point:curr_end_point:] == batch_data[2][1], "test_x 2"
        assert load_data[3][curr_start_point:curr_mid_point:] == batch_data[3][0], "test_y 1"
        assert load_data[3][curr_mid_point:curr_end_point:] == batch_data[3][1], "test_y 2"


def test_multiclass_batch_test_with_two_classes():
    path = (((Path(__file__).parent).parent).parent).parent / "data" / "verbosius" / "imdb"

    ds = datasource.dataset('imdb')
    ds = ds(two_cat=True)
    batch_size = 25000//2
    
    t0 = perf_counter()
    batch_data_orig = datasource.batch_data(dataset = ds,
                                            n_batches_per_mix=2,
                                            batch_size=batch_size,
                                            path=path,
                                            test=True,
                                            shuffle=False)

    t1 = perf_counter()
    t2 = perf_counter()


    batch_data_multi = datasource.batch_data_TEMP(dataset = ds,
                                            n_batches_per_mix=2,
                                            batch_size=batch_size,
                                            path=path,
                                            test=True,
                                            shuffle=False)
    t3 = perf_counter()
    


    assert (batch_data_orig[0][0] == batch_data_multi[0][0]).all(), "train_x 1"
    assert (batch_data_orig[0][1] == batch_data_multi[0][1]).all(), "train_x 2"
    assert (batch_data_orig[1][0] == batch_data_multi[1][0]).all(), "train_y 1"
    assert (batch_data_orig[1][1] == batch_data_multi[1][1]).all(), "train_y 2"
    assert (batch_data_orig[2][0] == batch_data_multi[2][0]).all(), "test_x 1"
    assert (batch_data_orig[2][1] == batch_data_multi[2][1]).all(), "test_x 2"
    assert (batch_data_orig[3][0] == batch_data_multi[3][0]).all(), "test_y 1"
    assert (batch_data_orig[3][1] == batch_data_multi[3][1]).all(), "test_y 2"


def test_multiclass_batch_test_with_10_classes():
    path = (((Path(__file__).parent).parent).parent).parent / "data" / "verbosius" / "imdb"

    ds = datasource.dataset('mnist')
    ds = ds(two_cat=True)
    batch_size = 25000//2
    
    t0 = perf_counter()
    batch_data_orig = datasource.batch_data(dataset = ds,
                                            n_batches_per_mix=2,
                                            batch_size=batch_size,
                                            path=path,
                                            test=True,
                                            shuffle=False)

    t1 = perf_counter()
    t2 = perf_counter()


    batch_data_multi = datasource.batch_data_TEMP(dataset = ds,
                                            n_batches_per_mix=2,
                                            batch_size=batch_size,
                                            path=path,
                                            test=True,
                                            shuffle=False)
    t3 = perf_counter()
    


    assert (batch_data_orig[0][0] == batch_data_multi[0][0]).all(), "train_x 1"
    assert (batch_data_orig[0][1] == batch_data_multi[0][1]).all(), "train_x 2"
    assert (batch_data_orig[1][0] == batch_data_multi[1][0]).all(), "train_y 1"
    assert (batch_data_orig[1][1] == batch_data_multi[1][1]).all(), "train_y 2"
    assert (batch_data_orig[2][0] == batch_data_multi[2][0]).all(), "test_x 1"
    assert (batch_data_orig[2][1] == batch_data_multi[2][1]).all(), "test_x 2"
    assert (batch_data_orig[3][0] == batch_data_multi[3][0]).all(), "test_y 1"
    assert (batch_data_orig[3][1] == batch_data_multi[3][1]).all(), "test_y 2"

if __name__ == "__main__":
    # test_batch_data_with_one_batch()
    # test_batch_data_with_one_batch_with_shuffle()
    # test_batch_data_with_two_batches()
    # test_batch_data_mix_size_greater_than_data_size()
    # test_batch_data_two_rounds()
    # test_batch_data_multiple_rounds()
    test_multiclass_batch_test_with_two_classes()


    print(f"All tests passed in {__file__}")