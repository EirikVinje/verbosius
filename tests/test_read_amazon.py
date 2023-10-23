import chunking.get_data as gd
import numpy as np

DATA_PATH = "/home/bigtech/aggressive_dedup.json.gz"

def test_read_amazon():
    path = DATA_PATH
    data_size = 4

    amazon = gd.dataset("amazon")
    amazon = amazon(two_cat=True)
    train_data, val_data = amazon.load_data(path=path, data_size=data_size)


    assert len(train_data) == data_size, len(train_data)
    assert train_data.shape[1] == 2, train_data.shape[1]
    assert val_data == None, type(val_data)


def test_convert_classes_5_to_3():
    path = DATA_PATH
    data_size = 100

    amazon = gd.dataset("amazon")
    amazon = amazon(two_cat=True)
    train_data, _ = amazon.load_data(path, data_size=data_size)

    # assert that number of classes is 3, not 5
    assert len(np.unique(train_data[:, 1])) == 3, len(np.unique(train_data[:, 1]))


def test_large_data_size():
    path = DATA_PATH
    data_size = 100000

    amazon = gd.dataset("amazon")
    amazon = amazon(two_cat=True)

    train_data, _ = amazon.load_data(path, data_size=data_size)

    assert len(train_data) == data_size, len(train_data)


if __name__ == "__main__":
    test_read_amazon()
    test_convert_classes_5_to_3()
    # test_large_data_size()
    print(f"All tests passed in {__file__}")


