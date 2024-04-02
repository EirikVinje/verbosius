import gzip
import json
import pickle
import os

import datasets as ds
import numpy as np

from time import perf_counter
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class IMDB:

    def __init__(self, two_cat : bool):
        
        self.two_cat = two_cat
        self.exists_test_set = True
        self.exists_validation_set = False
        self.n_classes = 2
        self.test_data = None

    def load_test(self):
        dataset = ds.load_dataset("imdb")
        test_data = dataset["test"]

        test_x, test_y = [], []

        for i in range(len(test_data)):
            test_x.append(test_data[i]["text"])
            test_y.append(test_data[i]["label"])
        

        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)

        test_x = test_x.astype(object)
        test_y = test_y.astype(np.uint8)


        test_data = np.column_stack((test_x, test_y))

        return test_data

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

        self.test_data = test_data

        return train_data, None

 
class RottenTomatoes:

    def __init__(self, two_cat : bool):
        
        self.two_cat = two_cat
        self.exists_test_set = True
        self.exists_validation_set = True
        self.n_classes = 2
        self.test_data = None

    def load_test(self):
        return self.test_data
    
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

        self.test_data = test_data

        return train_data, val_data
    

class Amazon:

    def __init__(self, two_cat : bool, size : str) -> None:
        self.two_cat = two_cat
        self.exists_test_set = False
        self.exists_validation_set = False
        self.test_data = None
        self.n_classes = 5
        self.size = size

    def load_test(self):
        
        user = os.environ.get('USER')

        store_dir = f"/home/{user}/data/verbosius/amazon/pre_chunking/{self.size}/"

        with open(f"{store_dir}test_data.pkl", "rb") as f:
            self.test_data = pickle.load(f)

        return self.test_data
    
    def load_orig_labels(self):

        user = os.environ.get('USER')

        store_dir = f"/home/{user}/data/verbosius/amazon/pre_chunking/{self.size}/"

        with open(f"{store_dir}train_orig_labels.pkl", "rb") as f:
            train_orig_labels = pickle.load(f)

        return train_orig_labels

    def load_data(self):
        
        user = os.environ.get('USER')

        store_dir = f"/home/{user}/data/verbosius/amazon/pre_chunking/{self.size}/"

        with open(f"{store_dir}train_data.pkl", "rb") as f:
            train_data = pickle.load(f)

        return train_data
    

def get_dataset(dataset : str):
    """Function to pick which dataset to use in the pipeline. Returns class object which needs to be instantiated.

    ex: 
    amazon = dataset("amazon")
    amazon = amazon(two_cat=True)
    train_data, val_data = amazon.load_data(path=path, data_size=data_size)

    Args:
        dataset (str): string of dataset name

    Raises:
        ValueError: if dataset is not one of the following: imdb, rottentomatoes, amazon, mnist

    Returns:
        Class object: Returns class for chosen dataset
    """
    if dataset == "imdb":
        return IMDB
    elif dataset == "rottentomatoes":
        return RottenTomatoes
    elif dataset == "amazon":
        return Amazon
    else:
        raise ValueError("No such dataset exists")