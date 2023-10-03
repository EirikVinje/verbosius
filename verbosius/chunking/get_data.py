import datasets as ds
from time import perf_counter

import numpy as np

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

            
class MNIST:
    """
    Exists for testing purposes, not intended for use in the pipeline, will crash if used because not text data.
    """
    def __init__(self, two_cat : bool):
        self.two_cat = two_cat
        self.exists_test_set = False
        self.exists_validation_set = False
        self.n_classes = 10
        self.test_data = None

    def load_test(self):
        return self.test_data

    def load_data(self, path: str, test: bool = False, test_size: float = 0.2):
        t0 = perf_counter()
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, )

        if test:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            x_train = np.where(x_train.reshape((x_train.shape[0], 28 * 28)) > 75, 1, 0)
            x_test = np.where(x_test.reshape((x_test.shape[0], 28 * 28)) > 75, 1, 0)
            x_train = x_train.astype(np.uint8)
            x_test = x_test.astype(np.uint8)
            y_train = y_train.astype(np.int32)
            y_test = y_test.astype(np.int32)

            # train_data = np.column_stack((x_train, y_train))
            train_data = []
            for i in range(len(x_train)):
                train_data.append([x_train[i].tolist(), y_train[i]])
            train_data = np.asarray(train_data, dtype=object)

            test_data = []
            for i in range(len(x_test)):
                test_data.append([x_test[i].tolist(), y_test[i]])
            test_data = np.asarray(test_data, dtype=object)

            t1 = perf_counter()
            print(f"Time to load MNIST: {t1 - t0:.2f} seconds")
            return train_data, test_data

        return np.concatenate((X, y), axis=1), None, None


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


class SST5:

    def __init__(self, two_cat : bool):
        
        self.two_cat = two_cat
        self.exists_test_set = True
        self.exists_validation_set = True
        self.n_classes = 5
        self.test_data = None

    def load_test(self):
        return self.test_data

    def load_data(self, path: str, test: bool = False, test_size: float = 0.2):
        dataset = ds.load_dataset("SetFit/sst5")
        train_data = np.array(dataset["train"])
        test_data = np.array(dataset["test"])
        val_data = np.array(dataset["validation"])
        
        train_x, train_y, test_x, test_y, val_x, val_y = [], [], [], [], [], []
        train_all_labels, test_all_labels, val_all_labels = [], [], []
        combine_labels = {
            0: 0,
            1: 0,
            2: 1,
            3: 2,
            4: 2
        }
        for i in range(len(train_data)):
            train_x.append(train_data[i]["text"])
            train_y.append(combine_labels[int(train_data[i]["label"])])
            train_all_labels.append(int(train_data[i]["label"]))
        for i in range(len(test_data)):
            test_x.append(test_data[i]["text"])
            test_y.append(combine_labels[int(test_data[i]["label"])])
            test_all_labels.append(int(test_data[i]["label"]))
        for i in range(len(val_data)):
            val_x.append(val_data[i]["text"])
            val_y.append(combine_labels[int(val_data[i]["label"])])
            val_all_labels.append(int(val_data[i]["label"]))
        

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        train_all_labels = np.asarray(train_all_labels)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)
        test_all_labels = np.asarray(test_all_labels)
        val_x = np.asarray(val_x)
        val_y = np.asarray(val_y)
        val_all_labels = np.asarray(val_all_labels)

        train_x = train_x.astype(object)
        test_x = test_x.astype(object)
        val_x = val_x.astype(object)
        train_y = train_y.astype(np.uint8)
        test_y = test_y.astype(np.uint8)
        val_y = val_y.astype(np.uint8)

        self.train_all_labels = train_all_labels.astype(np.uint8)
        self.test_all_labels = test_all_labels.astype(np.uint8)
        self.val_all_labels = val_all_labels.astype(np.uint8)

        train_data = np.column_stack((train_x, train_y))
        test_data = np.column_stack((test_x, test_y, test_all_labels))
        val_data = np.column_stack((val_x, val_y))
        
        self.test_data = test_data

        return train_data, val_data


class Amazon:

    def __init__(self, two_cat : bool) -> None:
        
        self.two_cat = two_cat
        self.exists_test_set = False
        self.exists_validation_set = False
        self.test_data = None

    def load_test(self):
        return self.test_data
    
    def load_data(self, path: str):

        return None, None
    

    

def dataset(dataset : str):

    if dataset == "imdb":
        return IMDB
    elif dataset == "rottentomatoes":
        return RottenTomatoes
    elif dataset == "amazon":
        return Amazon
    elif dataset == "mnist":
        return MNIST
    elif dataset =="sst5":
        return SST5
    else:
        raise ValueError("No such dataset exists")