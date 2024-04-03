import os
import shutil
import gzip
import pickle

import numpy as np

from verbosius.trainingdata import Trainingdata
import utils.config as config

def test_tokenizing():

    part_n = 1234
    traindata = Trainingdata(part_n=part_n)

    chunk = [{"orig_y" : 0,
        "y" : 1,
        "token_x" : "the cat sat on the mat".split(),
        "labels" : [0, 1, 1, 1, 0, 0]
    }]
        
    tokenized = traindata._tokenize_and_align_labels(chunk)

    assert tokenized[0]["targets"][-1] == 0
    assert tokenized[0]["targets"][0] == 0

    assert tokenized[0]["labels"][-1] == -100
    assert tokenized[0]["labels"][0] == -100

    assert len(tokenized[0]["targets"]) == len(chunk[0]["token_x"]) + 2 
    assert len(tokenized[0]["attention_mask"]) == len(chunk[0]["token_x"]) + 2 
    assert len(tokenized[0]["labels"]) == len(chunk[0]["token_x"]) + 2 
    assert len(tokenized[0]["input_ids"]) == len(chunk[0]["token_x"]) + 2 

    chunk = [{"orig_y" : 0,
        "y" : 1,
        "token_x" : "did nt".split(),
        "labels" : [0, 1]
        }]    

    tokenized = traindata._tokenize_and_align_labels(chunk)

    assert tokenized[0]["targets"][-1] == 0
    assert tokenized[0]["targets"][0] == 0

    assert tokenized[0]["labels"][-1] == -100
    assert tokenized[0]["labels"][0] == -100

    assert len(tokenized[0]["targets"]) == len(chunk[0]["token_x"]) + 3
    assert len(tokenized[0]["attention_mask"]) == len(chunk[0]["token_x"]) + 3
    assert len(tokenized[0]["labels"]) == len(chunk[0]["token_x"]) + 3
    assert len(tokenized[0]["input_ids"]) == len(chunk[0]["token_x"]) + 3


def test_class_balance():

    config.root = "/home/bigtech/data/verbosius/testing/temp"

    part_n = 1234
    traindata = Trainingdata(part_n=part_n, force_write=True)


    if os.path.exists(config.root):
        shutil.rmtree(config.root)
        os.mkdir(config.root)
    
    else:
        os.mkdir(config.root)

    os.mkdir(os.path.join(config.root, "weighter"))
    os.mkdir(os.path.join(config.root, "weighter", f"part_{part_n}"))
    
    for i in range(3):

        y = [i for _ in range(100)]
        orig_y = [i+100 for _ in range(100)]

        data = [{"y" : y[i], "orig_y" : orig_y[i]} for i in range(100)]

        with gzip.open(os.path.join(config.root, "weighter", f"part_{part_n}", f"chunk_{i}"), "wb") as f:
            pickle.dump(data, f)
            
    traindata._get_class_balance()

    assert traindata.class_y_balance == [100, 100, 100]
    assert traindata.balance_train == [[80, 80, 80], [0, 0, 0]]
    assert traindata.balance_eval == [[20, 20, 20], [0, 0, 0]]

    shutil.rmtree("/home/bigtech/data/verbosius/testing/temp")


def test_main_loop():

    config.root = "/home/bigtech/data/verbosius/testing/temp"

    part_n = 1234
    traindata = Trainingdata(part_n=part_n, force_write=True)

    if os.path.exists(config.root):
        shutil.rmtree(config.root)
        os.mkdir(config.root)
    
    else:
        os.mkdir(config.root)

    os.mkdir(os.path.join(config.root, "weighter"))
    os.mkdir(os.path.join(config.root, "trainingdata"))
    os.mkdir(os.path.join(config.root, "weighter", f"part_{part_n}"))

    
    for i in range(3):

        l = 8000

        y = [i for _ in range(l)]
        
        orig_y = y
        
        data = [{"y" : y[i], "orig_y" : orig_y[i], "token_x" : ["test", "test", "test"], "labels" : [0, 0, 0]} for i in range(l)]

        with gzip.open(os.path.join(config.root, "weighter", f"part_{part_n}", f"chunk_{i}"), "wb") as f:
            pickle.dump(data, f)

    traindata.run()

    assert os.path.exists(os.path.join(config.root, "trainingdata", "part_1234"))

    in_traindir = os.listdir(os.path.join(config.root, "trainingdata", "part_1234", "train"))
    in_evaldir = os.listdir(os.path.join(config.root, "trainingdata", "part_1234", "eval"))

    for train in in_traindir:

        with gzip.open(os.path.join(config.root, "trainingdata", "part_1234", "train", train), "rb") as f:
            data = pickle.load(f)

        class_balance = np.unique([data[i]["sentiment"] for i in range(len(data))], return_counts=True)

        assert class_balance[0].size == 3
        assert class_balance[1][0] == class_balance[1][1] == class_balance[1][2]

    for eval in in_evaldir:

        with gzip.open(os.path.join(config.root, "trainingdata", "part_1234", "eval", eval), "rb") as f:
            data = pickle.load(f)

        class_balance = np.unique([data[i]["sentiment"] for i in range(len(data))], return_counts=True)

        assert class_balance[0].size == 3
        assert class_balance[1][0] == class_balance[1][1] == class_balance[1][2]

    shutil.rmtree("/home/bigtech/data/verbosius/testing/temp")






if __name__ == "__main__":

    # test_tokenizing()
    # test_class_balance()
    test_main_loop()


