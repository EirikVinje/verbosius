import pickle
import gzip
import os


import utils.config as config
from weighter import Weighter


def test_make_weighted_data():
    pass


def test_rulemaker():
    pass


def test_do_weighting():
    pass


def test_weight_tokens():
    
    lemma_x = ["did", "not", "work", "and", "have", "not", "eat", "suplie"]
    token_x = ["did", "nt", "work", "and", "have", "nt", "eaten", "suplied"]
    token_ids = [0, 0, 1, 2, 3, 3, 4, 5]
    
    vocabulary = {"did" : 0.1,
                  "not" : 0.1,
                  "and" : 0.1,
                  "not" : 0.1,
                  "suplie" : 0.1}

    newtokens_x, weights_x = Weighter(0)._weight_tokens(lemma_x, token_x, vocabulary, token_ids)

    assert newtokens_x == ['didnt', 'work', 'and', 'havent', 'eaten', 'suplied']
    assert weights_x == [0.2, 0.0, 0.1, 0.1, 0.0, 0.1]

def test_connect_tokens():
    
    weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    token_x = ["did", "nt", "work", "and", "have", "nt", "eaten", "suplied"]
    token_ids = [0, 0, 1, 2, 3, 3, 4, 5]
    new_toks, new_weights = Weighter(0)._connect_tokens(token_x, weights, token_ids)

    assert new_toks == ['didnt', 'work', 'and', 'havent', 'eaten', 'suplied']
    assert new_weights == [0.2, 0.1, 0.1, 0.2, 0.1, 0.1]

def test_label_tokens():
    
    y = 0
    weights = [0.3, 0.01, -0.02, -0.3, 0, 0.0]
    labels = Weighter(0)._label_tokens(y, weights)
    assert labels == [0, 0, 0, 0, 0, 0], f"{labels} != {weights} when y={y}"
    
    y = 1
    weights = [0.3, 0.01, -0.2, -0.3, 0, 0.0]
    labels = Weighter(0)._label_tokens(y, weights)
    assert labels == [1, 1, 2, 2, 0, 0], f"{labels} != {weights} when y={y}"
    
    y = 2
    weights = [0.3, 0.01, -0.02, -0.3, 0, 0.0]
    labels = Weighter(0)._label_tokens(y, weights)
    assert labels == [2, 2, 1, 1, 0, 0], f"{labels} != {weights} when y={y}"


def test_main_loop():

    config.root = "/home/bigtech/data/verbosius/testing/root"
    chunk = os.path.join(config.root, "preprocess", "part_1", "chunk_0_.pkl")

    with gzip.open(chunk, "rb") as f:
        data1 = pickle.load(f)    

    config.TM_EPOCHS = 1
    weighter = Weighter(1, force_write=True)
    weighter.run()

    config.root = "/home/bigtech/data/verbosius/testing/root"
    chunk = os.path.join(config.root, "weighter", "part_1", "chunk_0_.pkl")

    with gzip.open(chunk, "rb") as f:
        data2 = pickle.load(f)    

    s_idx = [s["sample_index"] for s in data2]
    
    s1 = None
    s2 = None
    for sample in data1:
        if sample["sample_index"] in s_idx and sample["y"] != 0:
            idx = s_idx.index(sample["sample_index"])
            
            s1 = sample
            s2 = data2[idx]
            

    assert s1["y"] == s2["y"]
    assert s1["orig_y"] == s2["orig_y"]    
    assert s1["x"] == " ".join(s2["token_x"])


if __name__ == "__main__":

    # test_label_tokens()
    # test_weight_tokens()
    # test_connect_tokens()
    test_main_loop()

    print("<done tests:", __file__, ">")
