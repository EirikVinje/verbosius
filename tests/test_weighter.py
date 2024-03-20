from trainingdata.weighter import Weighter


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



if __name__ == "__main__":

    # test_label_tokens()
    # test_weight_tokens()
    test_connect_tokens()

    print("<done tests:", __file__, ">")
