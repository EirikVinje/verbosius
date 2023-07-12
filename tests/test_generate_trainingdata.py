import verbosius.trainingdata.generate_trainingdata as generate_trainingdata


def test_grams1():

    text = "i was happy with the movie but not with the popcorn"
    tokens = text.split()

    # make vocabulary with unigrams, bigrams and trigrams
    vocabulary = {
        "i" : 1,
        "was happy" : 2,
        "happy with" : 1,
        "happy with the" : 1,
        "the movie but" : 1,
        "but" : 1,
        "not with" : 1,
        "with the popcorn" : 1,
    }

    sentiment = 1
    threshold = 0.0

    grams, weights  = generate_trainingdata.find_grams(tokens, vocabulary)
    
    expected = ["i", "was happy", "happy with the", "the movie but", "not with", "with the popcorn"]
    assert grams == expected, grams

    grams_fit, weights_fit = generate_trainingdata.fit_grams(grams, weights)
    
    expected = ["i", "was", "happy with the", "movie but", "not", "with the popcorn"]
    assert grams_fit == expected, grams_fit


def test_grams2():

    text = "i was happy with the movie but not with the popcorn"
    tokens = text.split()

    # make vocabulary with unigrams, bigrams and trigrams
    vocabulary = {
        "i" : 1,
        "was happy" : 2,
        "happy with" : 1,
        "happy with the" : 1,
        "the movie but" : 1,
        "but" : 1,
        "not with" : 1,
        "with the popcorn" : 1,
    }

    sentiment = 1
    threshold = 0.0


    
if __name__ == '__main__':
    
    pass