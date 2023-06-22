import src.generate_trainingdata as generate_trainingdata


def test_allign_tokens_labels_weights():

    """
    Test to check if alligning tokens, labels and weights is working as expected
    """

    tokens = ["i", "was", "not", "happy", "with", "the", "movie", "it", "was", "bad"]

    # make vocabulary with unigrams, bigrams and trigrams
    vocabulary = {
        "i" : 0.1,
        "not happy" : 0.2,
        "happy with" : 0.4,
        "with the movie" : 0.5,
        "the movie" : 0.6,
        "it" : 0.7,
        "was bad" : 0.8,
    }

    sentiment = 1
    threshold = 0.3

    alligned_tokens, alligned_weights, alligned_labels = generate_trainingdata.allign_tokens_labels_weights(tokens, vocabulary, sentiment, threshold)
    
    expected_tokens = ['i', 'was', 'not happy', 'with the movie', 'it', 'was bad']
    expected_weights = [0.1, 0.0, 0.2, 0.5, 0.7, 0.8]
    expected_labels = [0, 0, 1, 2, 0, 2]

    assert alligned_tokens == expected_tokens, alligned_tokens
    assert alligned_weights == expected_weights, alligned_weights
    assert alligned_labels == expected_labels, alligned_labels

    
if __name__ == '__main__':
    
    #test_allign_tokens_labels_weights()
    pass