import verbosius.trainingdata.generate_trainingdata as generate_trainingdata


def test_allign_tokens_labels_weights():

    """
    Test to check if alligning tokens, labels and weights is working as expected
    """

    tokens = ["i", "was", "not", "happy", "with", "the", "movie", "it", "was", "bad"]

    # make vocabulary with unigrams, bigrams and trigrams
    vocabulary = {
        "i" : 0.1,
        "was not" : -0.2,
        "not happy" : 0.2,
        "happy" : 0.4,
        "with the movie" : 0.5,
        "the movie" : -0.6,
        "it" : 0.7,
        "was bad" : -0.8,
    }

    sentiment = 0
    threshold = 0.0

    alligned_tokens, alligned_weights, alligned_labels = generate_trainingdata.allign_tokens_labels_weights_trigram(tokens, vocabulary, sentiment, threshold)
    
    expected_tokens = ['i', 'was not', 'happy', 'with the movie', 'it', 'was bad']
    expected_weights = [0.1, -0.2, 0.4, 0.5, 0.7, -0.8]
    expected_labels = [1, 2, 1, 1, 1, 2]
    expected_sentiment = 0 

    assert alligned_tokens == expected_tokens, alligned_tokens
    assert alligned_weights == expected_weights, alligned_weights
    assert alligned_labels == expected_labels, alligned_labels
    assert sentiment == expected_sentiment, sentiment

    text = "i was happy with the movie but not with the popcorn"
    tokens = text.split()

    # make vocabulary with unigrams, bigrams and trigrams
    vocabulary = {
        "i" : 0.1,
        "was happy" : -0.2,
        "happy with the" : 0.2,
        "the movie" : 0.2,
        "movie but" : 0.4,
        "not with the" : 0.5,
        "the popcorn" : -0.6,
        
    }

    sentiment = 1
    threshold = 0.0

    alligned_tokens, alligned_weights, alligned_labels = generate_trainingdata.allign_tokens_labels_weights_trigram(tokens, vocabulary, sentiment, threshold)

    expected_tokens = ['i', 'was happy', 'with the movie', 'but', 'not with the', 'popcorn']

    assert alligned_tokens == expected_tokens, alligned_tokens

    
if __name__ == '__main__':
    
    #test_allign_tokens_labels_weights()
    pass