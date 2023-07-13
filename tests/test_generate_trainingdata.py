import trainingdata.generate_trainingdata as generate_trainingdata


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
    assert " ".join(grams) == " ".join(expected), grams

    grams_fit, weights_fit = generate_trainingdata.fit_grams(grams, weights)
    
    expected = ["i", "was", "happy with the", "movie but", "not", "with the popcorn"]
    assert " ".join(grams_fit) == " ".join(expected), grams_fit


def test_grams2():

    text = "i have seen this movie and i liked it very much"
    tokens = text.split()

    # make vocabulary with unigrams, bigrams and trigrams
    vocabulary = {
        "i" : 1,
        "i have" : 2,
        "have seen this" : 1,
        "seen this movie" : 1,
        "this movie" : 1,
        "this movie and" : 1,
        "i liked" : 1,
        "liked it" : 1,
        "it very" : 1,
        "very much" : 1,
    }

    sentiment = 1
    threshold = 0.0

    grams, weights  = generate_trainingdata.find_grams(tokens, vocabulary)
    
    expected = ["i have", "have seen this", "seen this movie", "this movie and", "i liked", "liked it", "it very", "very much"]

    assert " ".join(grams) == " ".join(expected), grams

    fit_grams, fit_weights = generate_trainingdata.fit_grams(grams, weights)
    
    expected = ["i", "have seen this", "movie and", "i liked", "it very", "much"]

    assert " ".join(fit_grams) == " ".join(expected), fit_grams


def test_grams3():

    # make a test similar to test_grams2 but with a different vocabulary and text
    
    text = "i havent seen this movie yet but i will see it soon with my cats"
    tokens = text.split()

    # make vocabulary with unigrams, bigrams and trigrams
    vocabulary = {
        "i" : 1,
        "havent" : 1,
        "seen this" : 1,
        "this movie yet" : 1,
        "yet but" : 1,
        "i will" : 1,
        "will see" : 1,
        "see it" : 1,
        "it soon" : 1,
        "soon with" : 1,
        "with my" : 1,
        "my cats" : 1,
    }

    sentiment = 1
    threshold = 0.0

    grams, weights  = generate_trainingdata.find_grams(tokens, vocabulary)
    
    fit_grams, fit_weights = generate_trainingdata.fit_grams(grams, weights)

    assert " ".join(fit_grams) == text, fit_grams
    #assert fit_weights == [1], fit_weights


def test_grams4():

    # make a similar test to test_grams3 but with a different vocabulary and text

    text = "this is a very good movie and i think it is one of the best movies i have ever seen"
    tokens = text.split()

    # make vocabulary with unigrams, bigrams and trigrams
    vocabulary = {
        "this is" : 1,
        "is a" : 1,
        "a very" : 1,
        "very good" : 1,
        "good movie" : 1,
        "movie and" : 1,
        "i think" : 1,
        "think it" : 1,
        "it is" : 1,
        "is one" : 1,
        "one of" : 1,
        "of the" : 1,
        "the best" : 1,
        "best movies" : 1,
        "movies i" : 1,
        "i have" : 1,
        "have ever" : 1,
        "ever seen" : 1,
    }

    sentiment = 1
    threshold = 0.0

    grams, weights  = generate_trainingdata.find_grams(tokens, vocabulary)

    fit_grams, fit_weights = generate_trainingdata.fit_grams(grams, weights)

    assert " ".join(fit_grams) == text, fit_grams


def test_convert_to_not_lemma():

    tokens = ["i", "have", "nt", "seen", "your", "cat", "s", "since", "yesterday"]
    token_map = [0, 1, 1, 2, 3, 4, 4, 5, 6]
    grammies = ["i have", "nt seen your", "cat", "s since yesterday"]
    
    new_grammies = generate_trainingdata.convert_to_not_lemma(token_map, tokens, grammies)

    expected_grammies = ["i havent", "seen your", "cats" ,"since yesterday"]
    
    assert new_grammies == expected_grammies, new_grammies


if __name__ == '__main__':
    
    #test_grams2() 
    #test_grams1()
    pass