import pickle

import numpy as np

import trainingdata.helper_functions as helper_functions


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

    grams, weights  = helper_functions.find_grams(tokens, vocabulary)
    
    expected = ["i", "was happy", "happy with the", "the movie but", "not with", "with the popcorn"]
    assert " ".join(grams) == " ".join(expected), grams

    grams_fit, weights_fit = helper_functions.fit_grams(grams, weights)
    
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

    grams, weights  = helper_functions.find_grams(tokens, vocabulary)
    
    expected = ["i have", "have seen this", "seen this movie", "this movie and", "i liked", "liked it", "it very", "very much"]

    assert " ".join(grams) == " ".join(expected), grams

    fit_grams, fit_weights = helper_functions.fit_grams(grams, weights)
    
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

    grams, weights  = helper_functions.find_grams(tokens, vocabulary)
    
    fit_grams, fit_weights = helper_functions.fit_grams(grams, weights)

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

    grams, weights  = helper_functions.find_grams(tokens, vocabulary)

    fit_grams, fit_weights = helper_functions.fit_grams(grams, weights)

    assert " ".join(fit_grams) == text, fit_grams


def test_convert_to_not_lemma1():

    tokens = ["i", "have", "nt", "seen", "your", "cat", "s", "since", "yesterday"]
    token_map = [0, 1, 1, 2, 3, 4, 4, 5, 6]
    grammies = ["i have", "nt seen your", "cat", "s since yesterday"]
    weights = [1, 1, 1, 1]
    
    new_grammies, weights = helper_functions.convert_to_not_lemma(token_map, tokens, grammies, weights)

    expected_grammies = ["i havent", "seen your", "cats" ,"since yesterday"]
    
    assert new_grammies == expected_grammies, new_grammies


def test_convert_to_not_lemma2():

    vocabulary = pickle.load(open("/home/kolla/projects/verbosius_data/vocabulary.pkl", "rb"))
    
    tokens = ['lets', 'put', 'it', 'this', 'way', 'i', 'actually', 'get', 'this', 'movie', 'i', 'get', 'what', 'the', 'writer', 'directer', 'was', 'trying', 'to', 'do', 'i', 'understand', 'that', 'the', 'dialog', 'was', 'meant', 'to', 'be', 'dry', 'and', 'emotionless', 'i', 'understand', 'that', 'the', 'plot', 'was', 'supposed', 'to', 'be', 'non', 'climactic', 'and', 'stale', 'that', 'was', 'what', 'the', 'writer', 'director', 'was', 'going', 'for', 'a', 'very', 'very', 'very', 'dry', 'humor', 'comedy', 'with', 'all', 'that', 'understanding', 'i', 'still', 'think', 'the', 'movie', 'sucked', 'it', 'seemed', 'like', 'the', 'writer', 'director', 'was', 'trying', 'to', 'recreate', 'napolean', 'dynamite', 'with', 'this', 'movie', 'it', 'had', 'all', 'of', 'the', 'same', 'features', 'even', 'the', 'main', 'character', 'behaved', 'similar', 'to', 'napolean', 'but', 'napolean', 'dynamite', 'was', 'actually', 'funny', 'its', 'script', 'worked', 'this', 'movie', 'is', 'not', 'it', 'has', 'no', 'purpose', 'well', 'let', 'me', 'rephrase', 'that', 'its', 'only', 'purpose', 'is', 'to', 'rip', 'off', 'napolean', 'dynamite', 'and', 'try', 'to', 'capture', 'that', 'look', 'and', 'feel', 'too', 'bad', 'it', 'did', 'nt', 'work']
    
    token_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 143, 144]

    found_grammies, weights = helper_functions.find_grams(tokens, vocabulary)

    fit_grammies, fit_weights = helper_functions.fit_grams(found_grammies, weights)


def test_fit_grams():

    sentence = "i have seen this movie and i liked it very very much"
    tokens = sentence.split()

    vocabulary = {
        "i have seen" : 1,
        "have seen this" : 1,
        "this movie" : 1,
        "movie and i" : 1,
        "and i liked" : 1,
        "liked it very" : 1,
        "very" : 1
    }

    found_grammies, weights = helper_functions.find_grams(tokens, vocabulary)

    expected = ["i have seen", "have seen this", "this movie", "movie and i ", "and i liked", "liked it very", "very", "much"]
    assert found_grammies == expected, found_grammies

    fit_grammies, fit_weights = helper_functions.fit_grams(found_grammies, weights)

    expected = ["i have seen", "this", "movie and i", "liked it very", "very", "much"]
    assert fit_grammies == expected, fit_grammies


def test_find_grams():

    toks = ['lets', 'put', 'it', 'this', 'way']

    vocab = {
        "lets put it" : 1,
        "this" : 1,
        "this way" : 1
    }

    grams, _ = helper_functions.find_grams(toks, vocab)

    expected = ["lets put it", "this way"]
    assert grams == expected, grams

    vocab = {
        "lets put it" : 1,
        "put it this" : 1,
        "it this way" : 1
    }

    grams, _ = helper_functions.find_grams(toks, vocab)

    expected = ["lets put it", "put it this", "it this way"]
    assert grams == expected, grams

    vocab = {
        "lets put" : 1,
        "put" : 1,
        "it this" : 1,
        "this way" : 1
    }

    grams, _ = helper_functions.find_grams(toks, vocab)

    expected = ["lets put", "it this", "this way"]
    assert grams == expected, grams

    vocab = {
        "lets" : 1,
        "put" : 1,
        "it" : 1,
        "this" : 1,
        "way" : 1
    }

    grams, _ = helper_functions.find_grams(toks, vocab)

    expected = ["lets", "put", "it", "this", "way"]
    assert grams == expected, grams

    vocab = {
        "lets put it" : 1,
        "put it" : 1,
        "it" : 1,
        "put it this" : 1,
        "it this way" : 1,
    }

    grams, _ = helper_functions.find_grams(toks, vocab)

    assert grams == [], grams


def test_new_new_weight_grams():

    tokens = ["i", "have", "nt", "seen", "your", "cat", "s", "since", "yesterday", "and", "i", "like", "them", "very", "very", "much", "kjell", "aage"]
    
    lemmas = ["i", "have", "not", "see", "your", "cat", "s", "since", "yesterday", "and", "i", "like", "them", "very", "very", "much", "kjell", "aage"]
    
    token_map = [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    vocabulary = {
        "i have" : 2,
        "i have not" : 3,
        "have not see" : 3,
        "not" : 1,
        "not see" : 2,
        "your" : 1,
        "your cat" : 2,
        "cat" : 1,
        "cat s" : 2,
        "s" : 1,
        "since yesterday" : 2,
        "yesterday" : 1,
        "yesterday and i" : 3,
        "like them" : 2,
        "them" : 1,
        "them very" : 2,
        "very" : 1,
        "very much" : 2
    }
    
    tokens, weights = helper_functions.weight_tokens(lemmas, tokens, vocabulary, token_map)
    
    expected_tokens = ["i", "havent", "seen", "your", "cats", "since", "yesterday", "and", "i", "like", "them", "very", "very", "much", "kjell", "aage"]
    assert tokens == expected_tokens, tokens

    expected_weights = [2, 7, 2, 2, 5, 1, 3, 1, 1, 1, 3, 2, 2, 1, 0, 0]
    expected_weights = [float(i) for i in expected_weights]
    
    assert weights == expected_weights, weights







if __name__ == '__main__':
    
    pass