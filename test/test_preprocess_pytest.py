import src.preprocess as preprocess


def test_clean_text():
    """
    Simple test to check that html tags, special characters, large numbers and 
    non-alphanumeric characters are handled correctly in the clean text function
    """

    textdata = ['<p> This is a test sentence with SPECIAL CHARACTERS @#@#$ and numbers 10 1000000000 100 1000 1000000000. </p>', 
                "<body> Didn't, shouldn't, wouldn't cat's </body>",
                "<i> 1029384610987246591827631582641"]
    textdata = preprocess.clean_text(textdata)

    assert textdata[0] == 'this is a test sentence with special characters and numbers 10', textdata[0]
    assert textdata[1] == "didnt shouldnt wouldnt cats", textdata[1]
    assert textdata[2] == "", textdata[2]


def test_lemmatize():
    
    """
    Simple test to check if lemmatization is working as expected
    """
    clean_texts = ["i didnt see this movie", "i have seen cats and dogs"]

    texts, tokens, lemmas, labels = preprocess.lemmatize(clean_texts, [0, 1])

    assert lemmas[0] == ['i', 'do', 'not', 'see', 'this', 'movie'], lemmas[0]
    assert lemmas[1] == ['i', 'have', 'see', 'cat', 'and', 'dog'], lemmas[1]

    assert tokens[0] == ['i', 'did', 'nt', 'see', 'this', 'movie'], tokens[0]
    assert tokens[1] == ['i', 'have', 'seen', 'cats', 'and', 'dogs'], tokens[1]

    assert texts[0] == ["i", "didnt", "see", "this", "movie"], texts[0]
    assert texts[1] == ["i", "have", "seen", "cats", "and", "dogs"], texts[1]


def test_map_tokens():

    """
    Simple test to check if mapping tokens to lemmas is working as expected
    """

    tokens = ["i", "did", "nt", "see", "this", "movie"]
    stext = ["i", "didnt", "see", "this", "movie"]

    ids = preprocess.map_tokens(stext, tokens)

    assert ids == [0, 1, 1, 2, 3, 4], ids

    tokens = ["i", "l", "o", "v", "e", "c", "a", "t", "s"]
    stext = ["i", "love", "cats"]

    ids = preprocess.map_tokens(stext, tokens)

    assert ids == [0, 1, 2, 3, 4, 5, 6, 7, 8], ids

        

if __name__ == "__main__":
    
    #test_clean_text()
    #test_lemmatize()
    #test_map_tokens()
    pass