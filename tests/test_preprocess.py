from preprocess.preprocess import Preprocess


def test_clean_text():
    
    textdata = ['<p> This is a test sentence with SPECIAL CHARACTERS @#@#$ and numbers 10 1000000000 100 1000 1000000000. </p>', 
                "<body> Didn't, shouldn't, wouldn't cat's </body>",
                "<i> 1029384610987246591827631582641"]
    
    
    textdata = Preprocess(0)._clean_text(textdata)    
    
    assert textdata[0] == 'this is a test sentence with special characters and numbers 10', textdata[0]
    assert textdata[1] == "didnt shouldnt wouldnt cats", textdata[1]
    assert textdata[2] == "", textdata[2]


def test_lemmatize():
    
    texts = ["i didnt see this movie", "i have seen cats and dogs"]

    tokens, lemmas = Preprocess(0)._lemmatize(texts)

    assert lemmas[0] == ['i', 'do', 'not', 'see', 'this', 'movie'], lemmas[0]
    assert lemmas[1] == ['i', 'have', 'see', 'cat', 'and', 'dog'], lemmas[1]

    assert tokens[0] == ['i', 'did', 'nt', 'see', 'this', 'movie'], tokens[0]
    assert tokens[1] == ['i', 'have', 'seen', 'cats', 'and', 'dogs'], tokens[1]


def test_map_tokens():

    tokens = [["i", "did", "nt", "see", "this", "movie"], ["i", "l", "o", "v", "e", "c", "a", "t", "s"]]
    texts = ["i didnt see this movie", "i love cats"]

    ids = Preprocess(0)._map_tokens(texts, tokens)

    assert ids[0] == [0, 1, 1, 2, 3, 4], ids[0]
    assert ids[1] == [0, 1, 1, 1, 1, 2, 2, 2, 2], ids[1]


if __name__ == "__main__":

    # test_clean_text()
    # test_lemmatize()
    test_map_tokens()
    print("<done tests:", __file__, ">")


