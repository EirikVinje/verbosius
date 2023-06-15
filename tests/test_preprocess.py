import src.preprocess as preprocess

def test_clean_text():
    """
    Simple test to check that html tags, special characters, large numbers and 
    non-alphanumeric characters are handled correctly in the clean text function
    """

    textdata = ['<p> This is a test sentence with SPECIAL CHARACTERS @#@#$ and numbers 10 1000000000 100 1000 1000000000. </p>', 
                "<body> Didn't, shouldn't, wouldn't </body>",
                "<i> 1029384610987246591827631582641"]
    textdata = preprocess.clean_text(textdata)

    assert textdata[0] == 'this is a test sentence with special characters and numbers 10  100 1000', textdata[0]
    assert textdata[1] == 'didnt shouldnt wouldnt', textdata[1]
    assert textdata[2] == "", textdata[2]

def test_lemmatize():
    """
    Simple test to check if lemmatization is working as expected
    """
    textdata = ["I walked to the store and bought some apples, I didn't buy any oranges"]
    processed_textdata = preprocess.clean_text(textdata)
    lemmas, tokens, part_idxs = preprocess.lemmatize(processed_textdata)
    
    
    assert lemmas[0] == ['I','walk','to','the','store','and','buy','some','apple','I','do','not','buy','any','orange'], lemmas[0]
    assert len(tokens) == len(lemmas), len(tokens)
    assert part_idxs[0] == [11], part_idxs[0]



    textdata = ["yes i did it", "no i didn't do it"]
    processed_textdata = preprocess.clean_text(textdata)
    lemmas, tokens, part_idxs = preprocess.lemmatize(processed_textdata)


    expected_lemmas = [['yes', 'I', 'do', 'it'], ['no', 'I', 'do', 'not', 'do', 'it']]
    expected_part_idx = [[], [3]]


    assert lemmas == expected_lemmas, lemmas
    assert len(tokens) == len(lemmas), len(tokens)
    assert part_idxs == expected_part_idx, part_idxs


def test_concatenate_lemmas():
    """
    Simple test to check if combining lemmas to sentences is working as expected
    """
    lemma_output = [['I', 'walk', 'to', 'the', 'store', 'and', 'buy', 'some', 'apple', 'I', 'do', 'not', 'buy', 'any', 'orange'], 
                    ['yes', 'I', 'do', 'it'], 
                    ['no', 'I', 'do', 'not', 'do', 'it']]
    
    sentences = preprocess.concatenate_lemmas(lemma_output)

    assert sentences == ['i walk to the store and buy some apple i do not buy any orange', 
                         'yes i do it', 
                         'no i do not do it'], sentences

def test_preprocess_from_text_to_lemma_text():
    data = ["<p>This is a test sentence that should try to assess that the complete pipeline of the procesing works as intended, and that it doesn't mess anything up. This is very important as I will receive a sum of 10000000 dollars if it works.</p>"]
    processed_data = preprocess.clean_text(data)
    lemmas, tokens, part_idxs = preprocess.lemmatize(processed_data)

    sentences = preprocess.concatenate_lemmas(lemmas)

    assert sentences == ['this be a test sentence that should try to assess that the complete pipeline of the procesing work as intend and that it do not mess anything up this be very important as i will receive a sum of dollar if it work'], sentences
    assert lemmas == [['this', 'be', 'a', 'test', 'sentence', 'that', 'should', 'try', 'to', 'assess', 'that', 'the', 'complete', 'pipeline', 'of', 'the', 'procesing', 'work', 'as', 'intend', 'and', 'that', 'it', 'do', 'not', 'mess', 'anything', 'up', 'this', 'be', 'very', 'important', 'as', 'I', 'will', 'receive', 'a', 'sum', 'of', 'dollar', 'if', 'it', 'work']], lemmas
    assert len(tokens) == len(lemmas), len(tokens)
    assert part_idxs == [[8, 24]], part_idxs

if __name__ == "__main__":
    test_clean_text()
    test_lemmatize()
    test_concatenate_lemmas()
    test_preprocess_from_text_to_lemma_text()
    print('all tests passed in {}'.format(__file__))