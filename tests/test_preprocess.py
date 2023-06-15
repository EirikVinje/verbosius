import src.preprocess as preprocess

def test_clean_text():
    """
    Simple test to check that html tags, special characters, large numbers and 
    non-alphanumeric characters are handled correctly in the clean text function
    """

    textdata = ['<p> This is a test sentence with SPECIAL CHARACTERS @#@#$ and numbers 10 100 1000. </p>', 
                "<body> Didn't, shouldn't, wouldn't"]
    textdata = preprocess.clean_text(textdata)

    assert textdata[0] == 'this is a test sentence with special characters and numbers 10 100 1000', textdata[0]
    assert textdata[1] == 'didnt shouldnt wouldnt', textdata[1]

def test_lemmatize():
    """
    Simple test to check if lemmatization is working as expected
    """
    textdata = ["I walked to the store and bought some apples, I didn't buy any oranges"]
    textdata = preprocess.clean_text(textdata)
    output = preprocess.lemmatize(textdata)
    
    lemma_text = ' '.join(output[0][0]).lower()

    #print(textdata)
    assert lemma_text == "i walk to the store and buy some apple i do not buy any orange", lemma_text
    assert output[0][0] == ['I','walk','to','the','store','and','buy','some','apple','I','do','not','buy','any','orange'], output[0][0]
    assert output[0][1] == ['i', 'walked', 'to', 'the', 'store', 'and', 'bought', 'some', 'apples', 'i', 'did', "nt", 'buy', 'any', 'oranges'], output[0][1]
    assert output[0][2] == [11], output[0][2]

    textdata = ["yes i did it", "no i didn't do it"]
    processed_textdata = preprocess.clean_text(textdata)
    output = preprocess.lemmatize(processed_textdata)

    
    expected_token_text = ['yes i did it', "no i did nt do it"]
    expected_lemma_text = ['yes I do it', 'no I do not do it']
    expected_part_idx = [[], [3]]
    for index, text in enumerate(output):
        lemma_text = ' '.join(text[0]).lower()

        assert lemma_text == expected_lemma_text[index].lower(), lemma_text
        assert text[0] == expected_lemma_text[index].split(), text[0]
        assert text[1] == expected_token_text[index].split(), text[1]
        assert text[2] == expected_part_idx[index], text[2]


if __name__ == "__main__":
    test_clean_text()
    test_lemmatize()
    print('all tests passed in {}'.format(__file__))