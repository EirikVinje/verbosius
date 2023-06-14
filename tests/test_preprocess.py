import src.preprocess as preprocess

def test_clean_text():

    textdata = ['<p> This is a test sentence with SPECIAL CHARACTERS @#@#$ and numbers 10 100 1000. </p>']
    textdata = preprocess.clean_text(textdata)

    # print(textdata)

    assert textdata[0] == 'this is a test sentence with special characters and numbers 10 100 1000', textdata[0]


def test_lemmatize():

    textdata = ["I walked to the store and bought some apples, I didn't buy any oranges"]
    textdata = preprocess.clean_text(textdata)
    output = preprocess.lemmatize(textdata)

    lemma_text = ' '.join(output[0]).lower()

    #print(textdata)
    assert lemma_text == "i walk to the store and buy some apple i do not buy any orange", lemma_text
    assert output[0] == ['I','walk','to','the','store','and','buy','some','apple','I','do','not','buy','any','orange'], output[0]
    assert output[1] == ['i', 'walked', 'to', 'the', 'store', 'and', 'bought', 'some', 'apples', ',', 'i', 'did', "nt", 'buy', 'any', 'oranges'], output[1]
    assert output[2] == [11], output[2]


if __name__ == "__main__":
    test_clean_text()
    test_lemmatize()
    print('all tests passed in {}'.format(__file__))