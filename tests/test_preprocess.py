import src.preprocess as preprocess

def test_clean_text():

    textdata = ['<p> This is a test sentence with SPECIAL CHARACTERS @#@#$ and numbers 10 100 1000. </p>']
    textdata = preprocess.clean_text(textdata)

    # print(textdata)

    assert textdata[0] == 'this is a test sentence with special characters and numbers 10 100 1000', textdata[0]


def test_lemmatize():

    textdata = ['I walked to the store and bought some apples']
    textdata = preprocess.clean_text(textdata)
    textdata = preprocess.lemmatize(textdata)

    #print(textdata)

    assert textdata[0] == 'I walk to the store and buy some apple', textdata[0]


if __name__ == "__main__":
    test_clean_text()
    test_lemmatize()
    print('all tests passed in {}'.format(__file__))