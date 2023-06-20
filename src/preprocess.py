import re
import spacy

import numpy as np
import pandas as pd

from time import time, perf_counter
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

"""def load_data(path):

    df_train = pd.read_csv(f'{directory}/imdb_train.csv')
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = pd.read_csv(f'{directory}/imdb_test.csv')
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    train_data = df_train.values.tolist()
    test_data = df_test.values.tolist()

    X_train = [x[0] for x in train_data]
    Y_train = [x[1] for x in train_data]
    X_test = [x[0] for x in test_data]
    Y_test = [x[1] for x in test_data]

    return X_train, Y_train, X_test, Y_test"""


def strip_html(textdata):
    
    soup = BeautifulSoup(textdata, "html.parser")
    return soup.get_text()


def clean_text(textdata):
    
    _only_letters_pattern = re.compile(r"[^A-Za-z0-9']+")
    _no_long_numbers_pattern = re.compile(r'\d{3,}')
    _no_multiple_quotes_pattern = re.compile(r"'+")
    _no_multiple_spaces_pattern = re.compile(r"  +")

    for i in range(len(textdata)):
    
        textdata[i] = strip_html(textdata[i])
        textdata[i] = textdata[i].lower()
        textdata[i] = _only_letters_pattern.sub(' ',textdata[i])
        textdata[i] = _no_long_numbers_pattern.sub('', textdata[i])
        textdata[i] = _no_multiple_quotes_pattern.sub("", textdata[i])
        textdata[i] = _no_multiple_spaces_pattern.sub(" ", textdata[i])
        textdata[i] = textdata[i].strip()

    return textdata


def concatenate_lemmas(lemma_output):
    """
    lemma_output: list of lemmas for each sentence
    """
    sentences = []

    for lemmas in lemma_output:
        sentence = ' '.join(lemmas)
        sentences.append(sentence)
    
    return sentences


def lemmatize(text, nlp, cores:int = 4):

    stext = text[0].split()
    docs = nlp(text[0])
    tokens = [doc.text for doc in docs]
    lemmas = [doc.lemma_ for doc in docs]
    ids = []

    i = 0
    j = 0
    idx = 0
    
    while True:

        if tokens[i] == stext[j]:
            ids.append(idx)
            i += 1
            j += 1
        
        elif tokens[i] + tokens[i+1] == stext[j]:
            ids.append(idx)
            ids.append(idx)
            i += 2
            j += 1

        else:
            ids.append(idx)
            i += 1
            j += 1


        if i == len(tokens) or j == len(stext):
            break
    
        idx += 1

    return tokens, lemmas, ids


if __name__ == "__main__":
    
    sentence = ["I haven't seen this movie yet, but I will see it soon with my cat's."]

    cleaned_sentence = clean_text(sentence)
    print(cleaned_sentence)