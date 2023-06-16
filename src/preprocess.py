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
    _no_long_numbers_pattern = re.compile(r'\d{5,}')
    _no_multiple_quotes_pattern = re.compile(r"'+")

    for i in range(len(textdata)):
    
        textdata[i] = strip_html(textdata[i])
        textdata[i] = textdata[i].lower()
        textdata[i] = _only_letters_pattern.sub(' ',textdata[i])
        textdata[i] = _no_long_numbers_pattern.sub('', textdata[i])
        # textdata[i] = _no_multiple_quotes_pattern.sub("", textdata[i])
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




def lemmatize(textdata, cores:int = 1):

    nlp = spacy.load('en_core_web_sm')
    docs = nlp.pipe(textdata, batch_size=100, disable=["parser", "ner"], n_process=cores)
    
    
    lemmas = []
    texts = []
    part_idxs = []
    
    for doc in docs:
        lemmas.append([token.lemma_.lower() for token in doc if len(token.lemma_.strip()) >= 1])
        texts.append([token.text for token in doc])
        part_idxs.append([idx for idx, token in enumerate(doc) if token.pos_ == 'PART'])


    return lemmas, texts, part_idxs


if __name__ == "__main__":
    """test_text = ["This isn't a test sentence, I'm a tests sentence"]
    test_text = clean_text(test_text)
    print(test_text)
    lemmas, texts, part_idxs = lemmatize(test_text)
    print(lemmas)
    print(texts)
    print(part_idxs)"""

    data = pd.read_csv('IMDB Dataset.csv')
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.values.tolist()
    X_train = [x[0] for x in data[:25000]]
    Y_train = [x[1] for x in data[:25000]]
    X_test = [x[0] for x in data[25000:]]
    Y_test = [x[1] for x in data[25000:]]
    # X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    print(len(X_train))

    X_train_cleaned = clean_text(X_train)
    X_test_cleaned = clean_text(X_test)
    print('cleaned')

    X_train_lemmas, X_train_tokens, X_train_part_idxs = lemmatize(X_train_cleaned, cores=7)
    X_test_lemmas, X_test_tokens, X_test_part_idxs = lemmatize(X_test_cleaned, cores=7)
    print('lemmatized')

    X_train_final = concatenate_lemmas(X_train_lemmas)
    X_test_final = concatenate_lemmas(X_test_lemmas)
    print('concatenated')

    cv = CountVectorizer(binary=True,
                         max_features=5000,
                         min_df=5,
                         max_df=0.5,
                         stop_words='english')

    X = cv.fit_transform(X_train_final)
    X_test = cv.transform(X_test_final)

    clf = LogisticRegression().fit(X, Y_train)
    print("Training Accuracy: %s" % clf.score(X, Y_train))
    print("Test Accuracy: %s" % clf.score(X_test, Y_test))