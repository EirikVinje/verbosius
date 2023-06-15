import re
from time import time, perf_counter

import spacy
import pandas as pd
from bs4 import BeautifulSoup


def load_data(path):

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

    return X_train, Y_train, X_test, Y_test


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
        textdata[i] = _no_multiple_quotes_pattern.sub("", textdata[i])
        textdata[i] = textdata[i].strip()

    return textdata


def concatenate_lemmas(lemma_output):
    """
    lemma_output: list of lemmas for each sentence
    """
    sentences = []

    for lemmas in lemma_output:
        sentence = ' '.join(lemmas)
        sentences.append(sentence.lower())
    
    return sentences




def lemmatize(textdata, cores:int = 1):

    nlp = spacy.load('en_core_web_sm')
    docs = nlp.pipe(textdata, batch_size=1000, disable=["parser", "ner"], n_process=cores)
    
    
    lemmas_complete = []
    tokens_complete = []
    part_idxs_complete = []
    
    for doc in docs:
        lemmas = []
        tokens = []
        part_idxs = []
        for idx, token in enumerate(doc):
            if len(token.lemma_.strip()) >= 1:
                tokens.append(token)
                lemmas.append(token.lemma_.strip())
                if token.pos_ == 'PART':
                    part_idxs.append(idx)
        lemmas_complete.append(lemmas)
        tokens_complete.append(tokens)
        part_idxs_complete.append(part_idxs)

    return lemmas_complete, tokens_complete, part_idxs_complete






if __name__ == "__main__":

    #test_clean_text()
    #test_lemmatize()

    directory = '/home/kolla/projects/datasets/imdb'

    X_train, Y_train, X_test, Y_test = load_data(directory)

    X_train = clean_text(X_train)
    X_test = clean_text(X_test)

    X_train = lemmatize(X_train)

    print(len(X_train))

