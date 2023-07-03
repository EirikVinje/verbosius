import re
import spacy

import numpy as np
import pandas as pd

from time import time, perf_counter
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def strip_html(textdata):
    
    soup = BeautifulSoup(textdata, "html.parser")
    return soup.get_text()


def clean_text(textdata):
    
    """
    Parameters:
    -----------
    textdata : list
        List of documents to be cleaned

    Returns:
    --------
    textdata : list
        List of cleaned documents

    """

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


def lemmatize(texts : list, cores:int = 4):
    
    """
    Parameters:
    -----------
    texts : list
        List of documents to be lemmatized
    
    labels : list
        List of labels for the strings

    Returns:
    --------
    texts : list
        List of documents split into words

    tokens : list
        List of documents split into tokens

    lemmas : list
        List of documents with tokens transformed into lemmas

    labels : list
        List of labels to each document
    """

    nlp = spacy.load("en_core_web_sm")
    docs = nlp.pipe(texts, n_process=cores) 

    texts = [text.split() for text in texts]
    tokens = []
    lemmas = []

    for doc in docs:

        tokens.append([token.text for token in doc])
        lemmas.append([token.lemma_.lower() for token in doc])

    return texts, tokens, lemmas


def map_tokens(stexts : list, tokens : list):

    """
    Parameters:
    -----------
    stext : list
        List of document split into words
    
    tokens : list
        List of document split into tokens

    Returns:
    --------
    ids : list
        List of ids correseponding to the tokens in the document

    """

    ids = []

    for stext, token in zip(stexts, tokens):
        
        id = []
        topw = 0
        topt = 0
        
        while True:

            if topt >= len(token) or topw >= len(stext):
                break

            elif token[topt] == stext[topw]:
                id.append(topw)
                topw += 1 
                topt += 1

            elif token[topt] in stext[topw]:
                top_len = len(token[topt])  
                id.append(topw)
                topt += 1

                while True:

                    if top_len == len(stext[topw]):
                        topw += 1
                        break
                    
                    elif stext[topw].find(token[topt], top_len) == top_len:
                        id.append(topw)
                        top_len += len(token[topt])
                        topt += 1
            
            else:
                id.append(-1)
                topt += 1

        ids.append(id)

    return ids



if __name__ == "__main__":

    sentence = ["I haven't seen this movie yet, but I will see it soon with my cat's."]

    cleaned_sentence = clean_text(sentence)
    print(cleaned_sentence)