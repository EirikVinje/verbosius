import re
from time import time, perf_counter

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

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

    print("Cleaning text...")
    for i in tqdm(range(len(textdata))):
    
        textdata[i] = strip_html(textdata[i])
        textdata[i] = textdata[i].lower()
        textdata[i] = _only_letters_pattern.sub(' ',textdata[i])
        textdata[i] = _no_long_numbers_pattern.sub('', textdata[i])
        textdata[i] = _no_multiple_quotes_pattern.sub("", textdata[i])
        textdata[i] = _no_multiple_spaces_pattern.sub(" ", textdata[i])
        textdata[i] = textdata[i].strip()

    print()
    return textdata


def lemmatize(texts : list, cores:int = 4, lemmatizer : str = "en_core_web_sm"):
    
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

    nlp = spacy.load(lemmatizer)
    docs = nlp.pipe(texts, n_process=cores) 

    texts = [text.split() for text in texts]
    tokens = []
    lemmas = []

    docs = list(docs)

    print("Lemmatizing text...")
    for i in tqdm(range(len(docs))):

        tokens.append([token.text for token in docs[i]])
        lemmas.append([token.lemma_.lower() for token in docs[i]])

    print()
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

    print("Mapping tokens...")
    for i in tqdm(range(len(stexts))):
        
        id = []
        topw = 0
        topt = 0
        
        while True:

            if topt >= len(tokens[i]) or topw >= len(stexts[i]):
                break

            elif tokens[i][topt] == stexts[i][topw]:
                id.append(topw)
                topw += 1 
                topt += 1

            elif tokens[i][topt] in stexts[i][topw]:
                top_len = len(tokens[i][topt])  
                id.append(topw)
                topt += 1

                while True:

                    if top_len == len(stexts[i][topw]):
                        topw += 1
                        break
                    
                    elif stexts[i][topw].find(tokens[i][topt], top_len) == top_len:
                        id.append(topw)
                        top_len += len(tokens[i][topt])
                        topt += 1
            
            else:
                id.append(-1)
                topt += 1

        ids.append(id)

    print()
    return ids



if __name__ == "__main__":
    
    sentence = ["I haven't seen this movie yet, but I will see it soon with my cat's."]

    cleaned_sentence = clean_text(sentence)
    print(cleaned_sentence)