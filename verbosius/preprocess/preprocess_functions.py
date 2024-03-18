import re
import pickle
import os
import gzip
import warnings

from bs4 import BeautifulSoup
import spacy

import config as config


warnings.filterwarnings("ignore", category=UserWarning, message="MarkupResemblesLocatorWarning")


def strip_html(textdata):
    
    with warnings.catch_warnings():
    
        warnings.simplefilter("ignore")

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

    if type(textdata) == type(None):
        return None

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
    
    if type(texts) == type(None):
        return None, None, None

    nlp = spacy.load(lemmatizer)
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

    if type(stexts) == type(None):
        return None

    ids = []

    for i in range(len(stexts)):
        
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

    return ids


def stage_data(token_x, lemma_x, token_ids_x, y, orig_labels, x):

    data_dicts = []
    
    if type(token_x) == type(None):
        return None
    
    for i in range(len(y)):
        
        instance = {
                    "tokens": token_x[i],
                    "sentiment": y[i],
                    "lemmas": lemma_x[i],
                    "orig_text" : x[i],
                    "token_ids": token_ids_x[i],
                    "orig_labels": orig_labels[i] if orig_labels is not None else None}

        data_dicts.append(instance)

    return data_dicts


def write_data(data, path, n):
    
    if not os.path.exists(path):
        os.mkdir(dir)

    dir = os.path.join(path, f"preprocess_chunk_{n}_.pkl")

    with gzip.open(dir, "wb") as f:
        pickle.dump(data, f)


def set_directory(chunkdist_name):

        if not os.path.exists(config.root):
            assert False, f"Directory {config.root} does not exist, please create it before continuing"

        preprocess_path = os.path.join(config.root, "preprocess")
        if not os.path.exists(preprocess_path):
            os.mkdir(preprocess_path)
        
        preprocess_path = os.path.join(preprocess_path, chunkdist_name)
        if not os.path.exists(preprocess_path):
            os.mkdir(preprocess_path)
        else:
            assert False, f"Directory {preprocess_path} already exists, please remove it before continuing" 

        chunking_path = os.path.join(config.root, "chunking")
        if not os.path.exists(chunking_path):
            assert False, f"Chunking folder {chunking_path} does not exist, please check your input"

        chunking_path = os.path.join(chunking_path, chunkdist_name)
        if not os.path.exists(chunking_path):
            assert False, f"Chunk distribution {chunking_path} does not exist, please check your input"


if __name__ == "__main__":

    print("Module")