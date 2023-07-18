from collections import Counter
from copy import deepcopy
import pickle
import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch

import green_tsetlin as gt
import config as config


def rulemaker(data):
    
    """
    Trains the Tsetlin Machine and creates a RulePredictor from the trained Tsetlin Machine that is used to
    weight the tokens in the data.

    Parameters:
    -----------
    data : dict("train" : [...], "test" : [...], "distributer" : str, "n_classes" : int)

    Returns:
    --------
    rp : RulePredictor
        RulePredictor created from the trained Tsetlin Machine
    
    """



    train_x = [instance["lemmas"] for instance in data["train"]]
    train_y = [instance["label"] for instance in data["train"]]
    test_x = [instance["lemmas"] for instance in data["test"]]
    test_y = [instance["label"] for instance in data["test"]]
    
    train_y = np.array(train_y, dtype=np.uint32)
    test_y = np.array(test_y, dtype=np.uint32)

    vectorizer = CountVectorizer(max_features=config.MAX_FEATURES,
                                 max_df=config.MAX_DF, 
                                 min_df=config.MIN_DF,
                                 ngram_range=config.N_GRAM_RANGE,
                                 binary=True,
                                 dtype=np.uint8,
                                 stop_words = config.STOPWORDS)
    
    train_x_bin = vectorizer.fit_transform([" ".join(x) for x in train_x])
    test_x_bin = vectorizer.transform([" ".join(x) for x in test_x])
    feature_names = vectorizer.get_feature_names_out()


    tm = gt.TsetlinMachine(n_literals=train_x_bin.shape[1], 
                           n_clauses=config.NUMBER_OF_CLAUSES, 
                           n_classes=data["n_classes"],
                           s=config.S, 
                           n_literal_budget=config.LITERAL_BUDGET)

    train_x_bin = train_x_bin.todense()
    test_x_bin = test_x_bin.todense()

    for i in range(len(data["train"])):
        data["train"][i]["bin"] = train_x_bin[i]
        data["test"][i]["bin"] = test_x_bin[i]

    tm.set_train_data(train_x_bin[:1:], train_y[:1:])
    tm.set_test_data(test_x_bin, test_y)
    trainer = gt.Trainer(config.T, 
                         n_epochs=config.TM_EPOCHS, 
                         seed=32, 
                         n_jobs=6, 
                         early_exit_acc=config.EARLY_STOP_ACC,
                         progress_bar=False)

    trainer.train(tm)    

    rp = gt.RulePredictor()
    fm = list(range(train_x_bin.shape[1]))
    rp.create_from_state(tm.get_state(), fm)
    
    return rp, feature_names
    

def weight_tokens(lemmas, tokens, vocabulary, token_map):
    """

    Weights each token separated by a space in the text based on the weights of the n-grams in the vocabulary.

    Parameters:
    -----------
    lemmas : list
        List of lemmas
    tokens : list
        List of tokens
    vocabulary : dict
        Dictionary of ngrams with weights
    token_map : list
        List of token ids
    
    Returns:
    --------
    new_toks : list
        List of tokens with ngrams connected
    
    new_weights : list
        List of weights for new tokens
    
    """

    weights = np.zeros(len(lemmas))
    for i, lemma in enumerate(lemmas):

        unigram = lemma if i < len(lemmas) else None
        bigram = "{} {}".format(lemma, lemmas[i+1]) if i+1 < len(lemmas) else None
        trigram = "{} {} {}".format(lemma, lemmas[i+1], lemmas[i+2]) if i+2 < len(lemmas) else None
        
        if trigram in vocabulary.keys() and trigram is not None:
            
            tri_w = vocabulary[trigram] * 1/3
            weights[i] += tri_w
            weights[i+1] += tri_w
            weights[i+2] += tri_w

        if bigram in vocabulary.keys() and bigram is not None:
            
            bi_w = vocabulary[bigram] * 1/2 
            weights[i] += bi_w
            weights[i+1] += bi_w

        if unigram in vocabulary.keys():
            
            uni_w = vocabulary[unigram]
            weights[i] += uni_w


    new_toks, new_weights = connect_tokens(tokens, weights, token_map)
    return new_toks, new_weights


def connect_tokens(tokens, weights, token_map):
    """
    Converts the lemma tokens to the original tokens and connects the tokens that are connected by the same id.

    Parameters:
    -----------
    tokens : list
        List of tokens

    weights : list
        List of weights for each token
    
    token_map : list
        List of token ids
    
    Returns:
    --------
    new_toks : list
        List of original tokens with weights connected
    
    new_weights : list
        List of weights for new tokens
    """



    new_toks = []
    new_weights = []
    pre_id = None

    for i, token in enumerate(tokens):
        
        if pre_id is not None:

            curr_id = token_map[i]

            if curr_id == pre_id:
                new_toks[-1] += token
                new_weights[-1] += weights[i]
                pre_id = token_map[i]
            
            else:
                new_toks.append(token)
                new_weights.append(weights[i])
                pre_id = token_map[i]

        else:
            new_toks.append(token)
            new_weights.append(weights[i])
            pre_id = token_map[i]

    return new_toks, new_weights


def label_tokens(sentiment, weights, threshold : float = 0.0):

    """
    Parameters:
    -----------
    sentiment : int
        sentiment of the text
    weights : list
        list of weights for each n-gram
    threshold : float
        threshold for the weights
    
    Returns:
    --------
    labels : list
        list of labels for each n-gram
    """

    if sentiment == 1:
        labels = [2 if x > threshold else 1 if x < -threshold else 0 for x in weights]
    else:
        labels = [1 if x > threshold else 2 if x < -threshold else 0 for x in weights]
        
    return labels


def do_weighting(data, feature_names, rm):
    """
    
    Applies the weighting of tokens to the data and returns the new data.

    Parameters:
    -----------
    data : list
        list of dicts with the data
    
    feature_names : list
        list of feature names
    
    rm : RulePredictor
        RulePredictor used to weight the tokens, generated from the Tsetlin Machine

    Returns:
    --------
    all_x : list
        list of dicts with the new data
    
    """


    all_x = []

    for i, inst in enumerate(data):
        
        y = inst["label"]
        bin_x = inst["bin"]
        lemmas_x = inst["lemmas"]
        tokens_x = inst["tokens"]
        tokenmap_x = inst["token_ids"]

        prediction, expl = rm.predict(bin_x, explain=True)

        if y == prediction:

            vocabulary = {feature_names[i]: expl[i] for i in range(len(feature_names))}

            newtokens_x, weights_x = weight_tokens(lemmas_x, tokens_x, vocabulary, tokenmap_x)

            labels = label_tokens(y, weights_x)

            new_x = {"tokens" : newtokens_x,
                     "weights" : weights_x,
                     "text" : " ".join(newtokens_x),
                     "sentiment" : y,
                     "labels" : labels}

            all_x.append(new_x)

    return all_x


def write_data(data, output, dataset, batchdist_n, n):
    
    path = os.path.join(output, f"{dataset}_batchdist_{batchdist_n}")

    if not os.path.exists(path):
        os.mkdir(path)

    file = open(os.path.join(path, f"batch_{n}.pkl"), "wb")
    pickle.dump(data, file)


if __name__ == "__main__":
    print("Module")