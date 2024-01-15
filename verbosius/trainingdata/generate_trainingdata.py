from collections import Counter
from copy import deepcopy
import pickle
import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
import green_tsetlin as gt

import config as config


def rulemaker(train_x, train_y, error_params : bool = False):
    
    """
    Generates the RulePredictor from the Tsetlin Machine. This enables us to wieght the tokens in the data.

    Parameters:
    -----------
    train_x : list

    train_y : list

    error_params : bool
        If True, the parameters for the Tsetlin Machine will be the error parameters.

    """

    MAX_FEATURES = config.MAX_FEATURES
    CV_MAX_FEATURES = config.CV_MAX_FEATURES
    MAX_DF = config.MAX_DF
    MIN_DF = config.MIN_DF
    N_GRAM_RANGE = config.N_GRAM_RANGE
    NUMBER_OF_CLAUSES = config.NUMBER_OF_CLAUSES
    LITERAL_BUDGET = config.LITERAL_BUDGET
    S = config.S
    T = config.T
    TM_EPOCHS = config.TM_EPOCHS
    EARLY_STOP_ACC = config.EARLY_STOP_ACC
    STOPWORDS = config.STOPWORDS
    N_JOBS = config.N_JOBS
    SEED = config.seed
    SKB_score_func = config.SKB_score_func
    NUM_TM_LABELS = config.NUM_TM_LABELS

    if error_params:
        
        MAX_FEATURES = config.ERROR_MAX_FEATURES
        NUMBER_OF_CLAUSES = config.ERROR_NUMBER_OF_CLAUSES
        LITERAL_BUDGET = config.ERROR_LITERAL_BUDGET
        S = config.ERROR_S
        T = config.ERROR_T
        MAX_DF = config.ERROR_MAX_DF
        MIN_DF = config.ERROR_MIN_DF

    train_y = np.array(train_y, dtype=np.uint32)
    
    vectorizer = CountVectorizer(max_features=CV_MAX_FEATURES,
                                 max_df=MAX_DF, 
                                 min_df=MIN_DF,
                                 ngram_range=N_GRAM_RANGE,
                                 binary=True,
                                 dtype=np.uint8,
                                 stop_words =STOPWORDS)
    

    train_x_bin = vectorizer.fit_transform([" ".join(x) for x in train_x])

    _feature_names = vectorizer.get_feature_names_out()

    SKB = SelectKBest(score_func=SKB_score_func, k=MAX_FEATURES)

    SKB.fit(train_x_bin, train_y)
    feature_names = SKB.get_feature_names_out(input_features=_feature_names)
    assert feature_names.all() == _feature_names[SKB.get_support(indices=True)].all()

    train_x_bin = SKB.transform(train_x_bin).toarray()
    
    tm = gt.TsetlinMachine(n_literals=train_x_bin.shape[1], 
                           n_clauses=NUMBER_OF_CLAUSES, 
                           n_classes=NUM_TM_LABELS,
                           s=S,
                           n_literal_budget=LITERAL_BUDGET)

    copy_train_x_bin = deepcopy(train_x_bin)
    copy_train_y = deepcopy(train_y)

    c_train_x, c_val_x, c_train_y, c_val_y = train_test_split(copy_train_x_bin, copy_train_y, test_size=0.2, random_state=SEED)

    tm.set_train_data(c_train_x, c_train_y)
    tm.set_test_data(c_val_x, c_val_y)
    
    trainer = gt.Trainer(threshold=T, 
                         n_epochs=TM_EPOCHS, 
                         seed=SEED, 
                         n_jobs=N_JOBS, 
                         early_exit_acc=EARLY_STOP_ACC,
                         progress_bar= True)

    trainer.train(tm)    

    rp = gt.RulePredictor()
    fm = list(range(train_x_bin.shape[1]))
    rp.create_from_state(tm.get_state(), fm)
    
    return rp, feature_names, train_x_bin
    

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

    # if sentiment == 1:
    #     labels = [2 if x > threshold else 1 if x < -threshold else 0 for x in weights]
    # else:
    #     labels = [1 if x > threshold else 2 if x < -threshold else 0 for x in weights]

    if sentiment == 2 or sentiment == 0:
        labels = [2 if x > threshold else 1 if x < -threshold else 0 for x in weights]
    
    elif sentiment == 1:
        labels = [1 if x > threshold else 2 if x < -threshold else 0 for x in weights]

    elif sentiment == 0:
        labels = [0 for x in weights]
    
    return labels


def do_weighting(train_data, feature_names, rm):
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

    if type(train_data) == type(None):
        return None, None

    true_x_idx = []
    false_x_idx = []
    explanations = []

    for idx, inst in enumerate(train_data):
    
        bin_x = inst["bin"]
        y = inst["sentiment"]
    
        prediction = rm.predict(bin_x, explain=False)
        
        expl = rm.explain(bin_x, [0, 1, 2])
        
        if prediction == 2:
            expl  = expl[2] - (expl[1] + expl[0])
        
        if prediction == 1:
            expl = expl[1] - (expl[2] + expl[0])
        
        if prediction == 0:
            expl = np.array([0 for _ in range(len(feature_names))])

        explanations.append(expl)
        
        votes = rm._inference.get_votes()
        n_votes = votes[prediction]
        
        if y == prediction and n_votes > 0: 
            true_x_idx.append([n_votes, idx])
        
        else:
            false_x_idx.append(idx)
    
    true_x_idx = np.array(true_x_idx)
    
    percentile_25 = np.percentile(true_x_idx[:, 0], 25)

    is_75_percentile = np.where(true_x_idx[:, 0] >= percentile_25)[0]
    is_25_percentile = np.where(true_x_idx[:, 0] < percentile_25)[0]

    true_x = true_x_idx[is_75_percentile]
    true_x = list(true_x[:, 1:3])

    is_25_percentile = true_x_idx[is_25_percentile]
    is_25_percentile = list(is_25_percentile[:, 1])

    false_x_idx.extend(is_25_percentile)
    false_x = false_x_idx

    true_data = []
    false_data = []

    for idx, inst in enumerate(train_data):
        
        y = inst["sentiment"]
        tokens_x = inst["tokens"]
        orig_label = inst["orig_labels"]

        lemmas_x = inst["lemmas"]
        tokenmap_x = inst["token_ids"]
        bin_x = inst["bin"]
        
        orig_x = inst["orig_text"]

        if idx in true_x:

            expl = explanations[idx]
            
            vocabulary = {feature_names[i]: expl[i] for i in range(len(feature_names))}
        
            newtokens_x, weights_x = weight_tokens(lemmas_x, tokens_x, vocabulary, tokenmap_x)

            labels = label_tokens(y, weights_x)

            true_inst = {"tokens" : newtokens_x,
                     "weights" : weights_x,
                     "text" : orig_x,
                     "sentiment" : y,
                     "labels" : labels,
                     "orig_label" : orig_label}

            true_data.append(true_inst)
            
        elif idx in false_x:
            
            false_inst = {"sentiment" : y,
                           "lemmas" : lemmas_x,
                           "tokens" : tokens_x,
                           "token_ids" : tokenmap_x,
                           "orig_labels" : orig_label,
                           "orig_text" : orig_x}

            false_data.append(false_inst)
    
    return true_data, false_data

    
def make_weighted_data(train_data, error_params : bool = False):
    """
    Trains the Tsetlin Machine and weights the tokens in the data.

    Parameters:
    -----------
    train_data : list
        list of dicts with the data
    
    error_params : bool
        If True, the parameters for the Tsetlin Machine will be the error parameters.

    """

    train_x = [instance["lemmas"] for instance in train_data]
    train_y = [instance["sentiment"] for instance in train_data]
    
    rm, feature_names, train_x_bin = rulemaker(train_x=train_x, 
                                                train_y=train_y, 
                                                error_params=error_params)
    
    for i in range(len(train_data)):
        train_data[i]["bin"] = train_x_bin[i]
    
    train_data, train_error_data = do_weighting(train_data, feature_names, rm)
    
    return train_data, train_error_data


def write_chunk(data, output, n):
    
    file = open(os.path.join(output, f"train_val_chunk_{n}.pkl"), "wb")
    pickle.dump(data, file)


def write_error_chunk(data, output, n):
    
    file = open(os.path.join(output, f"train_val_chunk_{n}_e.pkl"), "wb")
    pickle.dump(data, file)


if __name__ == "__main__":
    print("Module")