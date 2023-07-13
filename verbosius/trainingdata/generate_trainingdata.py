from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import green_tsetlin as gt
import config as config


def rulemaker(data):
    
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
                                 stop_words = 'english')
    
    train_x_bin = vectorizer.fit_transform([" ".join(x) for x in train_x])
    test_x_bin = vectorizer.transform([" ".join(x) for x in test_x])
    vocabulary = vectorizer.get_feature_names_out()

    tm = gt.TsetlinMachine(n_literals=train_x_bin.shape[1], 
                           n_clauses=config.NUMBER_OF_CLAUSES, 
                           n_classes=data["n_classes"],
                           s=config.S, 
                           n_literal_budget=config.LITERAL_BUDGET)

    train_x_bin = train_x_bin.todense()
    test_x_bin = test_x_bin.todense()

    tm.set_train_data(train_x_bin, train_y)
    tm.set_test_data(test_x_bin, test_y)
    trainer = gt.Trainer(config.T, n_epochs=config.TM_EPOCHS, seed=32, n_jobs=6, early_exit_acc=0.84)

    trainer.train(tm)    

    rp = gt.RulePredictor()
    fm = list(range(train_x_bin.shape[1]))
    rp.create_from_state(tm.get_state(), fm)
    
    return rp
    

def fit_grams(grammies, weights):
    """
    Parameters:
    -----------
    grammies : list
        list of n-grams
    weights : list
        list of weights for each n-gram
    
    Returns:
    --------
    grammies : list
        list of n-grams
    weights : list
        list of weights for each n-gram
    """

    print(f"grammies: {grammies}")
    print(f"weights: {weights}")
    
    j = 0
    i = 0
    while j < len(grammies)-1: 
        
        curr_gram = grammies[j]
        next_gram = grammies[j+1] if j+1 < len(grammies) else None
        
        split_curr_gram = curr_gram.split(" ")
        split_next_gram = next_gram.split(" ") if next_gram else [-1]

        n_space_curr_gram = Counter(curr_gram)[" "]
        n_space_next_gram = Counter(next_gram)[" "] if next_gram else -1

        weight_curr_gram = weights[j]
        weight_next_gram = weights[j+1] if next_gram else -1

        #print(f"curr_gram: {curr_gram}, next_gram: {next_gram}")

        if split_curr_gram[-1] == split_next_gram[0] and next_gram is not None:
            
            if n_space_curr_gram == 1 and n_space_next_gram == 1:
                
                new_tri = "{} {} {}".format(split_curr_gram[0], split_curr_gram[1], split_next_gram[1])
                grammies[j] = new_tri
                grammies.pop(j+1)

                weights[j] = weight_curr_gram + weight_next_gram
                weights.pop(j+1)

            elif n_space_curr_gram == 1 and n_space_next_gram == 2:
                
                grammies[j] = split_curr_gram[0]

                weights[j] = weight_curr_gram/2
                weights[j+1] += weight_curr_gram/2

                j += 1

            elif n_space_curr_gram == 2 and n_space_next_gram == 1:

                grammies[j+1] = split_next_gram[1]
                
                weights[j+1] = weight_next_gram/2
                weights[j] += weight_next_gram/2

                j += 1

            elif n_space_curr_gram == 2 and n_space_next_gram == 2:

                grammies[j] = " ".join(split_curr_gram[:-1])
                
                weights[j] = weight_curr_gram * 1/3
                weights[j+1] += weight_curr_gram * 2/3

                j += 1
            
            elif n_space_curr_gram == 0 and n_space_next_gram == 1:
                
                grammies[j] = next_gram
                weights[j] = weight_curr_gram + weight_next_gram                
                
                grammies.pop(j+1)
                weights.pop(j+1)
        
        elif n_space_curr_gram == 2 and n_space_next_gram == 2 and split_curr_gram[1:] == split_next_gram[:-1]:
            
            grammies[j] = split_curr_gram[0]
            
            weights[j] = weight_curr_gram * 1/3
            weights[j+1] += weight_curr_gram * 2/3
        
            j += 1
        
        
        else:
            j += 1

    return grammies, weights


def find_grams(tokens, vocabulary):

    """
    Parameters:
    -----------
    tokens : list
        list of tokens

    vocabulary : dict
        dictionary of n-grams and their weights
    
    Returns:
    --------
    grammies : list
        list of n-grams
    weights : list
        list of weights for each n-gram
    """

    grammies = []
    weights = []

    is_tri = False
    is_bi = False

    skip_tri = 0
    skip_bi = 0

    for i, curr_token in enumerate(tokens):
        
        #print()

        if is_tri and skip_tri < 3:
            skip_tri += 1
    
        elif skip_tri == 3:
            skip_tri = 0
            is_tri = False

        if is_bi and skip_bi < 2:
            skip_bi += 1
        
        elif skip_bi == 2:
            skip_bi = 0
            is_bi = False

        unigram = curr_token if i < len(tokens) else None
        bigram = "{} {}".format(curr_token, tokens[i+1]) if i+1 < len(tokens) else None
        trigram = "{} {} {}".format(curr_token, tokens[i+1], tokens[i+2]) if i+2 < len(tokens) else None

        #print(f"unigram: {unigram}, bigram: {bigram}, trigram: {trigram}")
        #print(f"skip_tri: {skip_tri}, skip_bi: {skip_bi}")
        #print(f"is_tri: {is_tri}, is_bi: {is_bi}")

        if trigram in vocabulary.keys():
            
            #print(f"adding trigram: {trigram}")
            grammies.append(trigram)
            weights.append(vocabulary[trigram])
            is_tri = True
            skip_tri = 0
        
        if bigram in vocabulary.keys():
            
            if skip_tri < 2 and is_tri:
                weights[-1] += vocabulary[bigram]

            else:
                #print(f"adding bigram: {bigram}")
                grammies.append(bigram)
                weights.append(vocabulary[bigram])
                is_bi = True
                skip_bi = 0

        if unigram in vocabulary.keys():
            
            if skip_tri < 3 and is_tri:
                weights[-1] += vocabulary[unigram]
            
            elif skip_bi < 2 and is_bi:
                weights[-1] += vocabulary[unigram]

            else:
                #print(f"adding unigram: {unigram}")
                grammies.append(unigram)
                weights.append(vocabulary[unigram])
        
        if trigram not in vocabulary.keys() and bigram not in vocabulary.keys() and unigram not in vocabulary.keys():
            
            if skip_tri < 3 and is_tri:
                continue
            
            elif skip_bi < 2 and is_bi:
                continue
            
            else:
                #print(f"adding unknown unigram: {unigram}")
                grammies.append(unigram)
                weights.append(0.0)

    return grammies, weights


def label_grams(sentiment, weights, threshold : float = 0.0):

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


def do_grams(rm, lemma_x, data_bin_x, data_y, feature_names):

    all_x = []

    for (x, lemmas, y) in zip(data_bin_x, lemma_x, data_y):
        
        if y == rm.predict(x):
            
            _, expl = rm.predict(x, explain=True)
            vocabulary = {feature_names[i]: expl[i] for i in range(len(feature_names))}

            grammies, weights = find_grams(lemmas, vocabulary)
            grammies, weights = fit_grams(grammies, weights)
            labels = label_grams(y, weights)

            temp = {"lemmas": lemmas, "grammies": grammies, "weights": weights, "labels": labels}
            all_x.append(temp)
            

if __name__ == "__main__":

    print("Module")