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
    

    for j, gram in enumerate(grammies):
        
        next_gram = grammies[j+1] if j+1 < len(grammies) else None
        
        s_gram = gram.split()
        s_next_gram = next_gram.split() if next_gram else [-1]

        space_gram = Counter(gram)[" "]
        space_next_gram = Counter(next_gram)[" "] if next_gram else -1

        if space_gram == 1 and space_next_gram == 2 and s_gram[1] == s_next_gram[0]:    
            w = weights[j]/2
            weights[j] = w
            weights[j+1] += w
            grammies[j] = gram.split()[0]


        elif space_gram == 1 and space_next_gram == 1 and s_gram[1] == s_next_gram[0]:
            new_tri = "{} {} {}".format(gram.split()[0], gram.split()[1], next_gram.split()[2])
            grammies[j] = new_tri
            weights[j] = weights[j] + weights[j+1]
            
            weights.pop(j+1)
            grammies.pop(j+1)


        elif space_gram == 2 and space_next_gram == 1 and s_gram[-1] == s_next_gram[0]:
            w = weights[j+1]/2
            weights[j] += w
            weights[j+1] = w
            grammies[j+1] = next_gram.split()[1]


        elif space_gram == 2 and space_next_gram == 2 and s_gram[-1] == s_next_gram[0]:
            curr_tri = 1 if weights[j] >= weights[j+1] else 0 

            if curr_tri:
                w = weights[j+1]
                weights[j] += w * 1/3
                weights[j+1] = w * 2/3
                grammies[j+1] = " ".join(s_next_gram[1:])
            
            else:
                w = weights[j]
                weights[j] = w * 2/3
                weights[j+1] += w * 1/3
                grammies[j] = " ".join(s_gram[:-1])


        elif space_gram == 2 and space_next_gram == 0 and s_gram[-1] == s_next_gram[0]:
            weights[j] += weights[j+1]

            weights.pop(j+1)
            grammies.pop(j+1)
            

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
    is_uni = False
    
    skip = 0

    for i, curr_token in enumerate(tokens):
        
        if is_tri and skip < 3:
            skip += 1
    
        if skip == 3:
            skip = 0
            is_tri = False

        unigram = curr_token if i < len(tokens) else None
        bigram = "{} {}".format(curr_token, tokens[i+1]) if i+1 < len(tokens) else None
        trigram = "{} {} {}".format(curr_token, tokens[i+1], tokens[i+2]) if i+2 < len(tokens) else None
        
        if trigram in vocabulary.keys():
            
            grammies.append(trigram)
            weights.append(vocabulary[trigram])
            is_tri = True
            skip = 0
        
        elif bigram in vocabulary.keys():
            
            if skip < 2 and is_tri:
                weights[-1] += vocabulary[bigram]

            else:
                grammies.append(bigram)
                weights.append(vocabulary[bigram])
            
        elif unigram in vocabulary.keys():
            
            if skip < 3 and is_tri:
                weights[-1] += vocabulary[unigram]

            else:
                grammies.append(unigram)
                weights.append(vocabulary[unigram])
            
        elif trigram not in vocabulary.keys() and bigram not in vocabulary.keys() and unigram not in vocabulary.keys():
            
            if skip < 3 and is_tri:
                continue
            
            else:
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