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
    

def allign_tokens_labels_weights_bigram(tokens, vocab_weights, sentiment, threshold):
    
    alligned_tokens = []
    alligned_weights = []

    for i in range(1, len(tokens) + 1):

        bigram = tokens[i-1] + " " + tokens[i] if i < len(tokens) else None
        unigram = tokens[i-1]

        if bigram in vocab_weights.keys():
            alligned_tokens.append(bigram)
            alligned_weights.append(vocab_weights[bigram])

        elif unigram in vocab_weights.keys():
            alligned_tokens.append(unigram)
            alligned_weights.append(vocab_weights[unigram])
        
        else:
            alligned_tokens.append(unigram)
            alligned_weights.append(0.0)


    for j in range(1, len(alligned_tokens)-1):
        
        if " " in alligned_tokens[j] and " " not in alligned_tokens[j-1] and alligned_tokens[j-1] in alligned_tokens[j]:
            alligned_tokens[j-1] = "#"
            weight = alligned_weights[j-1]
            alligned_weights[j-1] = "#"
            alligned_weights[j] += weight
            
        elif " " in alligned_tokens[j] and " " in alligned_tokens[j-1] and alligned_tokens[j-1].split(" ")[1] in alligned_tokens[j]:
            alligned_tokens[j-1] = alligned_tokens[j-1].split(" ")[0]
                
        if alligned_tokens[j] in alligned_tokens[j-1]:
            alligned_tokens[j] = "#"
            weight = alligned_weights[j-1]
            alligned_weights[j] = "#"
            alligned_weights[j-1] += weight
    
    alligned_tokens = [token for token in alligned_tokens if token != "#"]
    alligned_weights = [weight for weight in alligned_weights if weight != "#"]

    if sentiment == 1:
        alligned_labels = [2 if x > threshold else 1 if x < -threshold else 0 for x in alligned_weights]
    else:
        alligned_labels = [1 if x > threshold else 2 if x < -threshold else 0 for x in alligned_weights]
    
    return alligned_tokens, alligned_weights, alligned_labels


def allign_tokens_labels_weights_trigram(tokens, vocab_weights, sentiment, threshold : float = 0.0):
    
    """
    Parameters:
    -----------
    tokens : list : lemmas
    vocab_weights : dict : {ngram : weight}
    sentiment : int : class label
    
    Returns:
    --------
    alligned_tokens : list : alligned lemma tokens

    """


    alligned_tokens = []
    alligned_weights = []

    is_tri = False
    is_bi = False
    is_uni = False
    
    count = 0

    for i in range(0, len(tokens)):
        
        if is_tri and count < 2:
            count += 1
            continue
        
        elif is_tri:
            count = 0
            is_tri = False
            
        if is_bi and count < 1:
            count += 1
            continue
        
        elif is_bi:
            count = 0
            is_bi = False

        is_uni = False
        is_bi = False
        is_tri = False

        unigram = tokens[i] if i < len(tokens) else None
        bigram = tokens[i] + " " + tokens[i+1] if i+1 < len(tokens) else None
        trigram = tokens[i] + " " + tokens[i+1] + " " + tokens[i+2] if i+2 < len(tokens) else None

        if unigram in vocab_weights.keys():
            alligned_tokens.append(unigram)
            alligned_weights.append(vocab_weights[unigram])
            is_uni = True

        if bigram in vocab_weights.keys():
            if is_uni:
                alligned_tokens.pop()
                alligned_weights.pop()

            alligned_tokens.append(bigram)
            alligned_weights.append(vocab_weights[bigram])
            is_bi = True
            
        if trigram in vocab_weights.keys():
            if is_bi:
                alligned_tokens.pop()
                alligned_weights.pop()
            
            alligned_tokens.append(trigram)
            alligned_weights.append(vocab_weights[trigram])
            is_tri = True
            
        if trigram not in vocab_weights.keys() and bigram not in vocab_weights.keys() and unigram not in vocab_weights.keys():
            alligned_tokens.append(unigram)
            alligned_weights.append(0.0)

    if sentiment == 1:
        alligned_labels = [2 if x > threshold else 1 if x < -threshold else 0 for x in alligned_weights]
    else:
        alligned_labels = [1 if x > threshold else 2 if x < -threshold else 0 for x in alligned_weights]
    
    return alligned_tokens, alligned_weights, alligned_labels


def do_allign_tokens_labels_weights(rm, data_x, data_bin_x, data_y):

    for (x, text, y) in zip(data_bin_x, data_x, data_y):
        
        if y == rm.predict(x):
            
            _, expl = rm.predict(x, explain=True)
            
            #


if __name__ == "__main__":

    tokens = ["i", "was", "not", "happy", "with", "the", "movie", "it", "was", "bad"]

    # make vocabulary with unigrams, bigrams and trigrams
    vocabulary = {
        "i" : 0.1,
        "not happy" : -0.7,
        "happy with" : 0.4,
        "with the movie" : 0.5,
        "the movie" : 0.6,
        "it" : 0.7,
        "was bad" : 0.8,
    }

    sentiment = 1
    threshold = 0.3

    alligned_tokens, alligned_weights, alligned_labels = allign_tokens_labels_weights_trigram(tokens, vocabulary, sentiment, threshold)

    print(alligned_tokens)
    print(alligned_weights)
    print(alligned_labels)