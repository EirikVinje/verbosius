#import green_tsetlin as gt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from config import Parameters


def Rulemaker(train_lemma_text, test_lemma_text, train_y, test_y, n_classes):
    
    p = Parameters()

    vectorizer = CountVectorizer(max_features=p.MAX_FEATURES,
                                 max_df=p.MAX_DF, 
                                 min_df=p.MIN_DF,
                                 ngram_range=p.N_GRAM_RANGE,
                                 binary=True,
                                 dtype=np.uint8,
                                 stop_words = 'english')
    
    train_x_bin = vectorizer.fit_transform(train_lemma_text)
    test_x_bin = vectorizer.transform(test_lemma_text)
    vocabulary = vectorizer.get_feature_names_out()

    tm = gt.TsetlinMachine(n_literals=train_x_bin.shape[1], 
                           n_clauses=p.NUMBER_OF_CLAUSES, 
                           n_classes=n_classes,
                           s=p.S, 
                           n_literal_budget=p.LITERAL_BUDGET)

    train_x_bin = train_x_bin.todense()
    test_x_bin = test_x_bin.todense()
    
    tm.set_train_data(train_x_bin, train_y)
    tm.set_test_data(test_x_bin, test_y)
    trainer = gt.Trainer(p.T, n_epochs=p.TM_EPOCHS, seed=32, n_jobs=6, early_exit_acc=0.84)

    r = trainer.train(tm)    

    rp = gt.RulePredictor()
    fm = list(range(train_x_bin.shape[1]))
    rp.create_from_state(tm.get_state(), fm)
    
    return rp
    

def allign_tokens_labels_weights(tokens, vocab_weights, sentiment, threshold):
    
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


def allign_tokens_labels_weights_trigram(tokens, vocab_weights, sentiment, threshold):
    
    alligned_tokens = []
    alligned_weights = []

    for i in range(1, len(tokens) + 1):

        bigram = tokens[i-1] + " " + tokens[i] if i < len(tokens) else None
        unigram = tokens[i-1]
        trigram = tokens[i-1] + " " + tokens[i] + " " + tokens[i+1] if i < len(tokens) - 1 else None

        if bigram in vocab_weights.keys():
            alligned_tokens.append(bigram)
            alligned_weights.append(vocab_weights[bigram])

        elif unigram in vocab_weights.keys():
            alligned_tokens.append(unigram)
            alligned_weights.append(vocab_weights[unigram])

        elif trigram in vocab_weights.keys():
            alligned_tokens.append(trigram)
            alligned_weights.append(vocab_weights[trigram])
        
        else:
            alligned_tokens.append(unigram)
            alligned_weights.append(0.0)

    print(alligned_tokens)

    for j in range(1, len(alligned_tokens)-1):

        token_0 = alligned_tokens[j-1]
        token_1 = alligned_tokens[j]
        token_2 = alligned_tokens[j+1]
        
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


if __name__ == "__main__":

    tokens = ["i", "was", "not", "happy", "with", "the", "movie", "it", "was", "bad"]

    # make vocabulary with unigrams, bigrams and trigrams
    vocabulary = {
        "i" : 0.1,
        "not happy" : 0.2,
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