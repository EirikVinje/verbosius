#import green_tsetlin as gt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


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




