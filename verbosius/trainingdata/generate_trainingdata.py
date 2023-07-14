from collections import Counter
from copy import deepcopy
import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch


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

    tm.set_train_data(train_x_bin, train_y)
    tm.set_test_data(test_x_bin, test_y)
    trainer = gt.Trainer(config.T, n_epochs=config.TM_EPOCHS, seed=32, n_jobs=6, early_exit_acc=config.EARLY_STOP_ACC)

    trainer.train(tm)    

    rp = gt.RulePredictor()
    fm = list(range(train_x_bin.shape[1]))
    rp.create_from_state(tm.get_state(), fm)
    
    return rp, feature_names
    

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

    # print(f"grammies: {grammies}")
    # print(f"weights: {weights}")
    
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
                grammies.append(unigram)
                weights.append(vocabulary[unigram])
        
        if trigram not in vocabulary.keys() and bigram not in vocabulary.keys() and unigram not in vocabulary.keys():
            
            if skip_tri < 3 and is_tri:
                continue
            
            elif skip_bi < 2 and is_bi:
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


def convert_to_not_lemma(token_map, tokens, grammies, weights):

    gram_ranges = gram_map(grammies)

    # print(f"tokens: {tokens}")
    # print(f"grammies: {grammies}")
    # print(f"gram_ranges: {gram_ranges}")
    # print(f"token_map: {token_map}")
    # print()

    new_grammies = []
            
    i = 0
    while True:
        
        current_token = tokens[i]
        range_id = find_gram_interval(i, gram_ranges)

        if i == 0:
            new_grammies.append(current_token)
            i += 1

        elif token_map[i-1] < token_map[i]:
            new_grammies.append(current_token)
            i += 1

        elif token_map[i-1] == token_map[i]:
            
            weights = move_weights(i, tokens, weights, gram_ranges)
            new_grammies[-1] += current_token
            tokens.pop(i)
            gram_ranges = update_gram_ranges(range_id, gram_ranges, weights)
            
            token_map.pop(i)

        if i == len(tokens):
            break

    new_grammies = [" ".join([new_grammies[i] for i in range(r[0], r[1] + 1)]) for r in gram_ranges]

    return new_grammies, weights


def move_weights(i, tokens, weights, gram_ranges):

    curr_range = find_gram_interval(i, gram_ranges)
    pre_range = find_gram_interval(i-1, gram_ranges)

    if curr_range != pre_range:
        
        ngram = Counter(" ".join(tokens[gram_ranges[curr_range][0]:gram_ranges[curr_range][1]+1]))[" "]
        
        if gram_ranges[curr_range][0] > gram_ranges[curr_range][1]:
            return weights

        if ngram == 2:
            temp_weight = weights[curr_range]
            weights[pre_range] += temp_weight * 1/3
            weights[curr_range] = temp_weight * 2/3
            
        elif ngram == 1:
            temp_weight = weights[curr_range]
            weights[pre_range] += temp_weight * 1/2
            weights[curr_range] = temp_weight * 1/2
        
        else:
            temp_weight = weights[curr_range]
            weights[pre_range] += temp_weight
            weights[curr_range] = 0.0

    return weights


def update_gram_ranges(range_id, gram_ranges, weights):

    for i in range(range_id, len(gram_ranges)):

        gram_ranges[i][0] -= 1
        gram_ranges[i][1] -= 1

    gram_ranges[range_id][0] += 1

    if gram_ranges[range_id][0] > gram_ranges[range_id][1]:
        gram_ranges.pop(range_id)
        weights.pop(range_id)

    return gram_ranges


def find_gram_interval(i, gram_ranges):

    for l, gr in enumerate(gram_ranges):

        if gr[0] <= i <= gr[1]:
            return l


def gram_map(grammies):
    
    map_gram = 0
    gram_ranges = []
    head = 0
    tail = 0
    
    for j, gram in enumerate(grammies):
        
        if Counter(gram)[" "] == 2:
            map_gram += 3

        elif Counter(gram)[" "] == 1:
            map_gram += 2
        
        else:
            map_gram += 1

        tail = map_gram - 1
        gram_ranges.append([head, tail])
        head = tail + 1

    return gram_ranges 

    
def weight_texts(data, feature_names, testing = False, rm = None):

    if testing:
        
        y = data["label"]
        x_b = data["bin"]
        x_l = data["lemmas"]
        x_t = data["tokens"]
        token_map = data["token_ids"]
    
        vocabulary = {feature_names[i] : 1.0 for i in range(len(feature_names))}

        grammies, weights = find_grams(x_l, vocabulary)
        grammies, weights = fit_grams(grammies, weights)
        labels = label_grams(y, weights)
        grammies, weights = convert_to_not_lemma(token_map, x_t, grammies, weights)

        new_x = {"tokens" : grammies,
                    "weights" : weights,
                    "text" : " ".join(grammies),
                    "sentiment" : y,
                    "labels" : labels}
        
        return new_x

    
    all_x = []

    for i, inst in enumerate(data):
        
        y = inst["label"]
        x_b = inst["bin"]
        x_l = inst["lemmas"]
        x_t = inst["tokens"]
        token_map = inst["token_ids"]

        if y == rm.predict(x_b):
            
            _, expl = rm.predict(x_b, explain=True)
            vocabulary = {feature_names[i]: expl[i] for i in range(len(feature_names))}

            grammies, weights = find_grams(x_l, vocabulary)
            grammies, weights = fit_grams(grammies, weights)
            labels = label_grams(y, weights)
            grammies, weights = convert_to_not_lemma(token_map, x_t, grammies, weights)

            new_x = {"tokens" : grammies,
                     "weights" : weights,
                     "text" : " ".join(grammies),
                     "sentiment" : y,
                     "labels" : labels}

            all_x.append(new_x)

    return all_x

def tokenize_and_align_labels(data, tokenizer, device):
    
    Y = np.array([i['label'] for i in data])

    examples = data

    tokenized_inputs = tokenizer([example["tokens"] for example in examples], 
                                 truncation=True, 
                                 padding=True, 
                                 return_tensors='pt',
                                 is_split_into_words=True,
                                 max_length=512
                                 )
    
    labels = []
    targets = []
    
    for i, label in enumerate([example["labels"] for example in examples]):
        
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        
        previous_word_idx = None

        label_ids = []
        target_ids = []
        
        
        for word_idx in word_ids:  

            if word_idx is None:
                target_ids.append(0)
                label_ids.append(-100)

            elif word_idx != previous_word_idx:  
                target_ids.append(1)
                label_ids.append(label[word_idx])

            else:
                target_ids.append(0)
                label_ids.append(-100)

            previous_word_idx = word_idx
        targets.append(target_ids)
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = np.array(labels,dtype=np.int8)
    
    tokenized_inputs["targets"] = np.array(targets,dtype=np.int8)

    output = {}
    output["input_ids"] = tokenized_inputs["input_ids"].to(device = device) 
    output["attention_mask"] = torch.tensor(tokenized_inputs["attention_mask"], dtype=torch.int8).to(device = device)
    output["labels"] = torch.tensor(tokenized_inputs["labels"], dtype=torch.int8).to(device = device)
    output["targets"] = torch.tensor(tokenized_inputs["targets"], dtype=torch.int8).to(device = device)
    output["sentiment"] = torch.tensor(Y, dtype=torch.int8).to(device = device)
    # print("SENTIMENT SIZE", len(output['sentiment']))
    return output

class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels, targets, sentiment):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.targets = targets
        self.sentiment = sentiment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = self.labels[idx]
        targets = self.targets[idx]
        sentiment = self.sentiment[idx]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'targets': targets,
            'sentiment': sentiment
        }


if __name__ == "__main__":
    print("Module")