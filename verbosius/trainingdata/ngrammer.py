from collections import Counter
import numpy as np


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

    while j < len(grammies)-1: 
        
        curr_gram = grammies[j]
        next_gram = grammies[j+1] if j+1 < len(grammies) else None
        
        split_curr_gram = curr_gram.split(" ")
        split_next_gram = next_gram.split(" ") if next_gram else [-1]

        n_space_curr_gram = Counter(curr_gram)[" "]
        n_space_next_gram = Counter(next_gram)[" "] if next_gram else -1

        weight_curr_gram = weights[j]
        weight_next_gram = weights[j+1] if next_gram else -1

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
            
            elif n_space_curr_gram == 1 and n_space_next_gram == 0:
                grammies[j] = curr_gram
                weights[j] = weight_curr_gram + weight_next_gram
                grammies.pop(j+1)
                weights.pop(j+1)
            
            elif n_space_curr_gram == 0 and n_space_next_gram == 0:
                continue

            elif n_space_curr_gram == 0 and n_space_next_gram == 2:
                weights[j+1] += weight_curr_gram
                grammies.pop(j)
                weights.pop(j)
            
            elif n_space_curr_gram == 2 and n_space_next_gram == 0:
                weights[j] += weight_next_gram
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

        if trigram in vocabulary.keys():
            
            grammies.append(trigram)
            weights.append(vocabulary[trigram])
            is_tri = True
            skip_tri = 0
        
        elif bigram in vocabulary.keys():
            
            if skip_tri < 2 and is_tri:
                #weights[-1] += vocabulary[bigram]
                pass

            else:
                grammies.append(bigram)
                weights.append(vocabulary[bigram])
                is_bi = True
                skip_bi = 0

        elif unigram in vocabulary.keys():
            
            if skip_tri < 3 and is_tri:
                #weights[-1] += vocabulary[unigram]
                pass

            elif skip_bi < 2 and is_bi:
                weights[-1] += vocabulary[unigram]
                pass

            else:
                grammies.append(unigram)
                weights.append(vocabulary[unigram])
        
        if trigram not in vocabulary.keys() and bigram not in vocabulary.keys() and unigram not in vocabulary.keys():
            
            if skip_tri < 3 and is_tri:
                #continue
                pass
            
            elif skip_bi < 2 and is_bi:
                #continue
                pass
            
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

    print(f"grammies: {grammies}")
    print(f"tokens: {tokens}")

    print(" ".join(grammies) == " ".join(tokens))
    
    #print(f"g_ranges_start: {gram_ranges}")
    
    # print(f"tokens: {tokens}")
    # print(f"grammies: {grammies}")
    # print(f"gram_ranges: {gram_ranges}")
    # print(f"token_map: {token_map}")
    # print()

    new_grammies = []
            
    i = 0

    print(f"LENGTH TOKENS: {len(tokens)}")

    while True:
    
        if i == len(tokens):
            break

        current_token = tokens[i]
        range_id = find_gram_interval(i, gram_ranges)

        if i == 0:
            new_grammies.append(current_token)
            i += 1

        elif token_map[i-1] < token_map[i]:
            new_grammies.append(current_token)
            i += 1

        elif token_map[i-1] == token_map[i]:
            
            print(i)
            weights = move_weights(i, tokens, weights, gram_ranges)
            new_grammies[-1] += current_token
            tokens.pop(i)
            gram_ranges = update_gram_ranges(range_id, gram_ranges, weights)
            
            token_map.pop(i)

        

    new_grammies = [" ".join([new_grammies[i] for i in range(r[0], r[1] + 1)]) for r in gram_ranges]

    return new_grammies, weights


def move_weights(i, tokens, weights, gram_ranges):

    curr_range = find_gram_interval(i, gram_ranges)
    pre_range = find_gram_interval(i-1, gram_ranges)

    print(f"curr_range: {curr_range}, pre_range: {pre_range}")

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

    #if gram_ranges[range_id][0] > gram_ranges[range_id][1]:
    #    gram_ranges.pop(range_id)
    #    weights.pop(range_id)

    return gram_ranges


def find_gram_interval(i, gram_ranges):

    b = None
    for l, gr in enumerate(gram_ranges):

        if gr[0] <= i <= gr[1]:
            b = l
            return b
    
    if b is None:
        print(f"ERROR: {i} : {gram_ranges}")


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

            print(i)

            _, expl = rm.predict(x_b, explain=True)
            vocabulary = {feature_names[i]: expl[i] for i in range(len(feature_names))}

            # pickle vocabulary
            
            grammies, weights = find_grams(x_l, vocabulary)
            print("found grammies")
            grammies, weights = fit_grams(grammies, weights)
            print("fitted grammies")
            labels = label_grams(y, weights)
            print("labeled grammies")
            print(f"token_map: {token_map}")
            grammies, weights = convert_to_not_lemma(token_map, x_t, grammies, weights)
            print("converted from lemma to not lemma")

            new_x = {"tokens" : grammies,
                     "weights" : weights,
                     "text" : " ".join(grammies),
                     "sentiment" : y,
                     "labels" : labels}

            all_x.append(new_x)

    return all_x