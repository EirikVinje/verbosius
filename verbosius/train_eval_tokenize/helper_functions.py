import os
import gzip
import pickle

import numpy as np

import config as config


def set_directory(chunkdist_name):

    trainingdata_folder = os.path.join(config.root, "trainingdata")
    if not os.path.exists(trainingdata_folder):
        assert False, f"Trainingdata folder {trainingdata_folder} does not exist, please check your input"
    
    trainingdata_chunkdist = os.path.join(trainingdata_folder, chunkdist_name)
    if not os.path.exists(trainingdata_chunkdist):
        assert False, f"Chunk distribution {trainingdata_chunkdist} does not exist, please check your input"
    
    
    train_eval_tokenize_folder = os.path.join(config.root, "train_eval_tokenize")
    if not os.path.exists(train_eval_tokenize_folder):
        os.mkdir(train_eval_tokenize_folder)

    train_eval_tokenize_chunkdist = os.path.join(train_eval_tokenize_folder, chunkdist_name)
    if not os.path.exists(train_eval_tokenize_chunkdist):
        os.mkdir(train_eval_tokenize_chunkdist)
    else:
        os.system(f"rm -rf {train_eval_tokenize_chunkdist}")
        os.mkdir(train_eval_tokenize_chunkdist)

    train_folder = os.path.join(train_eval_tokenize_chunkdist, "train")
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    eval_folder = os.path.join(train_eval_tokenize_chunkdist, "eval")
    if not os.path.exists(eval_folder):
        os.mkdir(eval_folder)


def write_chunk(data, path, n):

    dir = os.path.join(path, f"chunk_{n}_.pkl")

    with gzip.open(dir, "wb") as f:
        pickle.dump(data, f)


def tokenize_and_align_labels(data, tokenizer, orig_labels : bool = True):
    
    if type(data) == type(None):
        return None

    if orig_labels:
        Y = np.array([(i['orig_label']) for i in data])
    else:
        Y = np.array([i['sentiment'] for i in data])

    tokenized_inputs = tokenizer([inst["tokens"] for inst in data], truncation=True, padding=False, is_split_into_words=True)

    labels = []
    targets = []
    
    for i, label in enumerate([inst["labels"] for inst in data]):
        
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
    
    tokenized_inputs["labels"] = labels
    tokenized_inputs["targets"] = targets

    output = {}
    output["input_ids"] = tokenized_inputs["input_ids"] 
    output["attention_mask"] = tokenized_inputs["attention_mask"]
    output["labels"] = tokenized_inputs["labels"]
    output["targets"] = tokenized_inputs["targets"]
    output["sentiment"] = Y

    # convert to list of dicts
    temp = []
    for i in range(len(output["input_ids"])):
        temp.append(
            {
                "input_ids": output["input_ids"][i],
                "attention_mask": output["attention_mask"][i],
                "labels": output["labels"][i],
                "targets": output["targets"][i],
                "sentiment": output["sentiment"][i]
            }
        )    

    output = temp

    return output


