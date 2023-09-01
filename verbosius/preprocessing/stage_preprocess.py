import pickle
import os
import argparse
import re

import numpy as np
from tqdm import tqdm

import preprocessing.preprocess_functions as preprocess_functions
import preprocessing.preprocess_functions as pf


def stage_preprocess(dataset:str, input:str, output:str, chunk_n : int):

    new_chunkdist = os.path.join(output, f"{dataset}_chunkdist_{chunk_n}")

    if not os.path.exists(new_chunkdist):
        os.mkdir(new_chunkdist)

    else:
        assert False, f"Directory {new_chunkdist} already exists, please remove it before continuing" 

    dist = os.listdir(input)
    dist = dist[chunk_n]

    path_to_dist = os.path.join(input, dist, "train_val") 
    chunks = sorted(os.listdir(path_to_dist))

    for i, chunk in enumerate(tqdm(chunks)):
        
        chunk = os.path.join(path_to_dist, chunk)
        data = pickle.load(open(chunk, "rb"))

        raw_train_x = data["train_x"]
        train_y = data["train_y"]
        orig_train_y = data["orig_train_y"]

        raw_val_x = data["val_x"]
        val_y = data["val_y"]
        orig_val_y = data["orig_val_y"]
        
        cleaned_train_x = preprocess_functions.clean_text(raw_train_x)
        cleaned_val_x = preprocess_functions.clean_text(raw_val_x)

        split_train_x, token_train_x, lemma_train_x = preprocess_functions.lemmatize(cleaned_train_x, lemmatizer="en_core_web_sm")
        split_val_x, token_val_x, lemma_val_x = preprocess_functions.lemmatize(cleaned_val_x, lemmatizer="en_core_web_sm") 

        token_ids_train_x = preprocess_functions.map_tokens(split_train_x, token_train_x)
        token_ids_val_x = preprocess_functions.map_tokens(split_val_x, token_val_x)

        train_data = pf.stage_data(token_x=token_train_x, 
                                      lemma_x=lemma_train_x, 
                                      token_ids_x=token_ids_train_x, 
                                      y=train_y, 
                                      orig_labels=orig_train_y,
                                      x=raw_train_x)
                
        val_data = pf.stage_data(token_x=token_val_x,
                                    lemma_x=lemma_val_x,
                                    token_ids_x=token_ids_val_x,
                                    y=val_y,
                                    orig_labels=orig_val_y,
                                    x=raw_val_x)

        train_val_data = {"train": train_data, "validation": val_data}

        pf.write_data(data=train_val_data, path=new_chunkdist, n=i, test=False)

    
def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon', 'mnist']
    if dataset.lower() not in valid_datasets:
        raise argparse.ArgumentTypeError(f"Invalid dataset, available datasets are: {(i for i in valid_datasets)}")
    return dataset.lower()


def input_checker(input):
    if os.access(os.path.dirname(input), os.W_OK) and os.path.isdir(input):
        return input
    else:
        raise argparse.ArgumentTypeError(f'Invalid input path, "{input}" is not writable or is not a directory')


def output_checker(output):
    if os.access(os.path.dirname(output), os.W_OK) and os.path.isdir(output):
        return output
    else:
        raise argparse.ArgumentTypeError(f'Invalid output path, "{output}" is not writable or is not a directory')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=dataset_checker, 
                        help="Dataset to stage")
    
    parser.add_argument("--input", type=input_checker, 
                        help="Path to input data, must be the absolute path to a valid directory where the datafiles are located.")
    
    parser.add_argument("--output", type=output_checker, 
                        help="Path to output data, must be a path to a directory that exists and is writable.")
    
    parser.add_argument("--chunk_n", type=int, help="Which chunk to stage")

    args = parser.parse_args()

    stage_preprocess(args.dataset, args.input, args.output, args.chunk_n)