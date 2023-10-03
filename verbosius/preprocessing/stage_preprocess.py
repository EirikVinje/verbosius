import pickle
import os
import argparse
import re

import numpy as np
from tqdm import tqdm

import preprocessing.preprocess_functions as preprocess_functions
import preprocessing.preprocess_functions as pf


def stage_preprocess(dataset:str, input:str, output:str, chunkdist_n : int):

    """
    Preprocesses data for training.

    Parameters
    ----------
    dataset : str
        Name of dataset to stage chunks for.
    
    input : str
        Path to input data. Must be absolute path to directory.
    
    output : str
        Path to output of this module. Must be absolute path to directory.
    
    chunkdist_n : int
        ID of chunkdistribution. Must be an integer.
    """

    new_chunkdist = os.path.join(output, f"{dataset}_chunkdist_{chunkdist_n}")

    if not os.path.exists(new_chunkdist):
        os.mkdir(new_chunkdist)
    else:
        assert False, f"Directory {new_chunkdist} already exists, please remove it before continuing" 

    chunk_dist = os.path.join(input, f"{dataset}_chunkdist_{chunkdist_n}", "train_val")
    chunks = sorted(os.listdir(chunk_dist))

    for i, chunk in enumerate(tqdm(chunks)):
        
        chunk = os.path.join(chunk_dist, chunk)
        data = pickle.load(open(chunk, "rb"))

        raw_train_x = list(data["train_x"])
        train_y = list(data["train_y"])
        orig_train_y = data["orig_train_y"]

        raw_train_x.extend(list(data["val_x"]))
        train_y.extend(list(data["val_y"]))
        
        cleaned_train_x = preprocess_functions.clean_text(raw_train_x)

        split_train_x, token_train_x, lemma_train_x = preprocess_functions.lemmatize(cleaned_train_x, lemmatizer="en_core_web_sm")

        token_ids_train_x = preprocess_functions.map_tokens(split_train_x, token_train_x)

        train_data = pf.stage_data(token_x=token_train_x, 
                                      lemma_x=lemma_train_x, 
                                      token_ids_x=token_ids_train_x, 
                                      y=train_y, 
                                      orig_labels=orig_train_y,
                                      x=raw_train_x)
                

        pf.write_data(data=train_data, path=new_chunkdist, n=i)

    
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


def chunkdist_checker(dataset, input, chunkdist_n):

    if not os.path.exists(os.path.join(input, f"{dataset}_chunkdist_{chunkdist_n}")):
        raise argparse.ArgumentTypeError(f"Invalid chunk dist, {dataset}_chunkdist_{chunkdist_n} does not exist") 

    return chunkdist_n

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=str, help="Dataset to stage")
    parser.add_argument("--input", type=str, help="Path to input data, must be the absolute path to a valid directory where the datafiles are located.")
    parser.add_argument("--output", type=str, help="Path to output data, must be a path to a directory that exists and is writable.")
    parser.add_argument("--chunkdist_n", type=int, help="Which chunk to stage")

    args = parser.parse_args()

    dataset_checker(args.dataset)
    input_checker(args.input)
    output_checker(args.output)
    chunkdist_checker(args.dataset, args.input, args.chunkdist_n)

    stage_preprocess(args.dataset, args.input, args.output, args.chunkdist_n)
    # stage_preprocess(dataset:str, input:str, output:str, chunkdist_n : int)