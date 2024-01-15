import pickle
import os
import argparse
import re

import numpy as np
from tqdm import tqdm

import preprocessing.preprocess_functions as preprocess_functions
import preprocessing.preprocess_functions as pf
import arg_funcs as af
import config as config


def stage_preprocess(dataset:str, chunkdist_n : int):

    """
    Preprocesses data for training. Lemmatizes text, and maps tokens to ids.

    Parameters
    ----------
    dataset : str
        Name of dataset to stage chunks for.
    
    input : str
        Path to input data. Must be absolute path to directory.
    
    output : str
        Path to output of this module. Must be absolute path to directory.
    
    chunkdist_n : int
        ID of chunkdistribution. Must be an integer. Will be used to name the output directory, e.g "path/to/output/{dataset}_chunkdist_{chunkdist_n}".
    """

    root = config.root
    
    preprocess_folder = os.path.join(root, dataset, "preprocess")
    if not os.path.exists(preprocess_folder):
        os.mkdir(preprocess_folder)
    
    new_chunkdist = os.path.join(preprocess_folder, f"{dataset}_chunkdist_{chunkdist_n}")
    if not os.path.exists(new_chunkdist):
        os.mkdir(new_chunkdist)
    else:
        assert False, f"Directory {new_chunkdist} already exists, please remove it before continuing" 

    
    chunking_folder = os.path.join(root, dataset, "chunking")
    if not os.path.exists(chunking_folder):
        assert False, f"Chunking folder {chunking_folder} does not exist, please check your input"

    chunk_dist = os.path.join(chunking_folder, f"{dataset}_chunkdist_{chunkdist_n}")
    if not os.path.exists(chunk_dist):
        assert False, f"Chunk distribution {chunk_dist} does not exist, please check your input"

    chunk_dist_data = os.path.join(chunk_dist, "train")
    chunks = sorted(os.listdir(chunk_dist_data))

    for i, chunk in enumerate(tqdm(chunks)):
        
        chunk = os.path.join(chunk_dist_data, chunk)
        data = pickle.load(open(chunk, "rb"))

        raw_train_x = list(data["train_x"])
        print(f"chunk : {i} size : {len(raw_train_x)}")
        
        train_y = list(data["train_y"])
        orig_train_y = data["orig_train_y"]

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

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=str, help="Dataset to stage")
    parser.add_argument("--chunkdist_n", type=int, help="Which chunk to stage")

    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    
    stage_preprocess(args.dataset, args.chunkdist_n)
    