import pickle
import os
import argparse
import gzip
import warnings

from tqdm import tqdm

from preprocess.preprocess_functions import clean_text, lemmatize, map_tokens, stage_data, write_data, set_directory
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

    chunkdist_name = f"{dataset}_chunkdist_{chunkdist_n}"

    set_directory(chunkdist_name)    

    preprocess_path = os.path.join(config.root, "preprocess", chunkdist_name)
    chunking_path = os.path.join(config.root, "chunking", chunkdist_name)

    chunk_dist_data = os.path.join(chunking_path, "train")
    chunks = sorted(os.listdir(chunk_dist_data), key=lambda x: int(x.split("_")[2]))

    for i, chunk in enumerate(tqdm(chunks, desc="preprocessing chunks")):
        
        chunk = os.path.join(chunk_dist_data, chunk)
        
        with gzip.open(chunk, "rb") as f:
            data = pickle.load(f)
        
        if data is None:
            assert False, "Data is None, please check your input."

        raw_train_x = list(data["train_x"])
        
        train_y = list(data["train_y"])
        orig_train_y = data["orig_train_y"]

        cleaned_train_x = clean_text(raw_train_x)

        split_train_x, token_train_x, lemma_train_x = lemmatize(cleaned_train_x, lemmatizer="en_core_web_sm")

        token_ids_train_x = map_tokens(split_train_x, token_train_x)

        train_data = stage_data(token_x=token_train_x, 
                                    lemma_x=lemma_train_x, 
                                    token_ids_x=token_ids_train_x, 
                                    y=train_y, 
                                    orig_labels=orig_train_y,
                                    x=raw_train_x)
                

        write_data(train_data, preprocess_path, i)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=str, help="Dataset to stage")
    parser.add_argument("--chunkdist_n", type=int, help="Which chunk to stage")

    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    
    stage_preprocess(args.dataset, args.chunkdist_n)
    