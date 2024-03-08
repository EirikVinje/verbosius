import os
import pickle 
import argparse
import gc
import gzip
import shutil

from tqdm import tqdm

from trainingdata.generate_trainingdata import make_weighted_data, write_train_chunk, write_error_chunk, set_directory, make_eval
from xai_transformer.helper_functions import tokenize_and_align_labels
import trainingdata.generate_trainingdata as gen_data
import config as config
import arg_funcs as af


def stage_trainingdata(dataset : str, chunkdist_n : int):

    """
    Stage trainingdata for transformer.
    Uses TM to weight individual tokens in a sequence of text.
    Prepares trainingdata for transformer by tokenizing and aligning labels.    

    Parameters
    ----------
    dataset : str
        Name of dataset to stage trainingdata for.

    chunkdist_n : int
        ID (int) of chunkdist to use for trainingdata. Must be an integer. Will be used to name the output directory, e.g "path/to/output/{dataset}_chunkdist_{chunkdist_n}".
    """

    os.system(f"rm -rf /home/{config.user}/data/verbosius/amazon/preprocess/amazon_chunkdist_{chunkdist_n}/*e.pkl")

    chunkdist_name = f"{dataset}_chunkdist_{chunkdist_n}"

    set_directory(chunkdist_name)

    preprocess_path = os.path.join(config.root, "preprocess", chunkdist_name)
    trainingdata_path = os.path.join(config.root, "trainingdata", chunkdist_name)

    train_path = os.path.join(trainingdata_path, "train")
    eval_path = os.path.join(trainingdata_path, "eval")

    total_size = 0
    dir_len = len(os.listdir(preprocess_path)) * 2
    
    for n in tqdm(range(dir_len), desc="Staging trainingdata"):
        
        dir = sorted(os.listdir(preprocess_path), key=lambda x: int(x.split("_")[2]))

        if n > len(dir):
            assert False, "Something went wrong here."

        chunk = dir[n]
        chunk = os.path.join(preprocess_path, chunk)

        with gzip.open(chunk, "rb") as f:
            train = pickle.load(f)

        if chunk[-5] != "e":
            
            train, train_error = make_weighted_data(train, error_params=False)
            tokenized_train = tokenize_and_align_labels(train, config.tokenizer, orig_labels=True)
            write_train_chunk(tokenized_train, train_path, n)
            write_error_chunk(train_error, preprocess_path, n)
            total_size += len(tokenized_train)
            
        elif chunk[-5] == "e":

            train, _ = make_weighted_data(train, error_params=True)
            tokenized_train = tokenize_and_align_labels(train, config.tokenizer, orig_labels=True)
            write_train_chunk(tokenized_train, train_path, n)
            total_size += len(tokenized_train)
            
        train = None
        train_error = None
        gc.collect()
    
    
    make_eval(eval_path, train_path, total_size)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=str, help="Dataset to make trainingdata")
    parser.add_argument("--chunkdist_n", type=int, help="Set size for individual batch, must be greater than 0. Default value is 10000")

    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunckdist_n_checker(args.chunkdist_n)
    
    stage_trainingdata(args.dataset, args.chunkdist_n)
    