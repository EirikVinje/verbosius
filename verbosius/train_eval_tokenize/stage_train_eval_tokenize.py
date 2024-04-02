import os
import pickle 
import argparse
import gc
import gzip
import shutil

from tqdm import tqdm
import numpy as np

from train_eval_tokenize.helper_functions import set_directory, tokenize_and_align_labels, write_chunk
import utils.config as config
import utils.arg_funcs as af


def stage_train_eval_tokenize(dataset : str, chunkdist_n : int):

    """
    Parameters
    ----------
    dataset : str
        Name of dataset to stage trainingdata for.

    chunkdist_n : int
        ID (int) of chunkdist to use for trainingdata. Must be an integer. Will be used to name the output directory, e.g "path/to/output/{dataset}_chunkdist_{chunkdist_n}".
    """
    
    chunkdist_name = f"{dataset}_chunkdist_{chunkdist_n}"

    set_directory(chunkdist_name)

    trainingdata_path = os.path.join(config.root, "trainingdata", chunkdist_name)
    train_path = os.path.join(config.root, "train_eval_tokenize", chunkdist_name, "train")
    eval_path = os.path.join(config.root, "train_eval_tokenize", chunkdist_name, "eval")

    chunks = os.listdir(trainingdata_path)

    temp = []
    with tqdm(total=len(chunks), desc="Finding class balance", disable=True) as pbar:
        
        for chunk in chunks:
            
            with gzip.open(os.path.join(trainingdata_path, chunk), "rb") as f:
                data = pickle.load(f)
            
            temp.extend([data[i]["sentiment"] for i in range(len(data))])

            pbar.update(1)
    
    class_balance = np.unique(temp, return_counts=True)[1]
    balance_train = [list(class_balance * 0.8), [0, 0, 0]]
    balance_eval = [list(class_balance * 0.2), [0, 0, 0]]

    max_chunk = 8000
    train = []
    eval = []

    n_e = 0
    n_t = 0

    with tqdm(total=len(chunks), desc="Making tokenized train and eval", disable=False) as pbar:
        
        for chunk in chunks:
            
            with gzip.open(os.path.join(trainingdata_path, chunk), "rb") as f:
                data = pickle.load(f)
            
            for sample in data:

                if sample["sentiment"] == 0 and balance_train[1][0] < balance_train[0][0]:
                    balance_train[1][0] += 1
                    train.append(sample)
                elif sample["sentiment"] == 1 and balance_train[1][1] < balance_train[0][1]:
                    balance_train[1][1] += 1
                    train.append(sample)
                elif sample["sentiment"] == 2 and balance_train[1][2] < balance_train[0][2]:
                    balance_train[1][2] += 1
                    train.append(sample)

                if sample["sentiment"] == 0 and balance_eval[1][0] < balance_eval[0][0]:
                    balance_eval[1][0] += 1
                    eval.append(sample)
                elif sample["sentiment"] == 1 and balance_eval[1][1] < balance_eval[0][1]:
                    balance_eval[1][1] += 1
                    eval.append(sample)
                elif sample["sentiment"] == 2 and balance_eval[1][2] < balance_eval[0][2]:
                    balance_eval[1][2] += 1
                    eval.append(sample)

                if len(train) >= max_chunk:
                    
                    tokenized_train = tokenize_and_align_labels(train, config.tokenizer)
                    write_chunk(tokenized_train, train_path, n_t)
                    train = []
                    n_t += 1

                if len(eval) >= max_chunk:

                    tokenized_eval = tokenize_and_align_labels(eval, config.tokenizer)
                    write_chunk(tokenized_eval, eval_path, n_e)
                    eval = []
                    n_e += 1

            pbar.update(1)

        if len(train) > 0:
            tokenized_train = tokenize_and_align_labels(train, config.tokenizer)
            write_chunk(tokenized_train, train_path, n_t)

        if len(eval) > 0:
            tokenized_eval = tokenize_and_align_labels(eval, config.tokenizer)
            write_chunk(tokenized_eval, eval_path, n_e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=str, help="Dataset to make trainingdata")
    parser.add_argument("--chunkdist_n", type=int, help="Set size for individual batch, must be greater than 0. Default value is 10000")

    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunckdist_n_checker(args.chunkdist_n)

    stage_train_eval_tokenize(args.dataset, args.chunkdist_n)