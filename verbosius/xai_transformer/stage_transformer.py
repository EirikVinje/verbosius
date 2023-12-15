import argparse
import pickle
import os
import json
import datetime
import gc

from sklearn.model_selection import train_test_split

import config as config
import chunking.chunker_functions as cf
import chunking.get_data as gd
import xai_transformer.helper_functions as hf
import xai_transformer.transformer as tf
import xai_validation.helper_functions_xaival as hf_xaival

def stage_transformer(dataset : str, train_val_input : str, model_output : str, chunkdist_n : int):

    """
    Train transformer on weigthed trainingdata.

    Parameters
    ----------
    dataset : str
        Name of dataset to train on.

    train_val_input : str
        Path to trainingdata. Must be absolute path to directory.
    
    test_input : str
        Path to testdata. Must be absolute path to directory.
    
    model_output : str
        Path to output of this module. Must be absolute path to directory.
    
    chunkdist_n : int
        ID of chunkdist to use for trainingdata. Must be an integer.
    
    return_seq_acc : bool
        If True, returns the sequence accuracy of the trained model. If False, returns None.

    """
    
    model_dir = os.path.join(model_output, f"{dataset}_model_dist_{chunkdist_n}")

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    else:
        assert False, f"Directory {model_dir} already exists, please remove it before continuing"

    model_path = os.path.join(model_dir, "model")

    trainingdata_dist = os.path.join(train_val_input, f"{dataset}_chunkdist_{chunkdist_n}", "train_val")
    
    chunks = sorted(os.listdir(trainingdata_dist))
    all_train_data = []
    for _, chunk in enumerate(chunks):
        
        chunk = os.path.join(trainingdata_dist, chunk)
        train_data = pickle.load(open(chunk, "rb"))        
        all_train_data.extend(train_data)

    train_data, val_data = train_test_split(all_train_data, test_size=0.2, random_state=config.seed, shuffle=True)

    train_tokenized = hf.tokenize_and_align_labels(train_data, config.tokenizer, orig_labels=True)
    val_tokenized = hf.tokenize_and_align_labels(val_data, config.tokenizer, orig_labels=True) 

    print()    
    print("Train size: ", len(train_tokenized["input_ids"]))
    print("Validation size: ", len(val_tokenized["input_ids"]))
    print("Epochs: ", config.num_train_epochs)
    print("Batch size: ", config.per_device_train_batch_size)
    print()
    
    tf.transformer_pipeline_custom(output_dir=model_path, 
                                    train_data=train_tokenized, 
                                    val_data=val_tokenized)
    
    


def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon']
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

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=str, help="Dataset to train on")
    parser.add_argument("--input_traindata", type=str, help="train and val data path")
    parser.add_argument("--model_output", type=str, help="Path to output model, must be a path to a directory that exists and is writable.")
    parser.add_argument("--chunkdist_n", type=int, help="Select chunkdist to train on")

    args = parser.parse_args()

    dataset_checker(args.dataset)
    input_checker(args.input_traindata)
    output_checker(args.model_output)
    chunkdist_checker(args.dataset, args.input_traindata, args.chunkdist_n)

    stage_transformer(args.dataset, args.input_traindata, args.model_output, args.chunkdist_n)
