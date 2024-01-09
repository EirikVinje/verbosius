import argparse
import pickle
import os
import json
from datetime import datetime
import time
import gc

from sklearn.model_selection import train_test_split

import config as config
import chunking.chunker_functions as cf
import chunking.get_data as gd
import xai_transformer.helper_functions as hf
import xai_transformer.transformer as tf
import xai_validation.helper_functions_xaival as hf_xaival
import arg_funcs as af


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
    
    start_t = time.time()

    root = config.root

    models_folder = os.path.join(root, dataset, "models")
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    model_folder = os.path.join(model_output, f"{dataset}_model_dist_{chunkdist_n}")
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    else:
        assert False, f"Directory {model_folder} already exists, please remove it before continuing"

    
    model_path = os.path.join(model_folder, "model")

    trainingdata_folder = os.path.join(root, dataset, "trainingdata")
    if not os.path.exists(trainingdata_folder):
        assert False, f"Trainingdata folder {trainingdata_folder} does not exist, please check your input"

    chunk_dist = os.path.join(trainingdata_folder, f"{dataset}_chunkdist_{chunkdist_n}")

    chunks = sorted(os.listdir(chunk_dist))
    all_train_data = []
    for _, chunk in enumerate(chunks):
        
        chunk = os.path.join(chunk_dist, chunk)
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
    
    end_t = time.time()

    time_dict = {"time_hours" : (end_t - start_t) / 3600}
    
    with open(os.path.join(model_folder, "time.json"), "w") as f:
        json.dump(time_dict, f, indent=4)
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=str, help="Dataset to train on")
    parser.add_argument("--input_traindata", type=str, help="train and val data path")
    parser.add_argument("--model_output", type=str, help="Path to output model, must be a path to a directory that exists and is writable.")
    parser.add_argument("--chunkdist_n", type=int, help="Select chunkdist to train on")

    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.input_checker(args.input_traindata)
    af.output_checker(args.model_output)
    af.chunkdist_checker(args.dataset, args.input_traindata, args.chunkdist_n)

    stage_transformer(args.dataset, args.input_traindata, args.model_output, args.chunkdist_n)
