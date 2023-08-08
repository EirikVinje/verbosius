import argparse
import pickle
import os
import re

import config as config
import xai_transformer.helper_functions as hf
import xai_transformer.transformer as tf
import xai_validation.helper_functions_xaival as hf_xaival


def stage_transformer(dataset : str, input : str, output : str, save_model : str, chunkdist_n : tuple):

    """
    
    Train transformer on staged trainingdata

    Parameters
    ----------
    dataset : str
        Name of dataset to stage trainingdata for
    
    input : str
        Path to batchdistros of dataset, must be the absolute path to a valid directory where the datafiles are located.
    
    output : str
        Path to output model, must be a path to a directory that exists and is writable.
    
    save_model : str
        Save model or not, either 'true' or 'false'.
    
    batchdist_n : tuple
        Batchdist_n to stage trainingdata for, must use tuple interval, e.g (0,-1) is all batchdistros

    """

    path = os.path.join(output, f"{dataset}_model_{len(os.listdir(output))}")
    if not os.path.exists(path):
        os.mkdir(path)
    output = path

    dists = sorted(os.listdir(input))

    chunk_dists = dists[chunkdist_n[0]:chunkdist_n[1]] if chunkdist_n[1] != -1 else dists[chunkdist_n[0]:]

    train_data = {"input_ids": [], "attention_mask": [], "labels": [], "targets": [], "sentiment": []}
    val_data = {"input_ids": [], "attention_mask": [], "labels": [], "targets": [], "sentiment": []}
    test_data = {"input_ids": [], "attention_mask": [], "targets": []}
    test_y = []
    
    tokenizer = config.tokenizer

    print("chunkdistros: ")
    for dist in chunk_dists:
        print(dist)
    print()

    for dist in chunk_dists:

        path = os.path.join(input, dist)
        dir = os.listdir(path)
        n_batches = len(dir)
        
        for n in range(n_batches):
            
            data = pickle.load(open(f"{path}/chunk_{n}.pkl", "rb"))
            new_train_batch = hf.tokenize_and_align_labels(data["train"], tokenizer) 
            new_val_batch = hf.tokenize_and_align_labels(data["validation"], tokenizer)
            train_data = hf.extend_data(train_data, new_train_batch)
            val_data = hf.extend_data(val_data, new_val_batch)  
            
            new_test_batch = hf_xaival.tokenize_to_model([ins["text"] for ins in data["test"]], tokenizer, config.device)
            test_data = hf.extend_test(test_data, new_test_batch)
            test_y.extend([ins["sentiment"] for ins in data["test"]])

    print("Train size: ", len(train_data["input_ids"]))
    print("Test size: ", len(test_data))
    print("Validation size: ", len(val_data["input_ids"])) if val_data["input_ids"] != [] else None

    train_data = hf.Dataset(**train_data)
    
    if type(val_data) != type(None):
        val_data = hf.Dataset(**val_data)
    
    else:
        val_data = None

    seq_acc, tok_acc = tf.transformer_pipeline(output_dir=output, 
                                               train_data=train_data, 
                                               val_data=val_data, 
                                               test_data=test_data 
                                               )
    
    return seq_acc, tok_acc

def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon']
    if dataset.lower() not in valid_datasets:
        raise argparse.ArgumentTypeError(f"Invalid dataset, available datasets are: {(i for i in valid_datasets)}")
    return dataset.lower()


def batchdist_checker(batchdist_range, input, dataset):

    if batchdist_range[1] != -1:

        for i in range(batchdist_range[0], batchdist_range[1]):
            if not os.path.exists(os.path.join(input, f"{dataset}_batchdist_{i}")):
                raise argparse.ArgumentTypeError(f"Invalid batch dist, batch dist {i} does not exist")

    elif batchdist_range[1] == -1:
        
        dir = os.listdir(input)
        n = len(dir)

        for i in range(batchdist_range[0], n):
            if not os.path.exists(os.path.join(input, f"{dataset}_batchdist_{i}")):
                raise argparse.ArgumentTypeError(f"Invalid batch dist, batch dist {batchdist_range[0]} does not exist")


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


def save_checker(save_model):
    if save_model.lower() == "true":
        return True
    elif save_model.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError(f'Invalid save_model, "{save_model}" is not "true" or "false"')


def n_batch_dist_checker(batchdist_range):

    regex = re.compile(r"\((\d+),(-?\d+)\)")

    match = regex.match(batchdist_range)

    if match is None:
        raise argparse.ArgumentTypeError(f"Invalid n_batch_dist, must be tuple interval, e.g (0,-1) is all batchdistros and on the exact form (int,int)")

    batchdist_range = tuple(map(int, batchdist_range.strip("()").split(",")))
    if batchdist_range[0] < 0:
        raise argparse.ArgumentTypeError(f"Invalid intervals in n_batch_dist")
    
    if batchdist_range[0] > batchdist_range[1] and batchdist_range[1] != -1:
        raise argparse.ArgumentTypeError(f"Invalid interval, {batchdist_range[0]} is larger than {batchdist_range[1]}")
    
    return batchdist_range

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=dataset_checker, 
                        help="Dataset to make trainingdata")
    
    parser.add_argument("--input", type=input_checker,
                        help="Path to batchdistros of dataset, must be the absolute path to a valid directory where the datafiles are located.")
    
    parser.add_argument("--output", type=output_checker, 
                        help="Path to output data, must be a path to a directory that exists and is writable.")
    
    parser.add_argument("--save_model", type=save_checker, nargs="?", default="false",
                        help="Save model or not, either 'true' or 'false'.")
    
    parser.add_argument("--batchdist_range", type=n_batch_dist_checker,
                        help="Number of batchdistros to use, must use tuple interval, e.g (0,-1) is all batchdistros")

    args = parser.parse_args()

    batchdist_checker(args.batchdist_range, args.input, args.dataset)

    stage_transformer(args.dataset, args.input, args.output, args.save_model, args.batchdist_range)
    