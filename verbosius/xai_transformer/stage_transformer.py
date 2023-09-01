import argparse
import pickle
import os
import re

import config as config
import xai_transformer.helper_functions as hf
import xai_transformer.transformer as tf
import xai_validation.helper_functions_xaival as hf_xaival


def stage_transformer(dataset : str, train_val_input : str, test_input : str, model_output : str, save_model : str, chunkdist_n : int):

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

    model_path = os.path.join(model_output, f"{dataset}_model_dist_{chunkdist_n}")
    
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    else:
        assert False, f"Directory {model_path} already exists, please remove it before continuing"
    
    trainingdata_chunkdist = os.path.join(train_val_input, f"{dataset}_chunkdist_{chunkdist_n}")
    
    chunks = sorted(os.listdir(trainingdata_chunkdist))
    
    train_data = {"input_ids": [], "attention_mask": [], "labels": [], "targets": [], "sentiment": []}
    val_data = {"input_ids": [], "attention_mask": [], "labels": [], "targets": [], "sentiment": []}
    
    for _, chunk in enumerate(chunks):
        
        chunk = os.path.join(trainingdata_chunkdist, chunk)

        train_val = pickle.load(open(chunk, "rb"))
        
        new_train_batch = hf.tokenize_and_align_labels(train_val["train"], config.tokenizer) 
        new_val_batch = hf.tokenize_and_align_labels(train_val["validation"], config.tokenizer)
        
        train_data = hf.extend_data(train_data, new_train_batch)
        val_data = hf.extend_data(val_data, new_val_batch)  
    
    test_x = {"input_ids": [], "attention_mask": [], "targets": []}
    test_y = []

    testdata_chunkdist = os.path.join(test_input, f"{dataset}_chunkdist_{chunkdist_n}", "test")
    test_chunks = sorted(os.listdir(testdata_chunkdist))

    for _, chunk in enumerate(test_chunks):

        chunk = os.path.join(testdata_chunkdist, chunk)
        test = pickle.load(open(chunk, "rb"))

        new_test_x = hf_xaival.tokenize_to_model([text for text in test["test_x"]], config.tokenizer, config.device)

        test_x = hf.extend_test(test_x, new_test_x)
        test_y.extend(test["test_y"])

    print()    
    print("Train size: ", len(train_data["input_ids"]))
    print("Test size: ", len(test_x["input_ids"]))
    print("Validation size: ", len(val_data["input_ids"])) if val_data["input_ids"] != [] else None
    print()
    
    seq_acc = tf.transformer_pipeline(output_dir=model_output, 
                                               train_data=train_data, 
                                               val_data=val_data, 
                                               test_x=test_x,
                                               test_y=test_y,
                                               save_model=save_model)
    
    return seq_acc

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
    