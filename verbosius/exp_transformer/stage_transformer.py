import argparse
import pickle
import os
import re

import config as config
import exp_transformer.helper_functions as hf
import exp_transformer.transformer as tf


def main(dataset : str, input : str, output : str, save_model : str, batchdist_n : tuple):
    
    tokenizer = config.tokenizer
    device = config.device
    learning_rate = config.learning_rate
    per_device_train_batch_size = config.per_device_train_batch_size
    per_device_eval_batch_size = config.per_device_eval_batch_size
    num_train_epochs = config.num_train_epochs
    weight_decay = config.weight_decay
    evaluation_strategy = config.evaluation_strategy
    save_strategy = config.save_strategy
    warmup_steps = config.warmup_steps
    load_best_model_at_end = config.load_best_model_at_end
    eval_accumulation_steps = config.eval_accumulation_steps
    label_names = config.label_names

    neutral_weight = config.neutral_weight
    loss_weight = config.loss_weight
    num_labels = config.num_labels
    num_seq_labels = config.num_seq_labels

    model = hf.CustomModel(num_labels, num_seq_labels, neutral_weight, loss_weight)
    model = model.to(device = device)

    
    dists = os.listdir(input)
    batch_dists = dists[batchdist_n[0]:batchdist_n[1]]

    train_data = []
    test_data = []

    for dist in batch_dists:

        path = os.path.join(input, dist)
        dir = os.listdir(path)
        n_batches = len(dir)

        for n in range(n_batches):
            
            data = pickle.load(open(f"{path}/batch_{n}.pkl", "rb"))
            
            train_data.extend(data["train"])
            test_data.extend(data["test"])

    train_data = hf.tokenize_and_align_labels(train_data, tokenizer, device)
    test_data = hf.tokenize_and_align_labels(test_data, tokenizer, device)
        
    train_data = hf.Dataset(**train_data)
    test_data = hf.Dataset(**test_data)

    tf.transformer_pipeline(device, 
                         output, 
                         learning_rate, 
                         per_device_train_batch_size, 
                         per_device_eval_batch_size, 
                         num_train_epochs, 
                         weight_decay, 
                         evaluation_strategy, 
                         save_strategy,
                         warmup_steps, 
                         load_best_model_at_end, 
                         eval_accumulation_steps, 
                         label_names,
                         train_data,
                         test_data,
                         tokenizer,
                         save_model,
                         model)

    if not save_model:
        pass


def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon']
    if dataset.lower() not in valid_datasets:
        raise argparse.ArgumentTypeError(f"Invalid dataset, available datasets are: {(i for i in valid_datasets)}")
    return dataset.lower()


def batchdist_checker(n_batchdist, input, dataset):

    if n_batchdist[1] != -1:

        for i in range(n_batchdist[0], n_batchdist[1]):
            if not os.path.exists(os.path.join(input, f"{dataset}_batchdist_{i}")):
                raise argparse.ArgumentTypeError(f"Invalid batch dist, batch dist {i} does not exist")

    elif n_batchdist[1] == -1:
        
        dir = os.listdir(input)
        n = len(dir)

        for i in range(n_batchdist[0], n):
            if not os.path.exists(os.path.join(input, f"{dataset}_batchdist_{i}")):
                raise argparse.ArgumentTypeError(f"Invalid batch dist, batch dist {n_batchdist[0]} does not exist")


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
    
    regex = re.compile(r"\((\d+),(\d+)\)")

    match = regex.match(batchdist_range)

    if match is None:
        raise argparse.ArgumentTypeError(f"Invalid n_batch_dist, must be tuple interval, e.g (0,-1) is all batchdistros and on the exact form (int,int)")

    batchdist_range = tuple(map(int, batchdist_range.strip("()").split(",")))
    if batchdist_range[0] < 0 or batchdist_range[1] < 0:
        raise argparse.ArgumentTypeError(f"Invalid n_batch_dist, must be tuple interval, e.g (0,-1) is all batchdistros")
    
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

    main(args.dataset, args.input, args.output, args.save_model, args.batchdist_range)
    