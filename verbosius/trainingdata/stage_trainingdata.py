import os
import pickle 
import argparse

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

import trainingdata.generate_trainingdata as gen_data
import config as config


def stage_trainingdata(dataset : str, input : str, batchdist_n : int, output : str):

    path = os.path.join(input, f"{dataset}_batchdist_{batchdist_n}")

    dir = os.listdir(path)

    n_batches = len(dir)

    print()
    print(f"Number of batches in {dataset} batchdist {batchdist_n}: {n_batches}")
    print()

    for n in tqdm(range(n_batches)):
        
        data = pickle.load(open(f"{path}/batch_{n}.pkl", "rb"))
        
        rm, feature_names = gen_data.rulemaker(data)
    
        train_data = gen_data.do_weighting(data["train"], feature_names, rm)
        val_data = gen_data.do_weighting(data["validation"], feature_names, rm)
        
        data = {"train": train_data, 
                "validation": val_data,
                "test": data["test"], 
                "distributer" : data["distributer"], 
                "n_classes" : data["n_classes"]}
        
        gen_data.write_data(data, output, dataset, batchdist_n, n)


def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon']
    if dataset.lower() not in valid_datasets:
        raise argparse.ArgumentTypeError(f"Invalid dataset, available datasets are: {(i for i in valid_datasets)}")
    return dataset.lower()


def batchdist_n_checker(batchdist_n, input, dataset):

    if batchdist_n == None:
        raise argparse.ArgumentTypeError(f"Invalid batch size, batch size must be greater than 0")

    if not os.path.exists(os.path.join(input, f"{dataset}_batchdist_{batchdist_n}")):
        raise argparse.ArgumentTypeError(f"Invalid batch dist, batch dist {batchdist_n} does not exist") 

    return batchdist_n


def input_checker(input):
    if os.access(os.path.dirname(input), os.W_OK) and os.path.isdir(input):
        return input
    else:
        raise argparse.ArgumentTypeError(f'Invalid input path, "{input}" is not writable or is not a directory')


def output_checker(output):
    #if os.access(os.path.dirname(output), os.W_OK) and os.path.isdir(output):
    #    return output
    #else:
    #    raise argparse.ArgumentTypeError(f'Invalid output path, "{output}" is not writable or is not a directory')

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")


    parser.add_argument("--dataset", type=dataset_checker, 
                        help="Dataset to make trainingdata")
    
    parser.add_argument("--input", type=input_checker,
                        help="Path to batchdistros of dataset, must be the absolute path to a valid directory where the datafiles are located.")
    
    
    parser.add_argument("--batchdist_n", type=int,
                        help="Set size for individual batch, must be greater than 0. Default value is 10000")
    
    
    parser.add_argument("--output", type=output_checker, 
                        help="Path to output data, must be a path to a directory that exists and is writable.")
    
    
    args = parser.parse_args()

    batchdist_n_checker(args.batchdist_n, args.input, args.dataset)

    stage_trainingdata(args.dataset, args.input, args.batchdist_n, args.output)