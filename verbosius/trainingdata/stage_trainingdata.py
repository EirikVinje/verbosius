import os
import pickle 
import argparse
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import trainingdata.generate_trainingdata as gen_data
import config as config


def stage_trainingdata(dataset : str, input : str, output : str, chunkdist_n : int):

    """
    Stage trainingdata for transformer.
    Uses TM to weight individual tokens in a sequence of text.

    Parameters
    ----------
    dataset : str
        Name of dataset to stage trainingdata for.
    
    input : str
        Path to input data from preprocessed chunk. Must be absolute path to directory.

    output : str
        Path to output of this module. Must be absolute path to directory.
    
    chunkdist_n : int
        ID of chunkdist to use for trainingdata. Must be an integer. Will be used to name the output directory, e.g "path/to/output/{dataset}_chunkdist_{chunkdist_n}".
    
    
    """

    trainingdata_chunkdist = os.path.join(output, f"{dataset}_chunkdist_{chunkdist_n}")

    if not os.path.exists(trainingdata_chunkdist):
        os.mkdir(trainingdata_chunkdist)

    else:
        assert False, f"Directory {trainingdata_chunkdist} already exists, please remove it before continuing"
    
    trainingdata_chunkdist = os.path.join(trainingdata_chunkdist, "train_val")

    if not os.path.exists(trainingdata_chunkdist):
        os.mkdir(trainingdata_chunkdist)
    
    #all_error_data = []
    n = 0

    preproc_dist = os.path.join(input, f"{dataset}_chunkdist_{chunkdist_n}", "train_val")

    if not os.path.exists(preproc_dist):
        raise FileNotFoundError(f"Directory {preproc_dist} does not exist")

    dir = sorted(os.listdir(preproc_dist))
    dir_len = len(dir)

    correct_x = 0

    while True:

        if n >= dir_len * 2:
            break
        
        chunk = dir[n]

        error_params = True if type(dir[n]) != type(dir[0]) else False

        chunk = os.path.join(preproc_dist, chunk) if not error_params else None
        train_data = pickle.load(open(chunk, "rb")) if not error_params else dir[n]

        print("train: ", len(train_data))

        train_data, train_error_data = gen_data.make_weighted_data(train_data, error_params)

        print("train: ", len(train_data), "error: ", len(train_error_data))

        correct_x += len(train_data)

        gen_data.write_chunk(train_data, trainingdata_chunkdist, n)
    
        #all_error_data.extend(train_error_data)
            
        dir.append(train_error_data)

        n += 1
    
    return correct_x
    
def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon']
    if dataset.lower() not in valid_datasets:
        raise argparse.ArgumentTypeError(f"Invalid dataset, available datasets are: {(i for i in valid_datasets)}")
    return dataset.lower()


def chunkdist_checker(dataset, input, chunkdist_n):
    if not os.path.exists(os.path.join(input, f"{dataset}_chunkdist_{chunkdist_n}")):
        raise argparse.ArgumentTypeError(f"Invalid chunk dist, {dataset}_chunkdist_{chunkdist_n} does not exist") 

    return chunkdist_n


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=str, help="Dataset to make trainingdata")
    parser.add_argument("--input", type=str, help="Path to batchdistros of dataset, must be the absolute path to a valid directory where the datafiles are located.")
    parser.add_argument("--output", type=str, help="Path to output data, must be a path to a directory that exists and is writable.")
    parser.add_argument("--chunkdist_n", type=int, help="Set size for individual batch, must be greater than 0. Default value is 10000")

    args = parser.parse_args()

    dataset_checker(args.dataset)
    input_checker(args.input)
    output_checker(args.output)
    chunkdist_checker(args.dataset, args.input, args.chunkdist_n)

    stage_trainingdata(args.dataset, args.input, args.output, args.chunkdist_n)
    