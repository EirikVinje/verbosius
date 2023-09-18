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
        ID of chunkdist to use for trainingdata. Must be an integer.
    
    Returns
    -------
    None

    """

    trainingdata_chunkdist = os.path.join(output, f"{dataset}_chunkdist_{chunkdist_n}")

    if not os.path.exists(trainingdata_chunkdist):
        os.mkdir(trainingdata_chunkdist)

    else:
        assert False, f"Directory {trainingdata_chunkdist} already exists, please remove it before continuing"
    
    trainingdata_chunkdist = os.path.join(trainingdata_chunkdist, "train_val")

    if not os.path.exists(trainingdata_chunkdist):
        os.makedirs(trainingdata_chunkdist)
    
    all_error_data = []
    n = 0

    preproc_dist = os.path.join(input, f"{dataset}_chunkdist_{chunkdist_n}", "train_val")

    print()

    dir = sorted(os.listdir(preproc_dist))

    while True:

        if n >= len(dir):
            break
        
        chunk = dir[n]

        verbose = True if type(dir[n]) != type(dir[0]) else False
        error_params = True if type(dir[n]) != type(dir[0]) else False

        chunk = os.path.join(preproc_dist, chunk) if not verbose else None
        data = pickle.load(open(chunk, "rb")) if not verbose else dir[n]

        data, train_error_data, eval_error_data = gen_data.make_weighted_data(data, config.error_chunk, verbose, error_params)

        gen_data.write_chunk(data, trainingdata_chunkdist, n)
    
        all_error_data.extend(train_error_data)
        all_error_data.extend(eval_error_data)

        if len(all_error_data) > config.n_badtexts:
            
            train_error_data, eval_error_data = train_test_split(all_error_data, test_size=0.2, random_state=42)

            data = {"train": train_error_data, "validation": eval_error_data,}
            
            dir.append(data)

            all_error_data = []

        n += 1
    
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
    