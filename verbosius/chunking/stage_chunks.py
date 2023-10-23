import argparse
import os
import numpy as np


import chunking.chunker_functions as chunker_functions
import chunking.get_data as get_data

import config as config


def stage_chunks(dataset : str, chunk_size : int, chunk_amount : int, input : str, output : str, chunkdist_n : int):

    """
    Makes chunks of data for further preprocessing and training.

    Parameters
    ----------
    dataset : str
        Name of dataset to stage chunks for.
    
    chunk_size : int
        Size of individual chunk. Must be greater than 0.
    
    chunk_amount : int
        Amount of chunks to stage at a time. Must be greater than 0.
    
    input : str
        Path to input data. Must be absolute path to directory.
    
    output : str
        Path to output of this module. Must be absolute path to directory.
    
    chunkdist_n : int
        ID of chunkdistribution. Must be an integer. Will be used to name the output directory, e.g "path/to/output/{dataset}_chunkdist_{chunkdist_n}".
    """

    print("Chunk size: ", chunk_size)
    print("Chunk amount: ", chunk_amount)

    ds = get_data.dataset(dataset)
    ds = ds(two_cat=True)
    
    chunked_data = chunker_functions.chunk_data_multiclass_supersample(dataset = ds,
                                                    n_chunks_per_mix=chunk_amount,
                                                    chunk_size = chunk_size,
                                                    path = input,
                                                    test_size=config.test_size,
                                                    validation = config.validation,
                                                    val_size=config.val_size,
                                                    shuffle=config.shuffle,
                                                    seed=config.seed)
    
    new_chunkdist = os.path.join(output, f"{dataset}_chunkdist_{chunkdist_n}")

    if not os.path.exists(new_chunkdist):
        os.mkdir(new_chunkdist)
    else:
        assert False, f"Directory {new_chunkdist} already exists, please remove it before continuing"


    for i, _ in enumerate(range(len(chunked_data[0]))):

        train_x = chunked_data[0][i]
        train_y = chunked_data[1][i]

        val_x = chunked_data[2][i] if chunked_data[2] is not None else None
        val_y = chunked_data[3][i] if chunked_data[3] is not None else None

        orig_train_y = chunked_data[4][i] if chunked_data[4] is not None else None
        orig_val_y = chunked_data[5][i] if chunked_data[5] is not None else None

        train_val = {"train_x": train_x,
                     "train_y": train_y,
                     "val_x": val_x,
                     "val_y": val_y,
                     "orig_train_y": orig_train_y,
                     "orig_val_y": orig_val_y
                     }
        
        chunker_functions.write_chunks(new_chunkdist, train_val, test=False)
    
    
    train_length = len(chunked_data[0][0])
    test_length = len(chunked_data[2][0])
    validation_length = len(chunked_data[4][0]) if chunked_data[4] is not None else 0
    n_classes = chunked_data[-1]
    
    chunker_functions.write_meta_chunks(new_chunkdist, 
                                        train_length, 
                                        validation_length, 
                                        test_length, 
                                        dataset, 
                                        n_classes, 
                                        config.seed, 
                                        config.shuffle, 
                                        chunk_amount)


def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon', 'mnist']
    if dataset.lower() not in valid_datasets:
        raise argparse.ArgumentTypeError(f"Invalid dataset, available datasets are: {(i for i in valid_datasets)}")
    return dataset.lower()


def chunk_size_checker(chunk_size):
    
    if chunk_size <= 0:
        raise argparse.ArgumentTypeError(f"Invalid chunk size, chunk size must be greater than 0")

    return chunk_size


def chunk_amount_checker(chunk_amount):
    
    if chunk_amount <= 0:
        raise argparse.ArgumentTypeError(f"Invalid chunk amount, chunk amount must be greater than 0")
    
    return chunk_amount


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


def chunckdist_n_checker(chunkdist_n):
    
    if chunkdist_n < 0:
        raise argparse.ArgumentTypeError(f"Invalid chunkdist_n, chunkdist_n must be greater than or equal to 0")
    
    return chunkdist_n


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=str, help="Dataset to stage")
    parser.add_argument("--input", type=str, help="Path to input data, must be the absolute path to a valid directory where the datafiles are located.")
    parser.add_argument("--output", type=str, help="Path to output data, must be a path to a directory that exists and is writable.")
    parser.add_argument("--chunk_size", type=int, nargs='?', default=10000, help="Set size for individual chunk, must be greater than 0. Default value is 10000. If used together with a train/test - split this chunk size will be split up into a train and test part accordingly. ")
    parser.add_argument("--chunk_amount", type=int, nargs ='?', default=1, help="Set amount of chunks to stage at a time. Minimum value is 1. Default value is 1.")
    parser.add_argument("--chunkdist_n", type=int, help="Set size for individual batch, must be >= 0. ")
    
    args = parser.parse_args()

    dataset_checker(args.dataset)
    chunk_size_checker(args.chunk_size)
    chunk_amount_checker(args.chunk_amount)
    input_checker(args.input)
    output_checker(args.output)
    chunckdist_n_checker(args.chunkdist_n)

    stage_chunks(args.dataset, args.chunk_size, args.chunk_amount, args.input, args.output, args.chunkdist_n)
    # stage_chunks(dataset : str, chunk_size : int, chunk_amount : int, input : str, output : str, chunkdist_n : int)