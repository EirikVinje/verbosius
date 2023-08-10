import argparse
import os
import numpy as np


import chunking.chunker_functions as chunker_functions


def stage_chunks(dataset : str, chunk_size : int, chunk_amount : int, input : str, output : str, test_size : float, validation : bool, seed : int, shuffle : int):
    
    ds = chunker_functions.dataset(dataset)
    ds = ds(two_cat=True)
    chunked_data = chunker_functions.chunk_data_multiclass(dataset = ds,
                                                    n_chunks_per_mix=chunk_amount,
                                                    chunk_size = chunk_size,
                                                    path = input,
                                                    test_size=test_size,
                                                    validation = validation,
                                                    val_size=0.2,
                                                    shuffle=shuffle,
                                                    seed=seed)
    
    test_size = len(chunked_data[2][0])
    validation_size = len(chunked_data[4][0]) if chunked_data[4] is not None else 0
    
    chunker_functions.write_chunks(chunked_data, output, dataset, chunk_size, test_size, validation_size, chunk_amount)


def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon', 'mnist']
    if dataset.lower() not in valid_datasets:
        raise argparse.ArgumentTypeError(f"Invalid dataset, available datasets are: {(i for i in valid_datasets)}")
    return dataset.lower()


def chunk_size_checker(chunk_size):
    chunk_size = int(chunk_size)

    if chunk_size <= 0:
        raise argparse.ArgumentTypeError(f"Invalid chunk size, chunk size must be greater than 0")

    return chunk_size


def chunk_amount_checker(chunk_amount):
    chunk_amount = int(chunk_amount)

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


def bool_checker(test):
    if test.lower() == "true" or int(test) == 1:
        return True
    elif test.lower() == "false" or int(test) == 0:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid value, {test} is not a valid value. Valid values are true and false")


def test_size_checker(test_size):
    test_size = float(test_size)

    if test_size <= 0 or test_size >= 1:
        raise argparse.ArgumentTypeError(f"Invalid test size, test size must be between 0 and 1")
    
    return test_size


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=dataset_checker, 
                        help="Dataset to stage")
    
    parser.add_argument("--chunk_size", type=chunk_size_checker, nargs='?', default=10000,
                        help="Set size for individual chunk, must be greater than 0. Default value is 10000. If used together with a train/test - split this chunk size will be split up into a train and test part accordingly. ")
    
    parser.add_argument("--chunk_amount", type=chunk_amount_checker, nargs ='?', default=1,
                        help="Set amount of chunks to stage at a time. Minimum value is 1. Default value is 1.")

    parser.add_argument("--input", type=input_checker, 
                        help="Path to input data, must be the absolute path to a valid directory where the datafiles are located.")
    
    parser.add_argument("--output", type=output_checker, 
                        help="Path to output data, must be a path to a directory that exists and is writable.")
    
    parser.add_argument("--test_size", type=test_size_checker, nargs='?', default=0.5,
                        help="Set the percentage size of the test data. If not sat test and train sizes will be equal.")

    parser.add_argument("--validation", type=bool_checker, nargs='?', default=0,
                        help="Set whether or not your data has its own test set already, if test==True and use_test_set==False test data is extracted from the training data. Default value is false. If sat test_size will be ignored, and test chunked will be extracted from the test set with same size and amount as for the training set. To change this set chunk_size_test and chunk_amount_test.")

    parser.add_argument("--seed", type=int, nargs='?', default=np.random.randint(0, 99999999),
                        help="Set seed for all randomizxation in chunking. Default value is a random seed.")
    
    parser.add_argument("--shuffle", type=bool_checker, nargs='?', default=1,
                        help="Set whether or not to shuffle the data. Default value is true.")

    args = parser.parse_args()

    stage_chunks(args.dataset, args.chunk_size, args.chunk_amount, args.input, args.output, args.test_size, args.validation, args.seed, args.shuffle)