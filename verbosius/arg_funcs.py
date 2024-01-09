import argparse
import os


def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon', 'mnist']
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


def chunckdist_n_checker(chunkdist_n):
    
    if chunkdist_n < 0:
        raise argparse.ArgumentTypeError(f"Invalid chunkdist_n, chunkdist_n must be greater than or equal to 0")
    
    return chunkdist_n
