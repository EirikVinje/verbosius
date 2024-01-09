import argparse
import os
import numpy as np

import chunking.chunker_functions as chunker_functions
import chunking.get_data as get_data
import arg_funcs as af
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

    root = config.root
    dataset_folder = os.path.join(root, dataset)
    
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)

    chunking_folder = os.path.join(dataset_folder, "chunking")

    if not os.path.exists(chunking_folder):
        os.mkdir(chunking_folder)
    
    ds = get_data.dataset(dataset)
    ds = ds(two_cat=True)
    
    chunked_data = chunker_functions.chunk_data_multiclass_supersample(dataset = ds,
                                                    n_chunks_per_mix=chunk_amount,
                                                    chunk_size = chunk_size,
                                                    path = input,
                                                    test_size= config.test_size,
                                                    validation = config.validation,
                                                    val_size=config.val_size,
                                                    shuffle=config.shuffle,
                                                    seed=config.seed)
    
    new_chunkdist = os.path.join(chunking_folder, f"{dataset}_chunkdist_{chunkdist_n}")

    if not os.path.exists(new_chunkdist):
        os.mkdir(new_chunkdist)
    else:
        assert False, f"Directory {new_chunkdist} already exists, please remove it before continuing"

    for i, _ in enumerate(range(len(chunked_data[0]))):

        train_x = chunked_data[0][i]
        train_y = chunked_data[1][i]

        val_x = None 
        val_y = None 

        orig_train_y = chunked_data[2][i] if chunked_data[2] is not None else None
        orig_val_y = None 

        train_val = {"train_x": train_x,
                     "train_y": train_y,
                     "val_x": val_x,
                     "val_y": val_y,
                     "orig_train_y": orig_train_y,
                     "orig_val_y": orig_val_y
                     }
        
        chunker_functions.write_chunks(new_chunkdist, train_val, test=False)
    

    train_length = len(chunked_data[0][0])
    chunk_amount = len(chunked_data[0])
    validation_length = 0 #len(chunked_data[2][0]) if chunked_data[2] is not None else 0
    n_classes = chunked_data[-1]

    print()
    print("**************************************************************")
    print(f"name:                       {dataset}_chunkdist_{chunkdist_n}")
    print(f"size per train-chunk:       {train_length}")
    print(f"size per validation-chunk:  {validation_length}")
    print(f"number of chunks:           {chunk_amount}")
    print(f"number of classes:          {n_classes}")
    print("**************************************************************")
    print()

    chunker_functions.write_meta_chunks(new_chunkdist, 
                                        train_length, 
                                        validation_length, 
                                        dataset, 
                                        n_classes, 
                                        config.seed, 
                                        config.shuffle, 
                                        chunk_amount)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=str, help="Dataset to stage")
    parser.add_argument("--input", type=str, help="Path to input data, must be the absolute path to a valid directory where the datafiles are located.")
    parser.add_argument("--output", type=str, help="Path to output data, must be a path to a directory that exists and is writable.")
    parser.add_argument("--chunk_size", type=int, nargs='?', default=10000, help="Set size for individual chunk, must be greater than 0. Default value is 10000. If used together with a train/test - split this chunk size will be split up into a train and test part accordingly. ")
    parser.add_argument("--chunk_amount", type=int, nargs ='?', default=1, help="Set amount of chunks to stage at a time. Minimum value is 1. Default value is 1.")
    parser.add_argument("--chunkdist_n", type=int, help="Set size for individual batch, must be >= 0. ")
    
    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunk_size_checker(args.chunk_size)
    af.chunk_amount_checker(args.chunk_amount)
    af.input_checker(args.input)
    af.output_checker(args.output)
    af.chunckdist_n_checker(args.chunkdist_n)

    stage_chunks(args.dataset, args.chunk_size, args.chunk_amount, args.input, args.output, args.chunkdist_n)
    