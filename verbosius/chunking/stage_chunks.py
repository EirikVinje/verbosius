import argparse
import os

from tqdm import tqdm

from chunking.chunker_functions import chunk_data_multiclass_supersample, write_chunks, write_meta_chunks
import chunking.get_data as get_data
import arg_funcs as af
import config as config


def stage_chunks(dataset : str, chunk_size : int, chunk_amount : int, chunkdist_n : int, size : str):

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
    
    chunkdist_n : int
        ID of chunkdistribution. Must be an integer. Will be used to name the output directory, e.g "path/to/output/{dataset}_chunkdist_{chunkdist_n}".
    """

    chunkdist_name = f"{dataset}_chunkdist_{chunkdist_n}"

    if not os.path.exists(config.root):
        assert False, f"Directory {config.root} does not exist, please create it before continuing"

    chunking_path = os.path.join(config.root, "chunking")

    if not os.path.exists(chunking_path):
        os.mkdir(chunking_path)
    
    chunking_path = os.path.join(chunking_path, chunkdist_name)
    if not os.path.exists(chunking_path):
        os.mkdir(chunking_path)
    else:
        assert False, f"Directory {chunking_path} already exists, please remove it before continuing"
    
    ds = get_data.dataset(dataset)
    ds = ds(two_cat=True, size=size)
    
    chunked_data = chunk_data_multiclass_supersample(dataset = ds,
                                                    n_chunks_per_mix=chunk_amount,
                                                    chunk_size = chunk_size,
                                                    shuffle=True,
                                                    seed=config.seed)
    

    for i in tqdm(range(len(chunked_data[0])), desc="Chunking data"):
        
        
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
        
        write_chunks(chunking_path, train_val)
    
    
    train_length = len(chunked_data[0][0])
    chunk_amount = len(chunked_data[0])
    validation_length = 0 
    n_classes = chunked_data[-1]

    write_meta_chunks(chunking_path, 
                    train_length, 
                    validation_length, 
                    dataset, 
                    n_classes, 
                    config.seed, 
                    True, 
                    chunk_amount)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=str, help="Dataset to stage")
    parser.add_argument("--chunk_size", type=int, nargs='?', default=10000, help="Set size for individual chunk, must be greater than 0. Default value is 10000. If used together with a train/test - split this chunk size will be split up into a train and test part accordingly. ")
    parser.add_argument("--chunk_amount", type=int, nargs ='?', default=1, help="Set amount of chunks to stage at a time. Minimum value is 1. Default value is 1.")
    parser.add_argument("--chunkdist_n", type=int, help="Set size for individual batch, must be >= 0. ")
    parser.add_argument("--size", type=str, help="Size of dataset to use")
    
    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunk_size_checker(args.chunk_size)
    af.chunk_amount_checker(args.chunk_amount)
    af.chunckdist_n_checker(args.chunkdist_n)

    stage_chunks(args.dataset, args.chunk_size, args.chunk_amount, args.chunkdist_n, args.size)
    