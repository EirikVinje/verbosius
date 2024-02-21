import os
import pickle 
import argparse

import trainingdata.generate_trainingdata as gen_data
import config as config
import arg_funcs as af


def stage_trainingdata(dataset : str, chunkdist_n : int):

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

    root = config.root

    trainingdata_folder = os.path.join(root, dataset, "trainingdata")
    if not os.path.exists(trainingdata_folder):
        os.mkdir(trainingdata_folder)

    trainingdata_chunkdist = os.path.join(trainingdata_folder, f"{dataset}_chunkdist_{chunkdist_n}")
    if not os.path.exists(trainingdata_chunkdist):
        os.mkdir(trainingdata_chunkdist)
    else:
        assert False, f"Directory {trainingdata_chunkdist} already exists, please remove it before continuing"
    
    
    trainingdata_chunkdist = os.path.join(trainingdata_chunkdist)
    if not os.path.exists(trainingdata_chunkdist):
        os.mkdir(trainingdata_chunkdist)
    
    preprocess_folder = os.path.join(root, dataset, "preprocess")
    if not os.path.exists(preprocess_folder):
        assert False, f"Preprocess folder {preprocess_folder} does not exist, please check your input"

    chunk_dist = os.path.join(preprocess_folder, f"{dataset}_chunkdist_{chunkdist_n}")
    if not os.path.exists(chunk_dist):
        assert False, f"Chunk distribution {chunk_dist} does not exist, please check your input"

    chunk_dist = os.path.join(chunk_dist)

    dir = sorted(os.listdir(chunk_dist))
    dir_len = len(dir)

    n = 0
    correct_x = 0

    while True:

        if n >= dir_len * 2:
            break
        
        chunk = dir[n]

        error_params = True if type(dir[n]) != type(dir[0]) else False

        chunk = os.path.join(chunk_dist, chunk) if not error_params else None
        train_data = pickle.load(open(chunk, "rb")) if not error_params else dir[n]

        train_data, train_error_data = gen_data.make_weighted_data(train_data, error_params)

        correct_x += len(train_data)

        gen_data.write_chunk(train_data, trainingdata_chunkdist, n)
    
        dir.append(train_error_data)

        n += 1
    
    return correct_x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=str, help="Dataset to make trainingdata")
    parser.add_argument("--chunkdist_n", type=int, help="Set size for individual batch, must be greater than 0. Default value is 10000")

    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunckdist_n_checker(args.chunkdist_n)
    
    stage_trainingdata(args.dataset, args.chunkdist_n)
    