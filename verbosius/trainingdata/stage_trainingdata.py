import os
import pickle 
import argparse
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import trainingdata.generate_trainingdata as gen_data
import config as config


def stage_trainingdata(dataset : str, input : str, output : str, chunkdist_n : int, n_badtexts : int = 2000, error_chunk : bool = False):

    trainingdata_chunkdist = os.path.join(output, f"{dataset}_chunkdist_{chunkdist_n}")

    if not os.path.exists(trainingdata_chunkdist):
        os.mkdir(trainingdata_chunkdist)

    else:
        assert False, f"Directory {trainingdata_chunkdist} already exists, please remove it before continuing"
    
    all_error_data = []
    n = 0

    preproc_chunkdist = os.path.join(input, f"{dataset}_chunkdist_{chunkdist_n}", "train_val")
    
    print(f"Number of chunks in {dataset} chunkdist {chunkdist_n}: {len(os.listdir(preproc_chunkdist))} \n")
    
    while True:

        dir = sorted(os.listdir(preproc_chunkdist))
        n_chunks = len(dir)

        if n >= len(dir):
            break
        
        chunk = dir[n]

        verbose = True if dir[n][-5] == "e" else False
        error_params = True if dir[n][-5] == "e" else False

        chunk = os.path.join(preproc_chunkdist, chunk)
        data = pickle.load(open(chunk, "rb"))

        print(dir)
        print("verbose : ", verbose)
        print("error_params : ", error_params)

        data, train_error_data, eval_error_data = gen_data.make_weighted_data(data, error_chunk, verbose, error_params)

        gen_data.write_chunk(data, trainingdata_chunkdist, n)
    
        all_error_data.extend(train_error_data)
        all_error_data.extend(eval_error_data)

        print("Number of error instances: ", len(all_error_data), "\n")

        if error_chunk and len(all_error_data) > n_badtexts:
            
            train_error_data, eval_error_data = train_test_split(all_error_data, test_size=0.2, random_state=42)

            data = {"train": train_error_data, "validation": eval_error_data,}
            
            gen_data.write_error_chunk(data, preproc_chunkdist, n_chunks)

            all_error_data = []

        n += 1

    

def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon']
    if dataset.lower() not in valid_datasets:
        raise argparse.ArgumentTypeError(f"Invalid dataset, available datasets are: {(i for i in valid_datasets)}")
    return dataset.lower()


def chunkdist_n_checker(chunkdist_n, input, dataset):

    if chunkdist_n == None:
        raise argparse.ArgumentTypeError(f"Invalid chunk size, chunk size must be greater than 0")

    if not os.path.exists(os.path.join(input, f"{dataset}_chunkdist_{chunkdist_n}")):
        raise argparse.ArgumentTypeError(f"Invalid batch dist, chunk dist {chunkdist_n} does not exist") 

    return chunkdist_n


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
    
    
    parser.add_argument("--chunkdist_n", type=int,
                        help="Set size for individual batch, must be greater than 0. Default value is 10000")
    
    
    parser.add_argument("--output", type=output_checker, 
                        help="Path to output data, must be a path to a directory that exists and is writable.")
    
    
    args = parser.parse_args()

    chunkdist_n_checker(args.chunkdist_n, args.input, args.dataset)

    stage_trainingdata(args.dataset, args.input, args.chunkdist_n, args.output)