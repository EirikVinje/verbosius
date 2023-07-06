import os
import pickle 
import argparse

import trainingdata.generate_trainingdata as gen_data


def main(batch_dist):

    batch_dist = f"batch_dist_{batch_dist}"
    root = os.path.expanduser('~')
    path = os.path.join(root, "projects/verbosius_data", batch_dist)
    
    n = len(os.listdir(path))/2
    
    for b in range(n):
    
        data = pickle.load(open(f"{path}/data_{b}.pkl", "rb"))

        rm = gen_data.rulemaker(data)


def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon']
    if dataset.lower() not in valid_datasets:
        raise argparse.ArgumentTypeError(f"Invalid dataset, available datasets are: {(i for i in valid_datasets)}")
    return dataset.lower()


def batchdist_n_checker(input, batchdist_n, dataset):
    
    if batchdist_n == None:
        raise argparse.ArgumentTypeError(f"Invalid batch size, batch size must be greater than 0")

    elif not os.path.exists(os.path.join(input, dataset, f"{dataset}_batchdist_{batchdist_n}")):
        raise argparse.ArgumentTypeError(f"Invalid batch dist, batch dist {batchdist_n} does not exist") 

    return batchdist_n


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


    parser.add_argument("--dataset", type=dataset_checker, 
                        help="Dataset to make trainingdata")
    
    parser.add_argument("--input", type=input_checker, 
                        help="Path to batchdistros of dataset, must be the absolute path to a valid directory where the datafiles are located.")
    
    
    parser.add_argument("--batchdist_n", type=batchdist_n_checker, nargs='?', default=10000,
                        help="Set size for individual batch, must be greater than 0. Default value is 10000")
    
    
    parser.add_argument("--output", type=output_checker, 
                        help="Path to output data, must be a path to a directory that exists and is writable.")
    
    
    args = parser.parse_args()


    main(args.dataset, args.batch_size, args.batch_amount, args.input, args.output, args.test)