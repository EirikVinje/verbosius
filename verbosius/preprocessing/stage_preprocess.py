import pickle
import os
import argparse

import numpy as np
from tqdm import tqdm

import preprocessing.preprocess as preprocess
import preprocessing.datasource as datasource
import preprocessing.stage as stage


def stage_preprocess(dataset:str, chunk_size:int, chunk_amount_per_mix:int, input:str, output:str, test:bool, test_size:float, use_test_set:bool, chunk_size_test:int, chunk_amount_test:int, seed:int, shuffle:bool):

    """
    Stage data for training

    Parameters
    ----------
    dataset : str
        Name of dataset to stage trainingdata for
    
    chuck_size : int
        Set size for individual chunk.
    
    chunk_amount : int
        Set amount of chunks to stage at a time. Minimum value is 1. Default value is 1.

    input : str
        Path to input data, must be the absolute path to a valid directory where the datafiles are located.
    
    output : str
        Path to output data, must be a path to a directory that exists and is writable.
    
    test : bool
        Set whether or not test data should also be prepared. Default value is true.
    
    test_size : float
        Set the percentage size of the test data. If not sat test and train sizes will be equal.
    
    use_test_set : bool
        Set whether or not your data has its own test set already, if test==True and use_test_set==False test data is extracted from the training data. Default value is false. If sat test_size will be ignored, and test chunked will be extracted from the test set with same size and amount as for the training set. To change this set chunk_size_test and chunk_amount_test.
    
    chunk_size_test : int
        Set size for individual chunk, must be greater than 0. If sat test_size argument will be ignored.
    
    chunk_amount_test : int
        Set amount of chunks to stage at a time. Minimum value is 1.
    
    seed : int
        Set seed for all randomizxation in chunking. Default value is a random seed.
    
    shuffle : bool
        Set whether or not to shuffle the data. Default value is true.
        
    """



    ds = datasource.dataset(dataset)
    ds = ds(two_cat=True)
    chunked_data = datasource.chunk_data_multiclass(dataset = ds,
                                                    n_chunks_per_mix = chunk_amount_per_mix,
                                                    chunk_size = chunk_size,
                                                    path = input,
                                                    use_test_set=use_test_set,
                                                    test_chunk_size=chunk_size_test,
                                                    test_size=test_size,
                                                    validation = True,
                                                    use_val_set=False,
                                                    val_chunk_size=-1,
                                                    val_size=.5,
                                                    shuffle=shuffle,
                                                    seed=seed)
    
    n_classes = chunked_data[-1]
    
    dir = os.listdir(output)
    n = len(dir)
    new_chunkdist = os.path.join(output, f"{dataset}_chunkdist_{n}")

    if not os.path.exists(new_chunkdist):
        os.mkdir(new_chunkdist)

    else:
        assert False, f"Directory {new_chunkdist} already exists, please remove it before continuing" 

    print(f"Preprocessing {dataset} chunkdist {n} with {chunk_amount_per_mix} chunks of size {chunk_size}")
    
    print(f"Train : {len(batched_data[0])} | Test : {len(batched_data[2])} | Val : {len(batched_data[4])}")
    
    assert False, "Stop here"


    for i, _ in enumerate(tqdm(range(len(batched_data[0])))):

        train_x = batched_data[0][i]
        train_y = batched_data[1][i]
        test_x = batched_data[2][i]
        test_y = batched_data[3][i]
        val_x = batched_data[4][i]
        val_y = batched_data[5][i]

        

    
    
        print("val x: ", val_x)

        # clean the data from unwanted symbols and such
        cleaned_train_x = preprocess.clean_text(train_x)
        cleaned_val_x = preprocess.clean_text(val_x)

        assert False, "Stop here"

        # lemmatize the data
        split_train_x, token_train_x, lemma_train_x = preprocess.lemmatize(cleaned_train_x, lemmatizer="en_core_web_sm")
        split_val_x, token_val_x, lemma_val_x = preprocess.lemmatize(cleaned_val_x, lemmatizer="en_core_web_sm") if val_x != None else None

        # get token maps
        token_ids_train_x = preprocess.map_tokens(split_train_x, token_train_x)
        token_ids_val_x = preprocess.map_tokens(split_val_x, token_val_x) if val_x != None else None

        # stage data
        train_data = stage.stage_data(cleaned_train_x, split_train_x, token_train_x, lemma_train_x, token_ids_train_x, train_y)
        val_data = stage.stage_data(cleaned_val_x, split_val_x, token_val_x, lemma_val_x, token_ids_val_x, val_y) if val_x != None else None

        test_data = [{"text" : text, "label" : label} for text, label in zip(test_x, test_y)]

        # write data
        data = {"train": train_data, 
                "validation": val_data,
                "test" : test_data,
                "distributer" : dataset, 
                "n_classes" : n_classes}
        
        stage.write_data(data=data, path=new_chunkdist)
        
    
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

def use_test_set_checker(use_test_set):
    if use_test_set.lower() == "true" or int(use_test_set) == 1:
        return True
    elif use_test_set.lower() == "false" or int(use_test_set) == 0:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid value, {use_test_set} is not a valid value. Valid values are true and false")




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
    
    parser.add_argument("--test", type=bool_checker, nargs='?', default=1,
                        help="Set whether or not test data should also be prepared. Default value is true.")

    parser.add_argument("--test_size", type=test_size_checker, nargs='?', default=0.5,
                        help="Set the percentage size of the test data. If not sat test and train sizes will be equal.")

    parser.add_argument("--use_test_set", type=bool_checker, nargs='?', default=0,
                        help="Set whether or not your data has its own test set already, if test==True and use_test_set==False test data is extracted from the training data. Default value is false. If sat test_size will be ignored, and test chunked will be extracted from the test set with same size and amount as for the training set. To change this set chunk_size_test and chunk_amount_test.")

    parser.add_argument("--chunk_size_test", type=chunk_size_checker, nargs='?', default=-1,
                        help="Set size for individual chunk, must be greater than 0. If sat test_size argument will be ignored.")

    parser.add_argument("--chunk_amount_test", type=chunk_amount_checker, nargs ='?', default=-1,
                        help="Set amount of chunks to stage at a time. Minimum value is 1.")
    
    parser.add_argument("--seed", type=int, nargs='?', default=np.random.randint(0, 99999999),
                        help="Set seed for all randomizxation in chunking. Default value is a random seed.")
    
    parser.add_argument("--shuffle", type=bool_checker, nargs='?', default=1,
                        help="Set whether or not to shuffle the data. Default value is true.")



    args = parser.parse_args()


    stage_preprocess(args.dataset, args.chunk_size, args.chunk_amount, args.input, args.output, args.test, args.test_size, args.use_test_set, args.chunk_size_test, args.chunk_amount_test, args.seed, args.shuffle)