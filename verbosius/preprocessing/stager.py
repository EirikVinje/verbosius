import pickle
import os
import argparse

import numpy as np
import preprocessing.preprocess as preprocess
import preprocessing.datasource as datasource
import preprocessing.stage as stage


def main(dataset:str, batch_size:int, batch_amount_per_mix:int, input:str, output:str, test:bool, test_size:float, use_test_set:bool, batch_size_test:int, batch_amount_test:int, seed:int, shuffle:bool):

    ds = datasource.dataset(dataset)
    ds = ds(two_cat=True)
    batched_data = datasource.batch_data_multiclass(dataset = ds,
                                                    n_batches_per_mix = batch_amount_per_mix,
                                                    batch_size = batch_size,
                                                    path = input,
                                                    test = test,
                                                    use_test_set=use_test_set,
                                                    test_batch_size=batch_size_test,
                                                    test_batches_per_mix=batch_amount_test,
                                                    test_size=test_size,
                                                    shuffle=shuffle,
                                                    seed=seed)
    
    n_classes = batched_data[-1]
    
    dir = os.listdir(output)
    n = len(dir)
    new_batchdist = os.path.join(output, f"{dataset}_batchdist_{n}")

    if not os.path.exists(new_batchdist):
        os.mkdir(new_batchdist)

    else:
        assert False, f"Directory {new_batchdist} already exists, please remove it before continuing" 

    for train_x, train_y, test_x, test_y in zip(batched_data[0], batched_data[1], batched_data[2], batched_data[3]):

        # clean the data from unwanted symbols and such
        cleaned_train_x = preprocess.clean_text(train_x)
        cleaned_test_x = preprocess.clean_text(test_x)

        # lemmatize the data
        split_train_x, token_train_x, lemma_train_x = preprocess.lemmatize(cleaned_train_x, lemmatizer="en_core_web_sm")
        split_test_x, token_test_x, lemma_test_x = preprocess.lemmatize(cleaned_test_x, lemmatizer="en_core_web_sm")

        # get token maps
        token_ids_train_x = preprocess.map_tokens(split_train_x, token_train_x)
        token_ids_test_x = preprocess.map_tokens(split_test_x, token_test_x)

        # stage data
        train_data = stage.stage_data(cleaned_train_x, split_train_x, token_train_x, lemma_train_x, token_ids_train_x, train_y)
        test_data = stage.stage_data(cleaned_test_x, split_test_x, token_test_x, lemma_test_x, token_ids_test_x, test_y)

        # write data
        data = {"train": train_data, "test": test_data, "distributer" : dataset, "n_classes" : n_classes}
        stage.write_data(data=data, path=new_batchdist)
        
    
def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon', 'mnist']
    if dataset.lower() not in valid_datasets:
        raise argparse.ArgumentTypeError(f"Invalid dataset, available datasets are: {(i for i in valid_datasets)}")
    return dataset.lower()


def batch_size_checker(batch_size):
    batch_size = int(batch_size)

    if batch_size <= 0:
        raise argparse.ArgumentTypeError(f"Invalid batch size, batch size must be greater than 0")

    return batch_size


def batch_amount_checker(batch_amount):
    batch_amount = int(batch_amount)

    if batch_amount <= 0:
        raise argparse.ArgumentTypeError(f"Invalid batch amount, batch amount must be greater than 0")
    
    return batch_amount


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
    
    parser.add_argument("--batch_size", type=batch_size_checker, nargs='?', default=10000,
                        help="Set size for individual batch, must be greater than 0. Default value is 10000. If used together with a train/test - split this batch size will be split up into a train and test part accordingly. ")
    
    parser.add_argument("--batch_amount", type=batch_amount_checker, nargs ='?', default=1,
                        help="Set amount of batches to stage at a time. Minimum value is 1. Default value is 1.")

    parser.add_argument("--input", type=input_checker, 
                        help="Path to input data, must be the absolute path to a valid directory where the datafiles are located.")
    
    parser.add_argument("--output", type=output_checker, 
                        help="Path to output data, must be a path to a directory that exists and is writable.")
    
    parser.add_argument("--test", type=bool_checker, nargs='?', default=1,
                        help="Set whether or not test data should also be prepared. Default value is true.")

    parser.add_argument("--test_size", type=test_size_checker, nargs='?', default=0.5,
                        help="Set the percentage size of the test data. If not sat test and train sizes will be equal.")

    parser.add_argument("--use_test_set", type=bool_checker, nargs='?', default=0,
                        help="Set whether or not your data has its own test set already, if test==True and use_test_set==False test data is extracted from the training data. Default value is false. If sat test_size will be ignored, and test batched will be extracted from the test set with same size and amount as for the training set. To change this set batch_size_test and batch_amount_test.")

    parser.add_argument("--batch_size_test", type=batch_size_checker, nargs='?', default=-1,
                        help="Set size for individual batch, must be greater than 0. If sat test_size argument will be ignored.")

    parser.add_argument("--batch_amount_test", type=batch_amount_checker, nargs ='?', default=-1,
                        help="Set amount of batches to stage at a time. Minimum value is 1.")
    
    parser.add_argument("--seed", type=int, nargs='?', default=np.random.randint(0, 99999999),
                        help="Set seed for all randomizxation in batching. Default value is a random seed.")
    
    parser.add_argument("--shuffle", type=bool_checker, nargs='?', default=1,
                        help="Set whether or not to shuffle the data. Default value is true.")


    args = parser.parse_args()


    main(args.dataset, args.batch_size, args.batch_amount, args.input, args.output, args.test, args.test_size, args.use_test_set, args.batch_size_test, args.batch_amount_test, args.seed, args.shuffle)