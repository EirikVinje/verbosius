import pickle
import os
import argparse

import preprocessing.preprocess as preprocess
import preprocessing.datasource as datasource
import preprocessing.stage as stage


def main(dataset:str, batch_size:int, batch_amount_per_mix:int, input:str, output:str, test:bool):

    ds = datasource.dataset(dataset)

    
    ds = ds(two_cat=True)
    batched_data = datasource.batch_data(dataset = ds,
                                        n_batches_per_mix = batch_amount_per_mix,
                                        batch_size = batch_size,
                                        start_point = 0,
                                        path = input,
                                        test = test)
    

    
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

        stage.write_data(train_data, path=output, name=f"{dataset}_train")
        stage.write_data(test_data, path=output, name=f"{dataset}_test")
        
    # TODO : python stager.py --dataset imdb --batch 0 1000 --input path/to/input --output path/to/stageroutput

    

def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon']
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

def test_checker(test):
    if test.lower() == "true" or int(test) == 1:
        return True
    elif test.lower() == "false" or int(test) == 0:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid value, {test} is not a valid value. Valid values are true and false")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")


    parser.add_argument("--dataset", type=dataset_checker, 
                        help="Dataset to stage")
    
    parser.add_argument("--batch_size", type=batch_size_checker, nargs='?', default=10000,
                        help="Set size for individual batch, must be greater than 0. Default value is 10000")
    
    parser.add_argument("--batch_amount", type=batch_amount_checker, nargs ='?', default=1,
                        help="Set amount of batches to stage at a time. Minimum value is 1. Default value is 1.")

    parser.add_argument("--input", type=input_checker, 
                        help="Path to input data, must be the absolute path to a valid directory where the datafiles are located.")
    
    parser.add_argument("--output", type=output_checker, 
                        help="Path to output data, must be a path to a directory that exists and is writable.")
    
    parser.add_argument("--test", type=test_checker, nargs='?', default=1,
                        help="Set whether or not test data should also be prepared. Default value is true.")

    args = parser.parse_args()


    main(args.dataset, args.batch_size, args.batch_amount_per_mix, args.input, args.output, args.test)


    