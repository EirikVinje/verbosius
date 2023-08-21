import pickle
import os
import argparse
import re

import numpy as np
from tqdm import tqdm

import preprocessing.preprocess_functions as preprocess_functions
import preprocessing.preprocess_functions as pf


def stage_preprocess(dataset:str, input:str, output:str, chunk_n : int, return_data=False):

    chunked_data = pf.load_chunk(dataset, chunk_n, input)         
        
    print("\n----------------------------------------------------------------------------------------------------")
    print(f"len train : {[len(chunk) for chunk in chunked_data[0]]}, len test : {[len(chunk) for chunk in chunked_data[2]]}, len val : {[len(chunk) for chunk in chunked_data[4]] if chunked_data[4] is not None else None}")
    print("----------------------------------------------------------------------------------------------------\n")

    n_classes = chunked_data[-1]
    
    new_chunkdist = os.path.join(output, f"{dataset}_chunkdist_{chunk_n}")

    if not os.path.exists(new_chunkdist) and return_data is False:
        os.mkdir(new_chunkdist)

    elif return_data is False:
        assert False, f"Directory {new_chunkdist} already exists, please remove it before continuing" 

    for i, _ in enumerate(tqdm(range(len(chunked_data[0])))):

        train_x = chunked_data[0][i]
        train_y = chunked_data[1][i]

        test_x = chunked_data[2][i] 
        test_y = chunked_data[3][i] 

        val_x = chunked_data[4][i] if chunked_data[4] is not None else None
        val_y = chunked_data[5][i] if chunked_data[5] is not None else None

        orig_train_y = chunked_data[6][i] if chunked_data[6] is not None else None
        orig_test_y = chunked_data[7][i] if chunked_data[7] is not None else None 
        orig_val_y = chunked_data[8][i] if chunked_data[8] is not None else None
    
        cleaned_train_x = preprocess_functions.clean_text(train_x)
        cleaned_val_x = preprocess_functions.clean_text(val_x)

        split_train_x, token_train_x, lemma_train_x = preprocess_functions.lemmatize(cleaned_train_x, lemmatizer="en_core_web_sm")
        split_val_x, token_val_x, lemma_val_x = preprocess_functions.lemmatize(cleaned_val_x, lemmatizer="en_core_web_sm") 

        token_ids_train_x = preprocess_functions.map_tokens(split_train_x, token_train_x)
        token_ids_val_x = preprocess_functions.map_tokens(split_val_x, token_val_x)

        train_data = pf.stage_data(cleaned_x=cleaned_train_x, 
                                      split_x=split_train_x, 
                                      token_x=token_train_x, 
                                      lemma_x=lemma_train_x, 
                                      token_ids_x=token_ids_train_x, 
                                      y=train_y, 
                                      orig_labels=orig_train_y,
                                      x=train_x)
                
        val_data = pf.stage_data(cleaned_x=cleaned_val_x,
                                    split_x=split_val_x,
                                    token_x=token_val_x,
                                    lemma_x=lemma_val_x,
                                    token_ids_x=token_ids_val_x,
                                    y=val_y,
                                    orig_labels=orig_val_y,
                                    x=val_x)

        test_data = [{"text" : text, "sentiment" : label, "orig_labels" : orig_test_y} for text, label in zip(test_x, test_y)]

        data = {"train": train_data, 
                "validation": val_data,
                "test" : test_data,
                "distributer" : dataset, 
                "n_classes" : n_classes}
        
        if return_data:
            return data

        else:
            pf.write_data(data=data, path=new_chunkdist)
    
    return None
        
    
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=dataset_checker, 
                        help="Dataset to stage")
    
    parser.add_argument("--input", type=input_checker, 
                        help="Path to input data, must be the absolute path to a valid directory where the datafiles are located.")
    
    parser.add_argument("--output", type=output_checker, 
                        help="Path to output data, must be a path to a directory that exists and is writable.")
    
    parser.add_argument("--chunk_n", type=int, help="Which chunk to stage")

    args = parser.parse_args()

    stage_preprocess(args.dataset, args.input, args.output, args.chunk_n)