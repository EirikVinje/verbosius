import pickle
import os
import argparse

import verbosius.preprocessing.preprocess as preprocess
import verbosius.preprocessing.datasource as datasource
import verbosius.preprocessing.stage as stage


def main(dataset:str, batch:tuple, input:str, output:str):

    imdb = datasource.dataset("imdb")

    imdb = imdb(two_cat=True, batch=(0, 1000))
    
    imdb.load_data(path="path/to/data") 
    
    
    # clean the data from unwanted symbols and such
    cleaned_train_x = preprocess.clean_text(train_x)
    cleaned_test_x = preprocess.clean_text(test_x)

    # lemmatize the data
    split_train_x, token_train_x, lemma_train_x = preprocess.lemmatize(cleaned_train_x)
    split_test_x, token_test_x, lemma_test_x = preprocess.lemmatize(cleaned_test_x)

    # get token maps
    token_ids_train_x = preprocess.map_tokens(split_train_x, token_train_x)
    token_ids_test_x = preprocess.map_tokens(split_test_x, token_test_x)

    # stage data
    train_data = stage.stage_data(cleaned_train_x, split_train_x, token_train_x, lemma_train_x, token_ids_train_x, train_y)
    test_data = stage.stage_data(cleaned_test_x, split_test_x, token_test_x, lemma_test_x, token_ids_test_x, test_y)

    # write data
    stage.write_data(train_data, path="data", name="imdb_train")
    stage.write_data(test_data, path="data", name="imdb_test")

    
    # TODO : python stager.py --dataset imdb --batch 0 1000 --input path/to/input --output path/to/stageroutput



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=str, help="Dataset to stage")
    parser.add_argument("--batch", type=int, nargs=2, help="Batch of data to stage, two integers")
    parser.add_argument("--input", type=str, help="Path to input data")
    parser.add_argument("--output", type=str, help="Path to output data")

    args = parser.parse_args()

    main(args.dataset.lower(), tuple(args.batch), args.input, args.output)


    