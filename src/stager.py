import pickle
import os

import preprocess as preprocess
import datasource as datasource
import stage as stage


def main():

    imdb = datasource.dataset("imdb")

    imdb = imdb(two_cat=True, batch=(0, 1000))

    imdb.load_data(path="path/to/data") 
    
    return

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

    
if __name__ == "__main__":

    main()



    