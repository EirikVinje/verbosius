import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import datasets

import preprocessing.preprocess_functions as preprocess_functions
import config as config
import xai_validation.helper_functions_xaival as vf


def stage_validation(model_path : str, model_name : str, batch_size_pred : int):

    ds = datasets.load_dataset("rotten_tomatoes")
    
    train_x = ds["train"]["text"]
    test_x = ds["test"]["text"]
    train_y = ds["train"]["label"]
    test_y = ds["test"]["label"]
    
    train_y = np.array(train_y).astype(int)
    test_y = np.array(test_y).astype(int)
    
    model_path = os.path.join(model_path, model_name, "model")
    
    model = torch.load(model_path)

    token_preds, input_ids, attention_masks = vf.get_prediction_outputs(model, train_x, batch_size_pred)
    
    vocabulary = vf.make_vocabulary(token_preds, input_ids, attention_masks, config.tokenizer)

    print("LENGTH VOCABULARY: ", len(vocabulary))
    print("VOCABULARY: ", vocabulary)


    vectorizer = CountVectorizer(binary=True, vocabulary=vocabulary)
    
    train_x_bin = vectorizer.fit_transform(train_x)
    test_x_bin = vectorizer.transform(test_x)

    logreg = LogisticRegression(verbose=1, 
                                max_iter=1000, 
                                penalty='l2', 
                                random_state=42, 
                                C=0.092705530127623, 
                                tol=0.748258213506498)
    
    logreg.fit(train_x_bin, train_y)
    
    log_res =  accuracy_score(test_y, logreg.predict(test_x_bin))
    
    print("-----------------------------------------------------------")
    print("accuracy / explanation score: ", round(log_res, 4))
    print("-----------------------------------------------------------")

    return log_res


def input_checker(input):
    if os.access(os.path.dirname(input), os.W_OK) and os.path.isdir(input):
        return input
    else:
        raise argparse.ArgumentTypeError(f'Invalid input path, "{input}" is not writable or is not a directory')


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--model_path", type=input_checker, required=True, 
                        help="Path to model")
    
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--batch_size_pred", type=int, required=True)

    args = parser.parse_args()

    stage_validation(args.model_path, args.model_name, args.batch_size_pred)
    