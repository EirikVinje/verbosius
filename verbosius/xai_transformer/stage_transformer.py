import argparse
import pickle
import os
import json
import datetime

import config as config
import chunking.chunker_functions as cf
import xai_transformer.helper_functions as hf
import xai_transformer.transformer as tf
import xai_validation.helper_functions_xaival as hf_xaival


def stage_transformer(dataset : str, train_val_input : str, test_input : str, model_output : str, chunkdist_n : int, return_seq_acc : bool = True):

    """
    
    Train transformer on staged trainingdata

    Parameters
    ----------
    dataset : str
        Name of dataset to stage trainingdata for
    
    input : str
        Path to batchdistros of dataset, must be the absolute path to a valid directory where the datafiles are located.
    
    output : str
        Path to output model, must be a path to a directory that exists and is writable.
    
    save_model : str
        Save model or not, either 'true' or 'false'.
    
    batchdist_n : tuple
        Batchdist_n to stage trainingdata for, must use tuple interval, e.g (0,-1) is all batchdistros

    """
    
    ds = cf.dataset(dataset)
    ds = ds(two_cat=True)

    test = ds.load_test()

    model_dir = os.path.join(model_output, f"{dataset}_model_dist_{chunkdist_n}")
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    else:
        assert False, f"Directory {model_dir} already exists, please remove it before continuing"
    
    model_path = os.path.join(model_dir, "model")

    trainingdata_dist = os.path.join(train_val_input, f"{dataset}_chunkdist_{chunkdist_n}", "train_val")
    
    chunks = sorted(os.listdir(trainingdata_dist))
    
    train_data = {"input_ids": [], "attention_mask": [], "labels": [], "targets": [], "sentiment": []}
    val_data = {"input_ids": [], "attention_mask": [], "labels": [], "targets": [], "sentiment": []}
    
    for _, chunk in enumerate(chunks):
        
        chunk = os.path.join(trainingdata_dist, chunk)

        train_val = pickle.load(open(chunk, "rb"))
        
        new_train_batch = hf.tokenize_and_align_labels(train_val["train"], config.tokenizer) 
        new_val_batch = hf.tokenize_and_align_labels(train_val["validation"], config.tokenizer)
        
        train_data = hf.extend_data(train_data, new_train_batch)
        val_data = hf.extend_data(val_data, new_val_batch)  
    
    test_x = {"input_ids": [], "attention_mask": [], "targets": []}
    test_y = []

    # testdata_chunkdist = os.path.join(test_input, f"{dataset}_chunkdist_{chunkdist_n}", "test")
    # test_chunks = sorted(os.listdir(testdata_chunkdist))

    # for _, chunk in enumerate(test_chunks):

        # chunk = os.path.join(testdata_chunkdist, chunk)
        # test = pickle.load(open(chunk, "rb"))

    new_test_x = hf_xaival.tokenize_to_model([text for text, _ in test], config.tokenizer, config.device)

    test_x = hf.extend_test(test_x, new_test_x)
    test_y = [label for _, label in test]

    print()    
    print("Train size: ", len(train_data["input_ids"]))
    print("Test size: ", len(test_x["input_ids"]))
    print("Validation size: ", len(val_data["input_ids"])) if val_data["input_ids"] != [] else None
    print()
    
    seq_acc = tf.transformer_pipeline(output_dir=model_path, 
                                               train_data=train_data, 
                                               val_data=val_data, 
                                               test_x=test_x,
                                               test_y=test_y)
    
    if return_seq_acc:
        return seq_acc

    meta_model = {"seq_acc": seq_acc,
                "dist" : chunkdist_n,
                "time_finished" : str(datetime.datetime.now())}

    os.system(f"git add --all")
    os.system(f"git commit -m 'new model trained'")
    os.system(f"git push origin HEAD")

    with open(os.path.join("/home/bigtech/projects/verbosius/model_logs", f"meta_model_{chunkdist_n}.json"), "w") as f:
        json.dump(meta_model, f)

    return None
    


def dataset_checker(dataset):
    valid_datasets = ['imdb', 'rottentomatoes', 'amazon']
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


def chunkdist_checker(dataset, input, chunkdist_n):
    if not os.path.exists(os.path.join(input, f"{dataset}_chunkdist_{chunkdist_n}")):
        raise argparse.ArgumentTypeError(f"Invalid chunk dist, {dataset}_chunkdist_{chunkdist_n} does not exist") 

    return chunkdist_n

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=str, help="Dataset to train on")
    parser.add_argument("--input_traindata", type=str, help="train and val data path")
    parser.add_argument("--input_testdata", type=str, help="test data path")
    parser.add_argument("--model_output", type=str, help="Path to output model, must be a path to a directory that exists and is writable.")
    parser.add_argument("--chunkdist_n", type=int, help="Select chunkdist to train on")

    args = parser.parse_args()

    dataset_checker(args.dataset)
    input_checker(args.input_traindata)
    input_checker(args.input_testdata)
    output_checker(args.model_output)
    chunkdist_checker(args.dataset, args.input_traindata, args.chunkdist_n)

    stage_transformer(args.dataset, args.input_traindata, args.input_testdata, args.model_output, args.chunkdist_n)
