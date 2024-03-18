import argparse
import pickle
import os
import json
import time
import gc
import gzip

from transformers import TrainingArguments, Trainer
from tqdm import tqdm

from xai_transformer.helper_functions import set_directory, compute_metrics, custom_data_collator, Dataset, convert
from xai_transformer.xai_model import CustomModel
import config as config
import arg_funcs as af
import torch

def stage_transformer(dataset : str, chunkdist_n : int):

    """
    Train transformer on weigthed trainingdata.

    Parameters
    ----------
    dataset : str
        Name of dataset to train on.

    train_val_input : str
        Path to trainingdata. Must be absolute path to directory.
    
    test_input : str
        Path to testdata. Must be absolute path to directory.
    
    model_output : str
        Path to output of this module. Must be absolute path to directory.
    
    chunkdist_n : int
        ID of chunkdist to use for trainingdata. Must be an integer.
    
    return_seq_acc : bool
        If True, returns the sequence accuracy of the trained model. If False, returns None.

    """
    
    start_t = time.time()

    chunkdist_name = f"{dataset}_chunkdist_{chunkdist_n}"
    set_directory(config.root, chunkdist_name)

    model_path = os.path.join(config.root, "models", chunkdist_name)
    train_path = os.path.join(config.root, "train_eval_tokenize", chunkdist_name, "train")
    eval_path = os.path.join(config.root, "train_eval_tokenize", chunkdist_name, "eval")

    train_chunks = sorted(os.listdir(train_path))
    train = []

    with tqdm(total=len(train_chunks), desc="trainsize: 0 ") as pbar:
        for _, chunk in enumerate(train_chunks):        

            chunk = os.path.join(train_path, chunk)
            with gzip.open(chunk, "rb") as f:
                chunk = pickle.load(f)
            train.extend(chunk)
            pbar.update(1)
            pbar.set_description(f"trainsize: {len(train)} ")
            
    eval_chunks = sorted(os.listdir(eval_path))
    eval = []

    with tqdm(total=len(eval_chunks), desc="evalsize : 0 ") as pbar:
        
        for _, chunk in enumerate(eval_chunks):
            chunk = os.path.join(eval_path, chunk)

            with gzip.open(chunk, "rb") as f:
                chunk = pickle.load(f)

            eval.extend(chunk)
            pbar.update(1)
            pbar.set_description(f"evalsize : {len(eval)} ")

    train = convert(train)
    eval = convert(eval)

    model = CustomModel(config.num_tok_labels, config.num_seq_labels, config.neutral_weight, config.loss_weight)
    model = model.to(device = config.device)

    train_data = Dataset(**train)
    val_data = Dataset(**eval)
    
    training_args = TrainingArguments(
        output_dir = model_path,
        learning_rate = config.learning_rate,
        per_device_train_batch_size = config.trainer_batch_size,
        per_device_eval_batch_size = config.trainer_batch_size,
        num_train_epochs = config.num_train_epochs,
        evaluation_strategy = config.evaluation_strategy,
        save_strategy = config.save_strategy,
        load_best_model_at_end = config.load_best_model_at_end,
        label_names = config.label_names
        )

    if config.device != "cpu":
        training_args = training_args.set_dataloader(pin_memory=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=config.tokenizer,
        compute_metrics=compute_metrics,
        data_collator=custom_data_collator)

    trainer.train()

    # os.system(f"rm -rf {os.path.join(model_path, "*")}")
    torch.save(model, os.path.join(model_path, "model_t"))

    del train_data
    del val_data
    del model
    
    gc.collect()
     
    torch.cuda.empty_cache()

    end_t = time.time()

    time_dict = {"time_hours" : (end_t - start_t) / 3600}
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=str, help="Dataset to train on")
    parser.add_argument("--chunkdist_n", type=int, help="Select chunkdist to train on")

    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunckdist_n_checker(args.chunkdist_n)
    
    stage_transformer(args.dataset, args.chunkdist_n)