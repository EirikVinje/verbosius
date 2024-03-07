import argparse
import logging
import pickle
import os
import json
import time
import gzip
import gc

from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from xai_transformer.helper_functions import custom_data_collator, IterableDataset, set_directory, compute_metrics
from xai_transformer.xai_model import CustomModel
import config as config
import arg_funcs as af


logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def stage_transformer(dataset : str, chunkdist_n : int):

    """
    Train transformer on weigthed trainingdata.

    Parameters
    ----------
    dataset : str
        Name of dataset to train on.

    chunkdist_n : int
        ID of chunkdist to use for trainingdata. Must be an integer.
    
    """
    
    start_t = time.time()

    chunkdist_name = f"{dataset}_chunkdist_{chunkdist_n}"

    set_directory(config.root, chunkdist_name)

    trainingdata_path = os.path.join(config.root, "trainingdata", chunkdist_name)
    chunks_list = os.listdir(trainingdata_path)

    dataset = IterableDataset(chunks_list, trainingdata_path, config.seed)

    # dataloader = DataLoader(dataset=dataset, 
    #                         batch_size=config.trainer_batch_size, 
    #                         collate_fn=custom_data_collator)
    
    model = CustomModel(config.num_tok_labels, config.num_seq_labels, config.neutral_weight, config.loss_weight).to(config.device)
    
    model_dir = os.path.join(config.root, "models", chunkdist_name)
    
    training_args = TrainingArguments(
        output_dir = model_dir,
        learning_rate = config.learning_rate,
        per_device_train_batch_size = config.trainer_batch_size,
        per_device_eval_batch_size = config.trainer_batch_size,
        num_train_epochs = config.num_train_epochs,
        evaluation_strategy = config.evaluation_strategy,
        save_strategy = config.save_strategy,
        load_best_model_at_end = config.load_best_model_at_end,
        label_names = config.label_names,
        )

    if config.device != "cpu":
        training_args = training_args.set_dataloader(pin_memory=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=config.tokenizer,
        compute_metrics=compute_metrics,
        data_collator=custom_data_collator)

    trainer.train()

    torch.save(model, os.path.join(model_dir, "model_t"))

    del train_data
    del val_data
    del model
    
    gc.collect()
    torch.cuda.empty_cache()

    end_t = time.time()

    time_dict = {"time_hours" : (end_t - start_t) / 3600}
    
    with open(os.path.join(config.root, "models", chunkdist_name, "time.json"), "w") as f:
        json.dump(time_dict, f, indent=4)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=str, help="Dataset to train on")
    parser.add_argument("--chunkdist_n", type=int, help="Select chunkdist to train on")

    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunckdist_n_checker(args.chunkdist_n)
    
    stage_transformer(args.dataset, args.chunkdist_n)
