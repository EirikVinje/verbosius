import argparse
import pickle
import os
import json
import time
import gzip
import logging
import gc

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

import config as config
import xai_transformer.helper_functions as hf
import arg_funcs as af
import xai_transformer.xai_model as xm


logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def stage_transformer(dataset : str, chunkdist_n : int):

    start_t = time.time()

    root = config.root
    
    model_folder, chunk_dist, chunks = hf.set_directory(root, dataset, chunkdist_n)

    device = config.device
    tokenizer = config.tokenizer
    num_tok_labels = config.num_tok_labels
    num_seq_labels = config.num_seq_labels
    neutral_weight = config.neutral_weight
    loss_weight = config.loss_weight
    learning_rate = config.learning_rate
    trainer_batch_size = config.trainer_batch_size
    num_train_epochs = config.num_train_epochs
    
    model = xm.CustomModel(num_tok_labels, num_seq_labels, neutral_weight, loss_weight)
    model = model.to(device = device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    progress_bar = tqdm(total=num_train_epochs * len(chunks), desc="")

    for i in range(num_train_epochs):
        
        progress_bar.set_description(f"Epoch {i+1}/{num_train_epochs}")

        for chunk in chunks:

            chunk_path = os.path.join(chunk_dist, chunk)

            with gzip.open(chunk_path, "rb") as f:
                chunk_data = pickle.load(f)

            train_tokenized = hf.tokenize_and_align_labels(chunk_data, tokenizer, orig_labels=True)
    
            train_dataset = hf.Dataset(**train_tokenized)

            train_dataloader = DataLoader(train_dataset, 
                                        batch_size=trainer_batch_size, 
                                        shuffle=True, 
                                        collate_fn=hf.custom_data_collator)

            model.train()

            for batch in train_dataloader:

                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                targets = batch["targets"].to(device)
                sentiment = batch["sentiment"].to(device)

                outputs = model(input_ids, attention_mask, labels, targets, sentiment)

                loss = outputs.loss
                loss.backward()
                optimizer.step()

            progress_bar.update(1)

            chunk_data = None
            train_tokenized = None
            train_dataset = None
            train_dataloader = None
            gc.collect()


        model_path = os.path.join(model_folder, f"model_epoch_{i}")
        torch.save(model, model_path)

    progress_bar.close()

    end_t = time.time()

    time_dict = {"time_hours" : (end_t - start_t) / 3600}
    
    with open(os.path.join(model_folder, "time.json"), "w") as f:
        json.dump(time_dict, f, indent=4)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=str, help="Dataset to train on")
    parser.add_argument("--chunkdist_n", type=int, help="Select chunkdist to train on")
    
    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunckdist_n_checker(args.chunkdist_n)
    
    stage_transformer(args.dataset, args.chunkdist_n)
