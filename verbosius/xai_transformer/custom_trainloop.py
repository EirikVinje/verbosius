import argparse
import pickle
import os
import json
import time
import gzip

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW

import config as config
import xai_transformer.helper_functions as hf
import arg_funcs as af
import xai_transformer.xai_model as xm


def stage_transformer(dataset : str, chunkdist_n : int):

    start_t = time.time()

    root = config.root
    seed = config.seed
    tokenizer = config.tokenizer

    models_folder = os.path.join(root, dataset, "models")
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    model_folder = os.path.join(models_folder, f"{dataset}_model_dist_{chunkdist_n}")
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    else:
        assert False, f"Directory {model_folder} already exists, please remove it before continuing"

    model_path = os.path.join(model_folder, "model")

    trainingdata_folder = os.path.join(root, dataset, "trainingdata")
    if not os.path.exists(trainingdata_folder):
        assert False, f"Trainingdata folder {trainingdata_folder} does not exist, please check your input"

    chunk_dist = os.path.join(trainingdata_folder, f"{dataset}_chunkdist_{chunkdist_n}")

    chunks = sorted(os.listdir(chunk_dist))
    all_train_data = []

    device = config.device
    tokenizer = config.tokenizer
    num_tok_labels = config.num_tok_labels
    num_seq_labels = config.num_seq_labels
    neutral_weight = config.neutral_weight
    loss_weight = config.loss_weight
    learning_rate = config.learning_rate
    per_device_train_batch_size = config.per_device_train_batch_size
    num_train_epochs = config.num_train_epochs
    evaluation_strategy = config.evaluation_strategy
    save_strategy = config.save_strategy
    load_best_model_at_end = config.load_best_model_at_end
    label_names = config.label_names

    model = xm.CustomModel(num_tok_labels, num_seq_labels, neutral_weight, loss_weight)
    model = model.to(device = device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for i in tqdm(range(num_train_epochs)):

        for chunk in chunks:

            chunk_path = os.path.join(chunk_dist, chunk)

            with gzip.open(chunk_path, "rb") as f:
                chunk_data = pickle.load(f)

            train_tokenized = hf.tokenize_and_align_labels(chunk_data, tokenizer, orig_labels=True)
    
            train_dataset = hf.Dataset(**train_tokenized)
            train_dataloader = DataLoader(train_dataset, batch_size=per_device_train_batch_size, shuffle=True, collate_fn=hf.custom_data_collator)

            model.train()

            for batch in tqdm(train_dataloader):

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--dataset", type=str, help="Dataset to train on")
    parser.add_argument("--chunkdist_n", type=int, help="Select chunkdist to train on")

    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunckdist_n_checker(args.chunkdist_n)
    
    stage_transformer(args.dataset, args.chunkdist_n)
