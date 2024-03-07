import argparse
import logging
import pathlib
import json
import os
import gc

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from transformers import Trainer, TrainingArguments, AutoModel
import matplotlib.pyplot as plt
import numpy as np
import torch

import xai_validation.helper_functions_xaival as hf_xaival
from xai_transformer.xai_model import CustomModel
import xai_transformer.helper_functions as hf
import chunking.get_data as gd
import config as config
import arg_funcs as af


logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def load_test(dataset, size):

    tokenizer = config.tokenizer
    device = config.device

    test = gd.dataset(dataset)(two_cat=True, size=size).load_test()

    new_test_x = hf_xaival.tokenize_to_model([text for text, _ in test], tokenizer, device)

    test_x = {"input_ids": [], "attention_mask": [], "targets": []}
    
    test_x = hf.extend_test(test_x, new_test_x)
    test_y = [label for _, label in test]

    test_x = hf.Test_Dataset(**test_x)

    return test_x, test_y


def accuracy(test_x, test_y, chunkdist_n, model_name, c):

    root = config.root
    dataset = config.dataset
    
    model_path = os.path.join(root, "models", f"{dataset}_model_dist_{chunkdist_n}")
    
    if not os.path.exists(model_path):
        assert False, f"Model path {model_path} does not exist."

    if c == 1:
        model = CustomModel(config.num_tok_labels, config.num_seq_labels, config.neutral_weight, config.loss_weight, os.path.join(model_path, model_name))
    
    else:
        model = torch.load(os.path.join(model_path, model_name))
    
    training_args = TrainingArguments(
        output_dir = "/home/bigtech/",
        per_device_train_batch_size = 64,
        per_device_eval_batch_size = 64,
        label_names = config.label_names,        
        )

    if config.device != "cpu":
        training_args = training_args.set_dataloader(pin_memory=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=config.tokenizer,
        compute_metrics=hf.compute_metrics,
        data_collator=hf.custom_data_collator)

    preds = trainer.predict(test_x)
    
    seq_logits = preds[0][1]
    seq_preds = np.argmax(seq_logits, axis=1)

    seq_acc = accuracy_score(test_y, seq_preds)
    seq_prec = precision_score(test_y, seq_preds, average="weighted")
    seq_f1 = f1_score(test_y, seq_preds, average="weighted")
    
    conf_mat = confusion_matrix(test_y, seq_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["1*", "2*", "3*", "4*", "5*"])
    disp.plot()

    plt.savefig(os.path.join(model_path, f"confusion_matrix_{model_name}.png"))

    metric_dict = {"seq_acc": seq_acc, "seq_prec": seq_prec, "seq_f1": seq_f1}

    with open(os.path.join(model_path, f"metrics_{model_name}.json"), "w") as f:
        json.dump(metric_dict, f)

    preds = None
    seq_logits = None
    seq_preds = None
    test_x = None
    test_y = None
    model = None

    gc.collect()


def model_accuracy(dataset, chunkdist_n, model_name, size, c):

    test_x, test_y = load_test(dataset, size)
    
    acc = accuracy(test_x, test_y, chunkdist_n, model_name, c)
    
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model performance")

    parser.add_argument("--id", type=int, help="Chunkdist number")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--c", type=int, help="Size of dataset")
    
    args = parser.parse_args()

    dataset = "amazon"
    size = "big"
    model_name = args.model
    chunkdist_n = args.id
    c = args.c

    model_accuracy(dataset, chunkdist_n, model_name, size, c)
    
    
