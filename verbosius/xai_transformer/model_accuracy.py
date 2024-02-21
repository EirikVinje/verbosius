import os
import argparse
import json

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt

import chunking.get_data as gd
import xai_validation.helper_functions_xaival as hf_xaival
import xai_transformer.helper_functions as hf
import config as config
import arg_funcs as af


def load_test(dataset):

    tokenizer = config.tokenizer
    device = config.device

    test = gd.dataset(dataset)(two_cat=True).load_test()

    new_test_x = hf_xaival.tokenize_to_model([text for text, _ in test], tokenizer, device)

    test_x = {"input_ids": [], "attention_mask": [], "targets": []}
    
    test_x = hf.extend_test(test_x, new_test_x)
    test_y = [label for _, label in test]

    test_x = hf.Test_Dataset(**test_x)

    return test_x, test_y


def accuracy(test_x, test_y, chunkdist_n):

    root = config.root
    dataset = config.dataset
    
    model_path = os.path.join(root, dataset, "models", f"{dataset}_model_dist_{chunkdist_n}")
    model_file = os.path.join(model_path, "model")
    
    if not os.path.exists(model_file):
        assert False, f"Model path {model_file} does not exist."

    model = torch.load(model_path)
    
    label_names = config.label_names
    device = config.device
    tokenizer = config.tokenizer

    training_args = TrainingArguments(
        output_dir = "/home/bigtech/",
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        per_gpu_train_batch_size= 32,
        per_gpu_eval_batch_size= 32,
        label_names = label_names,        
        )

    if device != "cpu":
        training_args = training_args.set_dataloader(pin_memory=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=hf.compute_metrics,
        data_collator=hf.custom_data_collator)

    preds = trainer.predict(test_x)
    
    seq_logits = preds[0][1]
    seq_preds = np.argmax(seq_logits, axis=1)

    seq_acc = accuracy_score(test_y, seq_preds)
    seq_prec = precision_score(test_y, seq_preds)
    seq_f1 = f1_score(test_y, seq_preds, average="weighted")
    
    print("******************************")
    print("Sequence accuracy: ", seq_acc)
    print("Sequence precision: ", seq_prec)
    print("Sequence f1: ", seq_f1)
    print("******************************")

    conf_mat = confusion_matrix(test_y, seq_preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["1*", "2*", "3*", "4*", "5*"])
    disp.plot()

    plt.savefig(os.path.join(model_path, "confusion_matrix.png"))

    metric_dict = {"seq_acc": seq_acc, "seq_prec": seq_prec, "seq_f1": seq_f1}

    with open(os.path.join(model_path, "metrics.json"), "w") as f:
        json.dump(metric_dict, f)

    return seq_acc


def model_accuracy(dataset, chunkdist_n):

    test_x, test_y = load_test(dataset)
    
    acc = accuracy(test_x, test_y, chunkdist_n)
    
    return acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--chunkdist_n", type=int, default=1)
    args = parser.parse_args()

    af.chunckdist_n_checker(args.chunkdist_n)
    af.dataset_checker(args.dataset)

    acc = model_accuracy(args.dataset, args.chunkdist_n)
    
    print("******************************")
    print("Sequence accuracy: ", acc)
    print("******************************")

