import argparse
import logging
import pathlib
import json
import os
import gc
import time

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

    rng = np.random.default_rng(seed=config.seed)

    test = gd.dataset(dataset)(two_cat=True, size=size).load_test()

    rng.shuffle(test)

    test = test[:10000]

    new_test_x = hf_xaival.tokenize_to_model([text for text, _ in test], config.tokenizer, config.device)

    test_x = {"input_ids": [], "attention_mask": [], "targets": []}
    
    test_x = hf.extend_test(test_x, new_test_x)
    test_y = [label for _, label in test]

    test_x = hf.Test_Dataset(**test_x)

    return test_x, test_y


def make_trainer(model):

    training_args = TrainingArguments(
        output_dir = "/home/bigtech/",
        per_device_train_batch_size = 64,
        per_device_eval_batch_size = 64,
        label_names = config.label_names,        
        )

    if config.device != "cpu":
        training_args = training_args.set_dataloader(pin_memory=False)

    trainer = Trainer(model=model, args=training_args)

    return trainer


def make_confusion_matrix(test_y, seq_preds, model_path, model_name):

    conf_mat = confusion_matrix(test_y, seq_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["1*", "2*", "3*", "4*", "5*"])
    disp.plot()
    plt.savefig(os.path.join(model_path, f"confusion_matrix_{model_name}.png"))
    

def save_metrics(seq_acc, seq_prec, seq_f1, model_path, model_name):

    metric_dict = {"seq_acc": seq_acc, "seq_prec": seq_prec, "seq_f1": seq_f1}

    with open(os.path.join(model_path, f"metrics_{model_name}.json"), "w") as f:
        json.dump(metric_dict, f)


def accuracy(test_x, test_y, chunkdist_n, model_name):

    root = config.root
    dataset = config.dataset
    
    model_path = os.path.join(root, "models", f"{dataset}_chunkdist_{chunkdist_n}")
    
    if not os.path.exists(model_path):
        assert False, f"Model path {model_path} does not exist."
    
    model = torch.load(os.path.join(model_path, model_name))
    
    trainer = make_trainer(model)

    preds = trainer.predict(test_x)
    
    seq_logits = preds[0][1]
    seq_preds = np.argmax(seq_logits, axis=1)

    seq_acc = accuracy_score(test_y, seq_preds)
    seq_prec = precision_score(test_y, seq_preds, average="weighted")
    seq_f1 = f1_score(test_y, seq_preds, average="weighted")


    path = "/home/bigtech/projects/verbosius/model_metrics"
    folder = f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    path_folder = os.path.join(path, folder)
    os.mkdir(path_folder)

    make_confusion_matrix(test_y, seq_preds, path_folder, model_name)
    save_metrics(seq_acc, seq_prec, seq_f1, path_folder, model_name)

    gitsave = f"git add {path_folder} && git commit -m 'model metrics' && git push origin HEAD"
    os.system(gitsave)
    
    model = None
    test_x = None
    test_y = None
    preds = None
    seq_logits = None
    seq_preds = None
    gc.collect()
    
    return [seq_acc, seq_prec, seq_f1]
    

def model_accuracy(dataset, chunkdist_n, model_name, size):

    test_x, test_y = load_test(dataset, size)
    
    acc = accuracy(test_x, test_y, chunkdist_n, model_name)
    
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model performance")

    parser.add_argument("--dataset", type=str, help="dataset")
    parser.add_argument("--chunkdist_n", type=int, help="chunkdist number")
    parser.add_argument("--size", type=str, help="size of dataset")
    
    args = parser.parse_args()

    dataset = args.dataset
    size = args.size
    model_name = "model_t"
    chunkdist_n = args.chunkdist_n

    metrics = model_accuracy(dataset, chunkdist_n, model_name, size)
    
    print(metrics)
