import torch
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments

import chunking.get_data as gd
import xai_validation.helper_functions_xaival as hf_xaival
import config as config
import xai_transformer.helper_functions as hf
import config as config


def load_test(dataset):

    test = gd.dataset(dataset)(two_cat=True).load_test()
    
    new_test_x = hf_xaival.tokenize_to_model([text for text, _ in test], config.tokenizer, config.device)

    test_x = {"input_ids": [], "attention_mask": [], "targets": []}
    
    test_x = hf.extend_test(test_x, new_test_x)
    test_y = [label for _, label in test]

    test_x = hf.Test_Dataset(**test_x)

    return test_x, test_y


def accuracy(model_path, test_x, test_y):

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
    
    print("******************************")
    print("Sequence accuracy: ", seq_acc)
    print("******************************")


def model_accuracy(dataset, chunkdist_n):

    test_x, test_y = load_test(dataset)
    
    model_path = f"/home/bigtech/data/verbosius/amazon/models/amazon_model_dist_{chunkdist_n}/model"

    acc = accuracy(model_path, test_x, test_y)
    
    return acc

if __name__ == "__main__":

    dataset = ""
    chunkdist_n = -1

    model_accuracy(dataset, chunkdist_n)