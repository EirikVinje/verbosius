import os

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification
import evaluate

import xai_transformer.helper_functions as hf
import config as config


class CustomTrainer_Tokenclassifier(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        labels = inputs.get("labels")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0, 1.0]).to(config.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def transformer_pipeline_tokenclassifier(output_dir, train_data, val_data):
    
    """
    Runs a tokenclassifier.

    Parameters:
    output_dir (str): Path to output directory.
    train_data (dict): Dictionary containing training data.
    val_data (dict): Dictionary containing validation data.
    """

    device = config.device
    learning_rate = config.learning_rate
    per_device_train_batch_size = config.per_device_train_batch_size
    per_device_eval_batch_size = config.per_device_eval_batch_size
    num_train_epochs = config.num_train_epochs
    evaluation_strategy = config.evaluation_strategy
    save_strategy = config.save_strategy
    load_best_model_at_end = config.load_best_model_at_end
    tokenizer = config.tokenizer

    num_seq_labels = 3

    model = AutoModelForTokenClassification.from_pretrained(config.model_name_, num_labels=num_seq_labels)

    train_data = hf.Dataset(**train_data)
    val_data = hf.Dataset(**val_data)
    
    training_args = TrainingArguments(
        output_dir = output_dir,
        learning_rate = learning_rate,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        num_train_epochs = num_train_epochs,
        evaluation_strategy = evaluation_strategy,
        save_strategy = save_strategy,
        load_best_model_at_end = load_best_model_at_end,
        )

    if device != "cpu":
        training_args = training_args.set_dataloader(pin_memory=False)

    trainer = CustomTrainer_Tokenclassifier(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=custom_data_collator_tokenclassifier,
        compute_metrics=compute_metrics_tokenclassifier,
        )

    trainer.train()

    os.system(f"rm -rf {output_dir}")
    torch.save(model, output_dir)


def custom_data_collator_tokenclassifier(batch_input):

    input_ids = [torch.tensor(inst["input_ids"], dtype=torch.long) for inst in batch_input]
    attention_mask = [torch.tensor(inst["attention_mask"], dtype=torch.long) for inst in batch_input]
    labels = [torch.tensor(inst["labels"], dtype=torch.long) for inst in batch_input] if "labels" in batch_input[0].keys() else None

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    if labels != None:
        labels = pad_sequence(labels, batch_first=True, padding_value=-100) 
    
    if labels == None:

        new_batch_input = {
        "input_ids": input_ids.to(config.device),
        "attention_mask": attention_mask.to(config.device)
        }

        return new_batch_input
    
    new_batch_input = {
        "input_ids": input_ids.to(config.device),
        "attention_mask": attention_mask.to(config.device),
        "labels": labels.to(config.device)
    }

    return new_batch_input


def compute_metrics_tokenclassifier(p):
    
    seqeval = evaluate.load("seqeval")
    
    label_list = ["neutral", "positive", "negative"]
    
    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)

    true_predictions = [

        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]

        for prediction, label in zip(predictions, labels)

    ]

    true_labels = [

        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]

        for prediction, label in zip(predictions, labels)

    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {

        "precision": results["overall_precision"],

        "recall": results["overall_recall"],

        "f1": results["overall_f1"],

        "accuracy": results["overall_accuracy"],

    }


