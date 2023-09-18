import os
import torch

from transformers import TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import accuracy_score

import config as config
import xai_transformer.xai_model as xm
import xai_transformer.helper_functions as hf

def transformer_pipeline(output_dir, train_data, test_x, test_y, val_data):
    
    device = config.device
    learning_rate = config.learning_rate
    per_device_train_batch_size = config.per_device_train_batch_size
    per_device_eval_batch_size = config.per_device_eval_batch_size
    num_train_epochs = config.num_train_epochs
    weight_decay = config.weight_decay
    evaluation_strategy = config.evaluation_strategy
    save_strategy = config.save_strategy
    warmup_steps = config.warmup_steps
    load_best_model_at_end = config.load_best_model_at_end
    eval_accumulation_steps = config.eval_accumulation_steps
    label_names = config.label_names
    tokenizer = config.tokenizer

    neutral_weight = config.neutral_weight
    loss_weight = config.loss_weight
    num_labels = config.num_labels
    num_seq_labels = config.num_seq_labels

    model = xm.CustomModel(num_labels, num_seq_labels, neutral_weight, loss_weight)
    model = model.to(device = device)

    train_data = hf.Dataset(**train_data)
    test_x = hf.Test_Dataset(**test_x)

    if type(val_data) != type(None):
        val_data = hf.Dataset(**val_data)
    
    else:
        val_data = None


    training_args = TrainingArguments(
        output_dir = output_dir,
        learning_rate = learning_rate,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        num_train_epochs = num_train_epochs,
        evaluation_strategy = evaluation_strategy,
        save_strategy = save_strategy,
        warmup_steps = warmup_steps,
        load_best_model_at_end = load_best_model_at_end,
        eval_accumulation_steps = eval_accumulation_steps,
        label_names = label_names,
        )

    if device != "cpu":
        training_args = training_args.set_dataloader(pin_memory=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        compute_metrics=hf.compute_metrics,
        data_collator=hf.custom_data_collator)

    trainer.train()
    
    preds = trainer.predict(test_x)
    seq_logits = preds[0][1]
    seq_preds = np.argmax(seq_logits, axis=1)

    seq_acc = accuracy_score(test_y, seq_preds)

    print("Sequence accuracy: ", seq_acc)

    os.system(f"rm -rf {output_dir}")
    #torch.save(model, output_dir)

    return seq_acc
