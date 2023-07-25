import os
import torch

from transformers import TrainingArguments, Trainer

import exp_transformer.helper_functions as hf
import config as config

def transformer_pipeline(output_dir, train_data, test_data, save_model, tokenizer):
    
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

    neutral_weight = config.neutral_weight
    loss_weight = config.loss_weight
    num_labels = config.num_labels
    num_seq_labels = config.num_seq_labels

    model = hf.CustomModel(num_labels, num_seq_labels, neutral_weight, loss_weight)
    model = model.to(device = device)

    

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
        label_names = label_names)

    training_args = training_args.set_dataloader(pin_memory=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        compute_metrics=hf.compute_metrics,
        data_collator=hf.custom_data_collator)

    trainer.train()
    
    res = trainer.evaluate()
    
    if not save_model:
        os.system(f"rm -rf {output_dir}")


    return  res["eval_sequence_accuracy"], res["eval_token_accuracy"]