import os
import torch
import gc
import logging

from transformers import TrainingArguments, Trainer
import torch

import config as config
import xai_transformer.xai_model as xm
import xai_transformer.helper_functions as hf


logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def transformer_pipeline_custom(output_dir, train_data, val_data):
    
    model = xm.CustomModel(config.num_labels, config.num_seq_labels, config.neutral_weight, config.loss_weight)
    model = model.to(device = config.device)

    train_data = hf.Dataset(**train_data)
    val_data = hf.Dataset(**val_data)
    
    training_args = TrainingArguments(
        output_dir = output_dir,
        learning_rate = config.learning_rate,
        per_device_train_batch_size = config.per_device_train_batch_size,
        per_device_eval_batch_size = config.per_device_train_batch_size,
        num_train_epochs = config.num_train_epochs,
        evaluation_strategy = config.evaluation_strategy,
        save_strategy = config.save_strategy,
        load_best_model_at_end = config.load_best_model_at_end,
        label_names = config.label_names
        )

    if config.device != "cpu":
        training_args = training_args.set_dataloader(pin_memory=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=config.tokenizer,
        compute_metrics=hf.compute_metrics,
        data_collator=hf.custom_data_collator)

    print("***********************")
    print(f"batch size: {config.per_device_train_batch_size}")
    print(f"learning rate: {config.learning_rate}")
    print(f"epochs: {config.num_train_epochs}")
    print("***********************")
    
    trainer.train()

    os.system(f"rm -rf {output_dir}")
    torch.save(model, output_dir)

    del train_data
    del val_data
    del model
    
    gc.collect()
    
    torch.cuda.empty_cache()

