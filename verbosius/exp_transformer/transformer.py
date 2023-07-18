import os

from transformers import TrainingArguments, Trainer

import verbosius.exp_transformer.helper_functions as hf


def transformer_pipeline(device, 
                         output_dir, 
                         learning_rate, 
                         per_device_train_batch_size, 
                         per_device_eval_batch_size, 
                         num_train_epochs, 
                         weight_decay, 
                         evaluation_strategy, 
                         save_strategy,
                         warmup_steps, 
                         load_best_model_at_end, 
                         eval_accumulation_steps, 
                         label_names,
                         train_data,
                         test_data,
                         tokenizer,
                         save_model):
    
    model = hf.CustomModel()
    model = model.to(device = device)

    training_args = TrainingArguments(
        output_dir = output_dir,
        learning_rate = learning_rate,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        num_train_epochs = num_train_epochs,
        weight_decay = weight_decay,
        evaluation_strategy = evaluation_strategy,
        save_strategy = save_strategy,
        warmup_steps = warmup_steps,
        load_best_model_at_end = load_best_model_at_end,
        eval_accumulation_steps = eval_accumulation_steps,
        label_names = label_names
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        compute_metrics=hf.compute_metrics
    )

    trainer.train()
    res = trainer.evaluate()
    
    if not save_model:
        os.system(f"rm -rf {output_dir}")
