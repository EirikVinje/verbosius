from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import datasets
import torch
import evaluate
import numpy as np


def main():

    df = datasets.load_dataset("rotten_tomatoes")

    train_x = df["train"]["text"]
    test_x = df["test"]["text"]
    train_y = df["train"]["label"]
    test_y = df["test"]["label"]

    train_x = train_x[:50]
    test_x = test_x[:25]
    train_y = train_y[:50]
    test_y = test_y[:25]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)

    train_x_tokenized = tokenizer(train_x, padding=True, truncation=True, return_tensors="pt")
    test_x_tokenized = tokenizer(test_x, padding=True, truncation=True, return_tensors="pt")

    train_y = torch.tensor(train_y)
    test_y = torch.tensor(test_y)

    train_data = [{"input_ids" : j, "attention_mask" : k, "labels" : l} for j, k, l in zip(train_x_tokenized["input_ids"], train_x_tokenized["attention_mask"], train_y)]
    test_data = [{"input_ids" : j, "attention_mask" : k, "labels" : l} for j, k, l in zip(test_x_tokenized["input_ids"], test_x_tokenized["attention_mask"], test_y)]

    training_args = TrainingArguments(
    output_dir='/home/kolla/data/dump/',          
    num_train_epochs=1,              
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,    
    warmup_steps=500,                
    weight_decay=0.01,               
    evaluation_strategy="no",
    eval_steps=5,
    greater_is_better=True,
    save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=metrics,
        )

    trainer.train()

    preds = trainer.predict(test_data)
    print(preds)
    
    
    trainer.eval_dataset = test_data

    res = trainer.evaluate()
    print(res)


def metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



if __name__ == "__main__":
    main()