import evaluate
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

import config as config


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels, targets, sentiment):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.targets = targets
        self.sentiment = sentiment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = self.labels[idx]
        targets = self.targets[idx]
        sentiment = self.sentiment[idx]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'targets': targets,
            'sentiment': sentiment
        }


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits[1], axis=1)
    token_predictions = np.argmax(logits[0], axis=2)
    
    nz = (predictions!=0).sum()
    token_negative = (token_predictions==1).sum()
    token_neutral = (token_predictions==0).sum()
    token_positive = (token_predictions==2).sum()

    seqeval = evaluate.load("seqeval")
    
    label_list = ["neutral", "positive", "negative"]

    true_predictions = [

        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]

        for prediction, label in zip(token_predictions, labels[0])

    ]

    true_labels = [

        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]

        for prediction, label in zip(token_predictions, labels[0])

    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    output = {
        "token_accuracy": results["overall_accuracy"],
        "sequence_accuracy": (metric.compute(predictions=predictions, references=labels[1]))['accuracy'],
        "nz": nz,
        "token_neutral": token_neutral,
        "token_negative": token_negative,
        "token_positive": token_positive
    }
    #df = pd.DataFrame(output, index=[0])
    #df.to_csv("eval_results.csv", mode="a", header=not os.path.exists("eval_results.csv"))
    return output


def tokenize_and_align_labels(data, tokenizer, orig_labels:bool = False):
    
    if type(data) == type(None):
        return None

    if orig_labels:
        Y = np.array([i['orig_label'] for i in data])
    else:
        Y = np.array([i['sentiment'] for i in data])

    tokenized_inputs = tokenizer([inst["tokens"] for inst in data], truncation=True, padding=False, is_split_into_words=True)
    
    labels = []
    targets = []
    
    for i, label in enumerate([inst["labels"] for inst in data]):
        
        word_ids = tokenized_inputs.word_ids(batch_index=i)  
        
        previous_word_idx = None

        label_ids = []
        target_ids = []
        
        for word_idx in word_ids:  

            if word_idx is None:
                target_ids.append(0)
                label_ids.append(-100)

            elif word_idx != previous_word_idx:  
                target_ids.append(1)
                label_ids.append(label[word_idx])

            else:
                target_ids.append(0)
                label_ids.append(-100)

            previous_word_idx = word_idx

        targets.append(target_ids)
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    tokenized_inputs["targets"] = targets

    output = {}
    output["input_ids"] = tokenized_inputs["input_ids"] 
    output["attention_mask"] = tokenized_inputs["attention_mask"]
    output["labels"] = tokenized_inputs["labels"]
    output["targets"] = tokenized_inputs["targets"]
    output["sentiment"] = Y
    
    return output


def extend_data(data, new_chunk):

    if type(data) == type(None):
        return None

    data["input_ids"].extend(new_chunk["input_ids"])
    data["attention_mask"].extend(new_chunk["attention_mask"])
    data["labels"].extend(new_chunk["labels"])
    data["targets"].extend(new_chunk["targets"])
    data["sentiment"].extend(new_chunk["sentiment"])

    return data


def extend_test(data, new_chunk):

    data["input_ids"].extend(new_chunk["input_ids"])
    data["attention_mask"].extend(new_chunk["attention_mask"])
    data["targets"].extend(new_chunk["targets"])

    return data


def custom_data_collator(batch_input):
    
    input_ids = [torch.tensor(inst["input_ids"], dtype=torch.long) for inst in batch_input]
    attention_mask = [torch.tensor(inst["attention_mask"], dtype=torch.long) for inst in batch_input]
    labels = [torch.tensor(inst["labels"], dtype=torch.long) for inst in batch_input]
    targets = [torch.tensor(inst["targets"], dtype=torch.long) for inst in batch_input]
    sentiment = [torch.tensor(inst["sentiment"], dtype=torch.long) for inst in batch_input]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)    

    new_batch_input = {
        "input_ids": input_ids.to(config.device),
        "attention_mask": attention_mask.to(config.device),
        "labels": labels.to(config.device),
        "targets": targets.to(config.device),
        "sentiment": torch.stack(sentiment).to(config.device)
    }

    return new_batch_input