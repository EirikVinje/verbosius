import argparse
import logging
import pickle
import os
import json
import time
import gzip
import gc

from transformers.modeling_outputs import TokenClassifierOutput
from transformers import TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel
from torch.optim import AdamW
from torch import nn
import numpy as np
import evaluate
import torch

from xai_transformer.xai_model import CustomModel
import utils.config as config
import utils.arg_funcs as af


logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class Transformer:
    def __init__(self, part_n : int, model_name : str):

        self.partition = f"part_{part_n}"
        self.trainingdata_dir = os.path.join(config.root, "trainingdata")
        self.model_dir = os.path.join(config.root, "models")
        self.model_name = model_name


    def _set_dir(self):

        pre_part_dir = os.path.join(self.trainingdata_dir, self.partition)

        if not os.path.exists(pre_part_dir):
            assert ValueError(f"Partition {self.partition} in {self.trainingdata_dir} does not exist")

        part_dir = os.path.join(self.model_dir, self.model_name)
        if not os.path.exists(part_dir):
            os.mkdir(part_dir)
        else:
            assert False, f"partion {part_dir} already exists in {self.model_dir}"


    def _set_train_eval(self):
        self.train_dataset = IterableDataset_Custom(os.path.join(self.trainingdata_dir, self.partition, "train"))
        self.eval_dataset = IterableDataset_Custom(os.path.join(self.trainingdata_dir, self.partition, "eval"))


    def _set_model(self):
        self.model = CustomModel(config.num_tok_labels, config.num_seq_labels, config.neutral_weight, config.loss_weight).to(config.device)
     

    def _data_collator(self, batch_input):

        input_ids = [torch.tensor(inst["input_ids"], dtype=torch.long) for inst in batch_input]
        attention_mask = [torch.tensor(inst["attention_mask"], dtype=torch.long) for inst in batch_input]
        targets = [torch.tensor(inst["targets"], dtype=torch.long) for inst in batch_input]
        labels = [torch.tensor(inst["labels"], dtype=torch.long) for inst in batch_input] if "labels" in batch_input[0].keys() else None
        sentiment = [torch.tensor(inst["sentiment"], dtype=torch.long) for inst in batch_input] if "sentiment" in batch_input[0].keys() else None

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        targets = pad_sequence(targets, batch_first=True, padding_value=0)
        
        if labels != None:
            labels = pad_sequence(labels, batch_first=True, padding_value=-100) 

        if sentiment == None and labels == None:
            
            new_batch_input = {
            "input_ids": input_ids.to(config.device),
            "attention_mask": attention_mask.to(config.device),
            "targets": targets.to(config.device)
            }

            return new_batch_input

        new_batch_input = {
            "input_ids": input_ids.to(config.device),
            "attention_mask": attention_mask.to(config.device),
            "labels": labels.to(config.device),
            "targets": targets.to(config.device),
            "sentiment": torch.stack(sentiment).to(config.device)
        }

        return new_batch_input


    def _compute_metrics(self, eval_preds):
    
        metric = evaluate.load("accuracy")
        logits, labels = eval_preds
        predictions = np.argmax(logits[1], axis=1)
        token_predictions = np.argmax(logits[0], axis=2)
        
        nz = (predictions!=0).sum()
        token_negative = (token_predictions==1).sum()
        token_neutral = (token_predictions==0).sum()
        token_positive = (token_predictions==2).sum()

        seqeval = evaluate.load("seqeval")
        
        label_list = ["neutral", "negative", "positive"]

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
        
        return output


    def _set_trainer(self):

        self.training_args = TrainingArguments(
            
            learning_rate = config.learning_rate,
            per_device_train_batch_size = config.trainer_batch_size,
            per_device_eval_batch_size = config.trainer_batch_size,
            
            output_dir = os.path.join(self.model_dir, self.model_name),
            num_train_epochs = config.num_train_epochs,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            load_best_model_at_end = True
            )

        if config.device != "cpu":
            self.training_args = self.training_args.set_dataloader(pin_memory=False)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=config.tokenizer,
            compute_metrics=self._compute_metrics,
            data_collator=self._data_collator,
            # optimizers=(AdamW(self.model.parameters(), lr=config.learning_rate), None)
            )


    def _train(self):
    
        start_t = time.time()

        self.trainer.train()

        end_t = time.time()

        torch.save(self.model, os.path.join(self.model_dir, self.model_name, self.model_name))

        time_dict = {"time_hours" : (end_t - start_t) / 3600}
        with open(os.path.join(self.model_dir, self.model_name, "time.json"), "w") as f:
            json.dump(time_dict, f, indent=4)

        self.train_dataset = None
        self.eval_dataset = None
        self.model = None
        self.trainer = None
        
        gc.collect()

    
    def run(self):

        self._set_dir()
        self._set_train_eval()
        self._set_model()
        self._set_trainer()
        self._train()


class IterableDataset_Custom(torch.utils.data.IterableDataset):
    def __init__(self, dir):        
        
        self.rng = np.random.default_rng(seed=config.seed)
        self.chunks = os.listdir(dir)
        self.rng.shuffle(self.chunks)
        self.dir = dir
    
    def __len__(self):

        length = 0
        for chunk in self.chunks:

            with gzip.open(os.path.join(self.dir, chunk), "rb") as f:
                data = pickle.load(f)
            length += len(data)

        return length
            
    def load_chunk(self, chunks):
        
        for chunk in chunks:
        
            with gzip.open(os.path.join(self.dir, chunk), "rb") as f:
                data = pickle.load(f)
            self.rng.shuffle(data)

            for sample in data:
                yield sample

    def __iter__(self):
        
        for sample in self.load_chunk(self.chunks):
            yield sample


class CustomModel(nn.Module): # transformers.modeling_utils.PreTrainedModel
    def __init__(self, num_tok_labels, num_seq_labels, neutral_weight, loss_weight, model_name='distilroberta-base'): 
        super(CustomModel,self).__init__() 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_weight = loss_weight
        
        self.seq_model = AutoModel.from_pretrained(model_name)
        self.token_model = AutoModel.from_pretrained(model_name)

        self.classifier = nn.Linear(768, num_tok_labels) 
        self.seq_classifier = nn.Linear(768, num_seq_labels)
        self.to_evidence = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
        
        self.cel = nn.CrossEntropyLoss(weight=torch.tensor([neutral_weight, 1.0, 1.0]).to(self.device))
        self.seq_cel = nn.CrossEntropyLoss()
        self.temp_evidence = None


    def forward(self, input_ids=None, attention_mask=None, labels=None, targets=None, sentiment=None):
        
        outputs_token = self.token_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        outputs_seq = self.seq_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)

        token_e = outputs_token.last_hidden_state
        
        token_labels_pred = self.classifier(token_e)
        
        evidence = torch.relu(self.to_evidence(token_labels_pred[:, :, 1:3]) - 0.1)
        
        seq_e = outputs_seq.last_hidden_state

        evidence_weighted = seq_e*(evidence.expand_as(token_e)).detach()
        
        w_targets = evidence_weighted*targets.unsqueeze(2).expand_as(token_e)
        
        seq_label_pred = self.seq_classifier(w_targets.sum(dim=1))
        
        logits = (token_labels_pred, seq_label_pred)
        
        if labels is not None:
            
            seq_loss = self.seq_cel(seq_label_pred, sentiment)
            
            token_loss = self.cel(token_labels_pred.view(-1, 3), labels.view(-1))
        
            loss = (seq_loss)+(token_loss*self.loss_weight)
            
            return TokenClassifierOutput(loss=loss, logits=logits)
        
        else:
            return TokenClassifierOutput(logits=logits)
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--part_n", type=int, help="")
    parser.add_argument("--model_name", type=str, help="")

    args = parser.parse_args()

    af.chunckdist_n_checker(args.part_n)
    
    transformer = Transformer(args.part_n, args.model_name)

    transformer.run()