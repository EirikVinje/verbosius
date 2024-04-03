import argparse
import logging
import pathlib
import json
import os
import gc
import time
import pickle

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from transformers import TrainingArguments, Trainer
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer import CustomModel
import utils.config as config



logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class ModelMetrics:
    def __init__(self, model_name : str, 
                 size : str, 
                 seed : int = 42, 
                 checkpoint : bool = False):
        
        self.model_name = model_name
        self.size = size
        self.seed = seed
        self.checkpoint = checkpoint
        self.model_dir = os.path.join(config.root, "models", self.model_name)


    def load_test(self):
        
        rng = np.random.default_rng(seed=config.seed)

        test_path = os.path.join(config.root, "pre_chunking", self.size, "test_data.pkl")

        with open(test_path, "rb") as f:
            test = pickle.load(f)

        rng.shuffle(test)

        new_test_x = self._tokenize_to_model([text for text, _ in test], config.tokenizer, config.device)

        test_x = {"input_ids": [], "attention_mask": [], "targets": []}
        
        test_x = self._extend_test(test_x, new_test_x)
        test_y = [label for _, label in test]

        test_x = Test_Dataset(**test_x)

        self.test_x = test_x
        self.test_y = test_y


    def _extend_test(self, data, new_chunk):

        data["input_ids"].extend(new_chunk["input_ids"])
        data["attention_mask"].extend(new_chunk["attention_mask"])
        data["targets"].extend(new_chunk["targets"])

        return data


    def _tokenize_to_model(self, data, tokenizer, device):
    
        tokenized_inputs = tokenizer(data, 
                                    truncation=True, 
                                    padding="longest", 
                                    return_tensors='pt',
                                    max_length=512,
                                    )
        
        targets = []
        
        for i in range(len(data)):
        
            word_ids = tokenized_inputs.word_ids(batch_index=i)      
            previous_word_idx = None
            target_ids = []
            
            for word_idx in word_ids:  
                
                if word_idx is None:
                    target_ids.append(0)

                elif word_idx != previous_word_idx: 
                    target_ids.append(1)

                else:
                    target_ids.append(0)

                previous_word_idx = word_idx
            targets.append(target_ids)
            
        tokenized_inputs["targets"] = torch.tensor(targets).to(device = device)
        tokenized_inputs["input_ids"]= tokenized_inputs["input_ids"].to(device = device) 
        tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].to(device = device) 
    
        return tokenized_inputs


    def set_model(self):

        if self.checkpoint:
            raise NotImplementedError

        else:
            self.model = torch.load(os.path.join(self.model_dir, self.model_name))

        training_args = TrainingArguments(
            output_dir = "/home/bigtech/",
            per_device_train_batch_size = 64,
            per_device_eval_batch_size = 64,
            label_names = config.label_names,        
            )

        if config.device != "cpu":
            training_args = training_args.set_dataloader(pin_memory=False)

        self.trainer = Trainer(model=self.model, args=training_args)    


    def get_metrics(self):

        preds = self.trainer.predict(self.test_x)
    
        seq_logits = preds[0][1]
        self.y_pred = np.argmax(seq_logits, axis=1)

        self.seq_acc = accuracy_score(self.test_y, self.y_pred)
        self.seq_prec = precision_score(self.test_y, self.y_pred, average="weighted")
        self.seq_f1 = f1_score(self.test_y, self.y_pred, average="weighted")

        self.metrics = {"seq_acc": self.seq_acc, "seq_prec": self.seq_prec, "seq_f1": self.seq_f1}  
    

    def save_metrics(self, gitsave : bool = False):
        
        path = "/home/bigtech/projects/verbosius/model_metrics"
        folder = f"test_model_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        self.metric_folder = os.path.join(path, folder)
        os.mkdir(self.metric_folder)

        conf_mat = confusion_matrix(self.test_y, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["1*", "2*", "3*", "4*", "5*"])
        disp.plot()
        plt.savefig(os.path.join(self.metric_folder, f"confusion_matrix_{self.model_name}.png"))
        
        metric_dict = {"seq_acc": self.seq_acc, "seq_prec": self.seq_prec, "seq_f1": self.seq_f1}



        with open(os.path.join(self.metric_folder, f"metrics_{self.model_name}.json"), "w") as f:
            json.dump(metric_dict, f)
        

        if gitsave:
            os.system(f"git add {self.metric_folder} && git commit -m 'save run' && git push origin HEAD")


class Test_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, targets):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        targets = self.targets[idx]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': targets
        }


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model performance")

    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--size", type=str, help="size")
    parser.add_argument("--checkpoint", type=int, default=0, help="checkpoint")

    args = parser.parse_args()

    model_name = args.model_name
    checkpoint = args.checkpoint
    size = args.size 

    model_metrics = ModelMetrics(model_name=model_name, checkpoint=checkpoint, size=size)
    model_metrics.load_test()
    model_metrics.set_model()
    model_metrics.get_metrics()

    print(model_metrics.metrics)

    model_metrics.save_metrics()