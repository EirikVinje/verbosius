import argparse
import logging
import pathlib
import json
import os
import gc
import time

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from transformers import Trainer, TrainingArguments, AutoModel
import matplotlib.pyplot as plt
import numpy as np
import torch

import xai_validation.helper_functions_xaival as hf_xaival
from xai_transformer.xai_model import CustomModel
import xai_transformer.helper_functions as hf
import chunking.get_data as gd
import config as config
import arg_funcs as af


logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class ModelMetrics:

    def __init__(self, model_name : str, chunkdist_n : int, checkpoint : bool = False, dataset : str = "amazon", size : str = "big", seed : int = 42):
        
        self.model_name = model_name
        self.chunkdist_n = chunkdist_n
        self.dataset = dataset
        self.size = size
        self.seed = seed
        self.checkpoint = checkpoint
        self.model_path = os.path.join(config.root, "models", f"{self.dataset}_chunkdist_{self.chunkdist_n}")


    def load_test(self):
        
        rng = np.random.default_rng(seed=config.seed)

        test = gd.dataset(self.dataset)(two_cat=True, size=self.size).load_test()

        rng.shuffle(test)

        new_test_x = hf_xaival.tokenize_to_model([text for text, _ in test], config.tokenizer, config.device)

        test_x = {"input_ids": [], "attention_mask": [], "targets": []}
        
        test_x = hf.extend_test(test_x, new_test_x)
        test_y = [label for _, label in test]

        test_x = hf.Test_Dataset(**test_x)

        self.test_x = test_x
        self.test_y = test_y


    def set_model(self):

        if self.checkpoint:
            self.model = CustomModel(config.num_tok_labels, config.num_seq_labels, config.neutral_weight, config.loss_weight, model_name=self.model_name)

        else:
            self.model = torch.load(os.path.join(self.model_path, self.model_name))

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

        return {"seq_acc": self.seq_acc, "seq_prec": self.seq_prec, "seq_f1": self.seq_f1}
    

    def save_metrics(self):
        
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
        
        os.system(f"git add {self.metric_folder} && git commit -m 'save run' && git push origin HEAD")


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model performance")

    parser.add_argument("--chunkdist_n", type=int, help="chunkdist number")
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--checkpoint", type=int, help="checkpoint")
    
    args = parser.parse_args()

    model_name = args.model_name
    chunkdist_n = args.chunkdist_n
    checkpoint = bool(args.checkpoint)

    model_metrics = ModelMetrics(model_name, chunkdist_n, checkpoint)
    model_metrics.load_test()
    model_metrics.set_model()
    model_metrics.get_metrics()
    model_metrics.save_metrics()