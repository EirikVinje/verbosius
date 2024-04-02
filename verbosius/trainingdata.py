import argparse
import pickle 
import shutil
import gzip
import os
import gc

from tqdm import tqdm
import numpy as np

import utils.arg_funcs as af
import utils.config as config


class Trainingdata:
    def __init__(self, part_n : int, progress_bar : bool = False, force_write : bool = False):
        
        self.partition = f"part_{part_n}"
        self.weighter_dir = os.path.join(config.root, "weighter")
        self.trainingdata_dir = os.path.join(config.root, "trainingdata")

        self.progress_bar = progress_bar

        self.force_write = force_write
    
    
    def _set_dir(self):

        pre_part_dir = os.path.join(self.weighter_dir, self.partition)

        if not os.path.exists(pre_part_dir):
            assert ValueError(f"Partition {self.partition} in {self.weighter_dir} does not exist")

        part_dir = os.path.join(self.trainingdata_dir, self.partition)
        if not os.path.exists(part_dir):
            os.mkdir(part_dir)
            os.mkdir(os.path.join(part_dir, "train"))
            os.mkdir(os.path.join(part_dir, "eval"))

        elif self.force_write:
            shutil.rmtree(part_dir)
            os.mkdir(part_dir)
            os.mkdir(os.path.join(part_dir, "train"))
            os.mkdir(os.path.join(part_dir, "eval"))

        else:
            assert False, f"partion {part_dir} already exists in {self.trainingdata_dir}"


    def _get_class_balance(self):
        
        pre_part_dir = os.path.join(self.weighter_dir, self.partition)
        chunks = os.listdir(pre_part_dir)

        temp_y = []
        temp_orig_y = []
        for chunk in chunks:
            
            with gzip.open(os.path.join(pre_part_dir, chunk), "rb") as f:
                data = pickle.load(f)
            
            temp_y.extend([data[i]["y"] for i in range(len(data))])
            temp_orig_y.extend([data[i]["orig_y"] for i in range(len(data))])
        
        class_y_balance = np.unique(temp_y, return_counts=True)[1]
        # class_orig_y_balance = np.unique(temp_orig_y, return_counts=True)[1]

        self.class_y_balance = list(class_y_balance)
        self.balance_train = [list((class_y_balance * 0.8).astype(int)), [0, 0, 0]]
        self.balance_eval = [list((class_y_balance * 0.2).astype(int)), [0, 0, 0]]

        
    def _tokenize_and_align_labels(self, chunk):
    
        Y = np.array([(i['orig_y']) for i in chunk])
        
        tokenized_inputs = config.tokenizer([inst["token_x"] for inst in chunk], truncation=True, padding=False, is_split_into_words=True)

        labels = []
        targets = []
        
        for i, label in enumerate([inst["labels"] for inst in chunk]):
            
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

        # convert to list of dicts
        temp = []
        for i in range(len(output["input_ids"])):
            temp.append(
                {
                    "input_ids": output["input_ids"][i],
                    "attention_mask": output["attention_mask"][i],
                    "labels": output["labels"][i],
                    "targets": output["targets"][i],
                    "sentiment": output["sentiment"][i]
                }
            )    

        output = temp

        return output


    def _write_chunk(self, data, n, settype : str):

        part_dir = os.path.join(self.trainingdata_dir, self.partition, settype)

        with gzip.open(os.path.join(part_dir, f"{settype}_{n}.pkl"), "wb") as f:
            pickle.dump(data, f)


    def _main_loop(self):

        pre_part_dir = os.path.join(self.weighter_dir, self.partition)
        chunks = os.listdir(pre_part_dir)

        train = []
        eval = []

        n_e = 0
        n_t = 0

        chunk_size = 24000

        with tqdm(total=len(chunks), desc="(trainingdata) trainsplit", disable=self.progress_bar is False) as pbar:
            
            for chunk in chunks:
                
                with gzip.open(os.path.join(pre_part_dir, chunk), "rb") as f:
                    data = pickle.load(f)
                
                for sample in data:

                    if sample["y"] == 0 and self.balance_train[1][0] < self.balance_train[0][0]:
                        self.balance_train[1][0] += 1
                        train.append(sample)
                    elif sample["y"] == 1 and self.balance_train[1][1] < self.balance_train[0][1]:
                        self.balance_train[1][1] += 1
                        train.append(sample)
                    elif sample["y"] == 2 and self.balance_train[1][2] < self.balance_train[0][2]:
                        self.balance_train[1][2] += 1
                        train.append(sample)

                    if sample["y"] == 0 and self.balance_eval[1][0] < self.balance_eval[0][0]:
                        self.balance_eval[1][0] += 1
                        eval.append(sample)
                    elif sample["y"] == 1 and self.balance_eval[1][1] < self.balance_eval[0][1]:
                        self.balance_eval[1][1] += 1
                        eval.append(sample)
                    elif sample["y"] == 2 and self.balance_eval[1][2] < self.balance_eval[0][2]:
                        self.balance_eval[1][2] += 1
                        eval.append(sample)

                    if len(train) >= chunk_size:
                        
                        tokenized_train = self._tokenize_and_align_labels(train)
                        
                        self._write_chunk(tokenized_train, n_t, "train")
                        
                        train = []
                        n_t += 1

                    if len(eval) >= chunk_size:

                        tokenized_eval = self._tokenize_and_align_labels(eval)
                        
                        self._write_chunk(tokenized_eval, n_e, "eval")
                        
                        eval = []
                        n_e += 1

                pbar.update(1)


        if len(train) > 0:
            tokenized_train = self._tokenize_and_align_labels(train)
            self._write_chunk(tokenized_train, n_t, "train")


        if len(eval) > 0:
            tokenized_eval = self._tokenize_and_align_labels(eval)
            self._write_chunk(tokenized_eval, n_e, "eval")


    def run(self):
        self._set_dir()
        self._get_class_balance()
        self._main_loop()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--part_n", type=int, help="Set size for individual batch, must be greater than 0. Default value is 10000")

    args = parser.parse_args()

    af.chunckdist_n_checker(args.part_n)

    trainingdata = Trainingdata(args.part_n, progress_bar=True)
    trainingdata.run()