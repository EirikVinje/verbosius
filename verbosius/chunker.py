import numpy as np
import gzip
import json
import pickle
import os
import argparse
import gzip
import shutil

from tqdm import tqdm

import utils.config as config

class Chunker:
    def __init__(self, 
                 n_chunks : int = 10,
                 n_reads : int = 10_000_000,
                 chunk_size : int = 8000,
                 inpath : str = "/home/bigtech/aggressive_dedup.json.gz",
                 outpath : str = "/home/bigtech/data/verbosius/amazon",
                 seed : int = 42,
                 force_write : bool = False):
        
        self.partition = f"nc_{n_chunks}"
        self.n_reads = n_reads
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        
        self.force_write = force_write

        self.inpath = inpath
        self.outpath = outpath
        
        self.rng = np.random.default_rng(seed)

        self.class_lookup = {
                                0: 1,
                                1: 1,
                                2: 0,
                                3: 2,
                                4: 2
                            }


    def _set_env(self):

        if not os.path.exists(self.inpath):
            raise ValueError(f"{self.inpath} does not exist")
        
        if not os.path.exists(self.outpath):
            os.mkdir(self.outpath)

        partdir = os.path.join(self.outpath, self.partition)
        if not os.path.exists(partdir):
            os.mkdir(partdir)

        elif self.force_write:
            shutil.rmtree(partdir)
            os.mkdir(partdir)

        else:
            raise ValueError(f"{partdir} already exists")
        

    def _raw_amazon_iterator(self):
        
        k = 0
        with gzip.open(self.inpath, mode="rt") as zp:
            for line in zp:
                try:
                    d = json.loads(line)
                except json.decoder.JSONDecodeError:
                    print("ok")
                    print(f"Skipped line {k}, len: {len(line)}")
                    k+=1
                    continue
                
                k += 1
                
                yield d
    

    def _write_chunk(self, data, n : int):

        part_dir = os.path.join(self.outpath, self.partition, "train")
        if not os.path.exists(part_dir):
            os.mkdir(part_dir)

        with gzip.open(os.path.join(part_dir, f"chunk_{n}_.pkl"), "wb") as f:
            pickle.dump(data, f)


    def _write_test(self):

        part_dir = os.path.join(self.outpath, self.partition, "test")
        if not os.path.exists(part_dir):
            os.mkdir(part_dir)

        self.rng.shuffle(self.eval)        

        test_size_count = np.array([[int((self.train_total_size*0.2) // 5), 0] for _ in range(5)])

        new_eval = []

        for sample in self.eval:
            
            orig_y = sample[2]

            if test_size_count[orig_y][1] < test_size_count[orig_y][0]:
                test_size_count[orig_y][1] += 1
                new_eval.append(sample)

            if np.sum(test_size_count[:, 1]) == np.sum(test_size_count[:, 0]):
                break
        
        new_eval = np.array(new_eval, dtype=object)

        with gzip.open(os.path.join(part_dir, "test.pkl"), "wb") as f:
            pickle.dump(new_eval[:, [0, 2]], f)


    def _read_amazon(self):
        
        data = []
        
        for i, d in enumerate(tqdm(self._raw_amazon_iterator(), desc="reading raw amazon")):
                
            try:
                if len(d["reviewText"].split(" ")) > 300:
                    continue    
                                    
                data.append([str(d["reviewText"]), int(d["overall"])])
                
                if len(data) == self.n_reads:
                    break
            
            except:
                continue
        
        self.data = np.array(data, dtype=object)
    

    def _split_data(self):

        class_y_balance = np.unique(self.data[:, 1], return_counts=True)[1]   

        class_train_balance = np.array([[(min(class_y_balance) * 0.8), 0] for _ in range(5)]).astype(int)
        class_eval_balance = np.array([[(min(class_y_balance) * 0.2), 0] for _ in range(5)]).astype(int)

        train = []
        eval = []

        for sample in tqdm(self.data, desc="splitting data into train/test"):
            
            orig_y = sample[1] - 1
            y = self.class_lookup[orig_y]

            s = [sample[0], y, orig_y]

            if class_train_balance[orig_y][1] < class_train_balance[orig_y][0]:
                class_train_balance[orig_y][1] += 1
                train.append(s)            

            elif class_eval_balance[orig_y][1] < class_eval_balance[orig_y][0]:
                class_eval_balance[orig_y][1] += 1
                eval.append(s)

            if np.sum(class_train_balance[:,1]) == np.sum(class_train_balance[:,0]) and np.sum(class_eval_balance[:,1]) == np.sum(class_eval_balance[:,0]):
                break
        
        self.train = np.array(train, dtype=object)
        self.eval = np.array(eval, dtype=object)


    def _chunk_train(self):
        
        x_y_0 = self.train[np.where(self.train[:, 2] == 0)[0]]
        x_y_1 = self.train[np.where(self.train[:, 2] == 1)[0]]
        x_y_2 = self.train[np.where(self.train[:, 2] == 2)[0]]
        x_y_3 = self.train[np.where(self.train[:, 2] == 3)[0]]
        x_y_4 = self.train[np.where(self.train[:, 2] == 4)[0]]
                
        self.rng.shuffle(x_y_0)
        self.rng.shuffle(x_y_1)
        self.rng.shuffle(x_y_2)
        self.rng.shuffle(x_y_3)
        self.rng.shuffle(x_y_4)

        labels_in_each = self.chunk_size // 5

        pre = 0
        curr = labels_in_each

        train_total_size = 0
    
        for i in tqdm(range(self.n_chunks), desc="chunking train data"):

            i_0 = x_y_0[pre:curr]
            i_1 = x_y_1[pre:curr]
            i_2 = x_y_2[pre:curr]
            i_3 = x_y_3[pre:curr]
            i_4 = x_y_4[pre:curr]

            if i_0.shape != (labels_in_each, 3):
                print("out of samples, last shape {} != {} .....".format(i_0.shape, (labels_in_each, 2)))
                print(f"{i} chunks in total...")
                break  

            i_x = np.vstack((i_0, i_1, i_2, i_3, i_4))

            self.rng.shuffle(i_x)

            pre = curr
            curr += labels_in_each

            train_total_size += i_x.shape[0]

            self._write_chunk(i_x, i)

        self.train_total_size = train_total_size


    def run(self):

        self._set_env()
        self._read_amazon()
        
        self._split_data()

        self._chunk_train()
        self._write_test()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocessing data")  
    parser.add_argument("--n_chunks", type=int, help="number of chunks")
    parser.add_argument("--n_reads", type=int, help="number of reads")
    
    args = parser.parse_args()
    
    amazon = Chunker(n_chunks=args.n_chunks, n_reads=args.n_reads)
    
    amazon.run()