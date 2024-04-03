import os
import gzip
import pickle
import argparse
import shutil

from tqdm import tqdm
import numpy as np

import utils.config as config
import utils.arg_funcs as af


class Chunker:
    def __init__(self, 
                 size : str, 
                 part_n : int, 
                 n_chunks : int,
                 seed : int = 42,
                 chunk_size : int = 8000,
                 force_write : bool = False,
                 progress_bar : bool = False):
        
        self.seed = seed
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.size = size
        
        self.train_orig_y = None
        self.train_data = None

        self.chunking_dir = os.path.join(config.root, "chunking")
        self.partition = f"part_{part_n}"

        self.progress_bar = progress_bar

        self.force_write = force_write
        

    def load_amazon(self):
        
        root = "/home/bigtech/data/verbosius/amazon"

        dir = os.path.join(root, "pre_chunking", self.size)
        dir_orig_y = os.path.join(dir, "train_orig_labels.pkl")
        dir_data = os.path.join(dir, "train_data.pkl")  

        with open(dir_orig_y, "rb") as f:
            self.train_orig_y = pickle.load(f)

        with open(dir_data, "rb") as f:
            self.train_data = pickle.load(f)


    def _chunk_data(self):
        
        if self.train_data is None:
            assert ValueError("self.train_data is none. Maybe forgot to load_amazon()")
        
        rng = np.random.default_rng(self.seed)

        x = self.train_data[:, 0].reshape(-1, 1)
        y = self.train_data[:, 1].reshape(-1, 1)
        self.train_orig_y = self.train_orig_y[:, 1].reshape(-1, 1)

        # _, labelcounts_y = np.unique(y, return_counts=True)
        # _, labelcounts_orig_y = np.unique(orig_y, return_counts=True)
        
        sample_index = np.arange(x.shape[0]).reshape(-1, 1)
        x = np.hstack((x, sample_index))
        
        is_0_idx = np.where(y[:,0] == 0)[0]
        is_1_idx = np.where(y[:,0] == 1)[0]
        is_2_idx = np.where(y[:,0] == 2)[0]

        x_0 = x[is_0_idx]
        x_1 = x[is_1_idx]
        x_2 = x[is_2_idx]

        rng.shuffle(x_0)
        rng.shuffle(x_1)
        rng.shuffle(x_2)

        labels_in_each = self.chunk_size // 3    

        pre = 0
        curr = labels_in_each

        with tqdm(total=self.n_chunks, disable=self.progress_bar is False) as bar:

            bar.set_description("(chunking) Processing chunk 1 of {}:".format(self.n_chunks))

            for i in range(self.n_chunks):

                _i_0 = x_0[pre:curr]
                _i_1 = x_1[pre:curr]
                _i_2 = x_2[pre:curr]

                if _i_0.shape != (labels_in_each, 2):
                    bar.close()
                    print("out of samples, last shape {} != {} .....".format(_i_0.shape, (labels_in_each, 2)))
                    print(f"{i} chunks in total...")
                    break

                i_x_0, idx_0 = _i_0[:, 0].reshape(-1, 1), _i_0[:, 1].astype(int)  
                i_x_1, idx_1 = _i_1[:, 0].reshape(-1, 1), _i_1[:, 1].astype(int)  
                i_x_2, idx_2 = _i_2[:, 0].reshape(-1, 1), _i_2[:, 1].astype(int)  

                i_y_0 = y[idx_0]        
                i_y_1 = y[idx_1]
                i_y_2 = y[idx_2]

                i_orig_y_0 = self.train_orig_y[idx_0]        
                i_orig_y_1 = self.train_orig_y[idx_1]
                i_orig_y_2 = self.train_orig_y[idx_2]
                
                i_0 = np.hstack((i_x_0, i_y_0, i_orig_y_0))
                i_1 = np.hstack((i_x_1, i_y_1, i_orig_y_1))
                i_2 = np.hstack((i_x_2, i_y_2, i_orig_y_2))

                i_x = np.vstack((i_0, i_1, i_2))

                pre = curr
                curr += labels_in_each

                self._write_chunk(i_x)
                
                bar.set_description("(chunking) Processing chunk {} of {}".format(i+1, self.n_chunks))
                bar.update(1)


    def _write_chunk(self, data):
    
        part_dir = os.path.join(self.chunking_dir, self.partition)
        
        n = len(os.listdir(part_dir))

        file = os.path.join(part_dir, f"chunk_{n}_.pkl")
        with gzip.open(file, "wb") as f:
            pickle.dump(data, f)


    def _build_environment(self):

        if not os.path.exists(config.root):
            assert False, f"Directory {config.root} does not exist, please create it before continuing"

        chunking_dir = os.path.join(config.root, "chunking")
        preprocess_dir = os.path.join(config.root, "preprocess")
        weighter_dir = os.path.join(config.root, "weighter")
        trainingdata_dir = os.path.join(config.root, "trainingdata")
        models_dir = os.path.join(config.root, "models")

        if not os.path.exists(chunking_dir):
            os.mkdir(chunking_dir)

        if not os.path.exists(preprocess_dir):
            os.mkdir(preprocess_dir)
        
        if not os.path.exists(weighter_dir):
            os.mkdir(weighter_dir)
        
        if not os.path.exists(trainingdata_dir):
            os.mkdir(trainingdata_dir)
        
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)

        part_dir = os.path.join(chunking_dir, self.partition)
        if not os.path.exists(part_dir):
            os.mkdir(part_dir)

        elif self.force_write:
            shutil.rmtree(part_dir)
            os.mkdir(part_dir)
        
        else:
            assert False, f"{part_dir} already exists, remove from {chunking_dir} before continuing"


    def run(self):
        self._build_environment()
        self.load_amazon()
        self._chunk_data()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--n_chunks", type=int)
    parser.add_argument("--part_n", type=int)
    parser.add_argument("--size", type=str)
    
    args = parser.parse_args()

    af.chunk_amount_checker(args.n_chunks)
    af.chunckdist_n_checker(args.part_n)

    chunker = Chunker(size=args.size, part_n=args.part_n, n_chunks=args.n_chunks, progress_bar=True)

    chunker.run()