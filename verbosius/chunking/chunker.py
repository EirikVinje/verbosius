import os
import gzip
import pickle
import argparse

from tqdm import tqdm
import numpy as np

from chunking.get_data import get_dataset
import config
import arg_funcs as af


class Chunker:
    def __init__(self, 
                 size : str, 
                 chunkdist_n : int, 
                 chunk_amount : int,
                 seed : int = 42,
                 chunk_size : int = 8000):
        
        self.seed = seed
        self.chunkdist_n = chunkdist_n
        self.chunk_size = chunk_size
        self.chunk_amount = chunk_amount
        self.size = size


    def __call__(self):
        print("Setting up environment...")
        self._build_environment()
        print("Loading data...")
        self.load_amazon()
        print("Chunking data...")
        self.chunk_data()
        print("Done!")
        

    def load_amazon(self):

        dir = os.path.join(config.root, "pre_chunking", self.size)
        dir_orig_y = os.path.join(dir, "train_orig_labels.pkl")
        dir_data = os.path.join(dir, "train_data.pkl")  

        with open(dir_orig_y, "rb") as f:
            self.train_orig_y = pickle.load(f)

        with open(dir_data, "rb") as f:
            self.train_data = pickle.load(f)


    def chunk_data(self):

        if self.train_data is None:
            assert ValueError("self.data is None")
        
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

        with tqdm(total=self.chunk_amount, disable=False) as bar:

            bar.set_description("Processing chunk 1 of {}:".format(self.chunk_amount))

            for i in range(self.chunk_amount):

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

                bar.set_description("Processing chunk {} of {}".format(i+1, self.chunk_amount))
                bar.update(1)

                self._write_chunk(i_x)

    
    def _write_chunk(self):
    
        n = len(dir)
        file = os.path.join(self.chunking_dir, f"train_chunk_{n}_.pkl")
        with gzip.open(file, "wb") as f:
            pickle.dump(file, f)


    def _build_environment(self):

        if not os.path.exists(config.root):
            assert False, f"Directory {config.root} does not exist, please create it before continuing"

        self.chunking_dir = os.path.join(config.root, "chunking")
        preprocess_dir = os.path.join(config.root, "preprocess")
        weighted_dir = os.path.join(config.root, "weighted")
        trainingdata_dir = os.path.join(config.root, "trainingdata")
        models_dir = os.path.join(config.root, "models")

        if not os.path.exists(self.chunking_dir):
            os.mkdir(self.chunking_dir)

        if not os.path.exists(preprocess_dir):
            os.mkdir(preprocess_dir)
        
        if not os.path.exists(weighted_dir):
            os.mkdir(weighted_dir)
        
        if not os.path.exists(trainingdata_dir):
            os.mkdir(trainingdata_dir)
        
        train = os.path.join(trainingdata_dir, "train")
        eval = os.path.join(trainingdata_dir, "eval")

        if not os.path.exists(train):
            os.mkdir(train)

        if not os.path.exists(eval):
            os.mkdir(eval)

        if not os.path.exists(models_dir):
            os.mkdir(models_dir)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--chunk_amount", type=int)
    parser.add_argument("--chunkdist_n", type=int)
    parser.add_argument("--size", type=str)
    
    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunk_size_checker(args.chunk_size)
    af.chunk_amount_checker(args.chunk_amount)
    af.chunckdist_n_checker(args.chunkdist_n)

    Chunker(size=args.size, chunkdist_n=args.chunkdist_n, chunk_amount=args.chunk_amount)