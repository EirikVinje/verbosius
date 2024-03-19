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

    
    def _write_chunk(self, data):
    
        output = os.path.join(output, "train")

        if not os.path.exists(output):
            os.mkdir(output)
        
        dir = os.listdir(output)

        n = len(dir)

        file = os.path.join(output, f"train_chunk_{n}_.pkl")

        with gzip.open(file, "wb") as f:
            pickle.dump(data, f)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--dataset", type=str, help="Dataset to stage")
    parser.add_argument("--chunk_size", type=int, nargs='?', default=10000, help="Set size for individual chunk, must be greater than 0. Default value is 10000. If used together with a train/test - split this chunk size will be split up into a train and test part accordingly. ")
    parser.add_argument("--chunk_amount", type=int, nargs ='?', default=1, help="Set amount of chunks to stage at a time. Minimum value is 1. Default value is 1.")
    parser.add_argument("--chunkdist_n", type=int, help="Set size for individual batch, must be >= 0. ")
    parser.add_argument("--size", type=str, help="Size of dataset to use")
    
    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunk_size_checker(args.chunk_size)
    af.chunk_amount_checker(args.chunk_amount)
    af.chunckdist_n_checker(args.chunkdist_n)


    chunker = Chunker(size=args.size, chunkdist_n=args.chunkdist_n, chunk_amount=args.chunk_amount)
    chunker.chunk_data()