import os
import gzip
import pickle

from tqdm import tqdm
import numpy as np

from chunking.get_data import get_dataset


class Chunker:
    def __init__(self, 
                 size : str, 
                 chunkdist_n : int, 
                 chunk_amount : int,
                 seed : int = 42,
                 chunk_size : int = 8000, 
                 dataset : str = "amazon"):
        
        self.seed = seed
        self.chunkdist_n = chunkdist_n
        self.chunk_size = chunk_size
        self.chunk_amount = chunk_amount

        self.data = get_dataset(dataset)(two_cat=True, size=size)


    def chunk_data(self):

        if self.data is None:
            assert ValueError("self.data is None")
        
        rng = np.random.default_rng(self.seed)

        x = self.data[:, 0].reshape(-1, 1)
        y = self.data[:, 1].reshape(-1, 1)
        orig_y = orig_y[:, 1].reshape(-1, 1)

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

                i_orig_y_0 = orig_y[idx_0]        
                i_orig_y_1 = orig_y[idx_1]
                i_orig_y_2 = orig_y[idx_2]
                
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