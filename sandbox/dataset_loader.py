import torch
import pickle
import os
from itertools import cycle
import numpy as np


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, chunks, dir):        
        
        self.rng = np.random.default_rng(seed=42)

        self.chunks = chunks
        self.rng.shuffle(self.chunks)
        
        self.dir = dir

    def load_chunk(self, chunks):
        
        for chunk in chunks:
            
            with open(os.path.join(self.dir, chunk), "rb") as f:
                data = pickle.load(f)

            self.rng.shuffle(data)

            for sample in data:
                yield sample

    def __iter__(self):
        
        for sample in self.load_chunk(self.chunks):
            yield sample


def custom_collate_fn(batch):
    
    return batch




if __name__ == "__main__":
    

    dir = "/home/bigtech/data/verbosius/testing"
    chunks = ["chunk1.pkl", "chunk2.pkl", "chunk3.pkl", "chunk4.pkl", "chunk5.pkl"]
    
    
    dataset = Dataset(chunks, dir)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=12, collate_fn=custom_collate_fn)

    for i, batch in enumerate(dataloader):
        
        print(batch)
        

# chunk1.pkl : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# chunk2.pkl : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# chunk3.pkl : [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# chunk4.pkl : [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# chunk5.pkl : [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]