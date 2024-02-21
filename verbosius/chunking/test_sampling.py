import gzip
import json
import os
import argparse
from collections import Counter

import pickle
import numpy as np
from tqdm import tqdm


def raw_amazon_iterator(data_path):
    
    k = 0
    with gzip.open(data_path, mode="rt") as zp:
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


def sample_amazon(path, rng, data_size: int, test_size: int):
    
    data = []
    
    for index, d in enumerate(tqdm(raw_amazon_iterator(path))):
        
        try:

            if len(d["reviewText"].split(" ")) > 400:
                 continue    
                                
            data.append([index, str(d["reviewText"]), int(d["overall"])])

            if len(data) == 82_000_000:
                break
        
        except:
            continue

    data = np.array(data, dtype=object)

    print(Counter(data[:, 2]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    user = os.environ["USER"]
    datapath = "/home/bigtech/aggressive_dedup.json.gz"

    # store_path = f"/home/{user}/data/verbosius/amazon/pre_chunking/"    
    # store_dir = os.path.join(store_path, args.name)

    # if not os.path.exists(store_dir):
    #     os.makedirs(store_dir)
    # else:
    #     assert False, "Store dir already exists"


    sample_amazon(datapath, 
                  rng=rng,
                  data_size=1_000_000, 
                  test_size=100_000)
