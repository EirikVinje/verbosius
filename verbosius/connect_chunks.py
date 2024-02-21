import pickle
import os
import argparse

from numba import njit

import config as config


@njit
def connect_chunks(chunkdists : list, new_chunkdist_id : int):

    chunkdist_folder = os.path.join(config.root, config.dataset, "trainingdata")

    new_chunkdist_folder = os.path.join(chunkdist_folder, f"{config.dataset}_chunkdist_{new_chunkdist_id}")

    for n in chunkdists:

        chunkdist_n_folder = os.path.join(chunkdist_folder, f"{config.dataset}_chunkdist_{n}")
        chunkdist_n = sorted(os.listdir(chunkdist_folder))

        for chunk in chunkdist_n:

            chunk = os.path.join(chunkdist_n_folder, chunk)
            chunk = pickle.load(open(chunk, "rb"))

            chunk_id = len(os.listdir(new_chunkdist_folder))

            pickle.dump(chunk, open(os.path.join(new_chunkdist_folder, f"train_val_chunk_{chunk_id}.pkl"), "wb"))
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--chunkdists", type=str, required=True, help="Chunkdists to connect, on form '1 2 3 4'")
    parser.add_argument("--new_chunkdist_id", type=int, required=True, help="New chunkdist id")
    
    args = parser.parse_args()

    chunkdists = args.chunkdists
    new_chunkdist_id = args.new_chunkdist_id

    chunkdists = list(map(int, chunkdists.split()))

    connect_chunks(chunkdists, new_chunkdist_id)

