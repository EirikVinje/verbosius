import re
import pickle
import os
import gzip
import warnings
import argparse
import shutil

from bs4 import BeautifulSoup
from tqdm import tqdm
import spacy

import utils.config as config
import utils.arg_funcs as af


class Preprocess:
    def __init__(self, part_n : int, progress_bar : bool = False, force_write : bool = False):

        self.chunking_dir = os.path.join(config.root, "chunking")
        self.preprocess_dir = os.path.join(config.root, "preprocess")
        self.partition = f"part_{part_n}"

        self.progress_bar = progress_bar
        
        self.force_write = force_write


    def _set_dir(self):

        if not os.path.exists(os.path.join(self.chunking_dir, self.partition)):
            assert ValueError(f"Partition {self.partition} in {self.chunking_dir} does not exist")

        part_dir = os.path.join(self.preprocess_dir, self.partition)
        if not os.path.exists(part_dir):
            os.mkdir(part_dir)

        elif self.force_write:
            shutil.rmtree(part_dir)
            os.mkdir(part_dir)

        else:
            assert False, f"partion {part_dir} already exists in {self.preprocess_dir}"


    def _load_chunk(self, chunkname):

        chunk_path = os.path.join(self.chunking_dir, self.partition, chunkname)
        if not os.path.exists(chunk_path):
            assert False, f"Chunk does not exist : {chunk_path}"

        with gzip.open(chunk_path, "rb") as f:
            data = pickle.load(f)

        return data
    

    def _clean_text(self, x):

        _only_letters_pattern = re.compile(r"[^A-Za-z0-9']+")
        _no_long_numbers_pattern = re.compile(r'\d{3,}')
        _no_multiple_quotes_pattern = re.compile(r"'+")
        _no_multiple_spaces_pattern = re.compile(r"  +")

        for i in range(len(x)):
        
            x[i] = self._strip_html(x[i])
            x[i] = x[i].lower()
            x[i] = _only_letters_pattern.sub(' ',x[i])
            x[i] = _no_long_numbers_pattern.sub('', x[i])
            x[i] = _no_multiple_quotes_pattern.sub("", x[i])
            x[i] = _no_multiple_spaces_pattern.sub(" ", x[i])
            x[i] = x[i].strip()

        return x


    def _strip_html(self, x):
    
        with warnings.catch_warnings():
        
            warnings.simplefilter("ignore")

            soup = BeautifulSoup(x, "html.parser")
            return soup.get_text()
        
    
    def _lemmatize(self, x):

        nlp = spacy.load("en_core_web_sm")
        docs = nlp.pipe(x, n_process=-1) 

        tokens = []
        lemmas = []

        for doc in docs:

            tokens.append([token.text for token in doc])
            lemmas.append([token.lemma_.lower() for token in doc])

        return tokens, lemmas
    

    def _map_tokens(self, x, token_x):
        
        split_x = [t.split() for t in x]

        ids = []

        for i in range(len(split_x)):
            
            id = []
            topw = 0
            topt = 0
            
            while True:

                if topt >= len(token_x[i]) or topw >= len(split_x[i]):
                    break

                elif token_x[i][topt] == split_x[i][topw]:
                    id.append(topw)
                    topw += 1 
                    topt += 1

                elif token_x[i][topt] in split_x[i][topw]:
                    top_len = len(token_x[i][topt])  
                    id.append(topw)
                    topt += 1

                    while True:

                        if top_len == len(split_x[i][topw]):
                            topw += 1
                            break
                        
                        elif split_x[i][topw].find(token_x[i][topt], top_len) == top_len:
                            id.append(topw)
                            top_len += len(token_x[i][topt])
                            topt += 1
                
                else:
                    id.append(-1)
                    topt += 1

            ids.append(id)

        return ids


    def _write_chunk(self, token_x, lemma_x, token_id_x, y, orig_y, x, sample_index):

        train_data = []

        for i in range(len(y)):
            
            instance = {
                        "sample_index" : sample_index,
                        "token_x": token_x[i],
                        "lemma_x": lemma_x[i],
                        "y": y[i],
                        "token_ids_x": token_id_x[i],
                        "orig_y": orig_y[i],
                        "x" : x[i]}

            train_data.append(instance)

            sample_index += 1

        part_dir = os.path.join(self.preprocess_dir, self.partition)
        n = len(os.listdir(part_dir))
        chunk_dir = os.path.join(part_dir, f"chunk_{n}_.pkl")

        with gzip.open(chunk_dir, "wb") as f:
            pickle.dump(train_data, f)

        return sample_index


    def _main_loop(self):

        chunks = os.listdir(os.path.join(self.chunking_dir, self.partition))
        sorted_chunks = sorted(chunks, key=lambda x: int(x.split("_")[1]))

        with tqdm(total=len(chunks), disable=self.progress_bar is False) as bar:
        
            bar.set_description("(preprocess) Processing chunk 1 of {}".format(len(chunks)))

            sample_index = 0
            for i, chunkname in enumerate(sorted_chunks):

                chunk = self._load_chunk(chunkname)

                x = chunk[:, 0]
                y = chunk[:, 1]
                orig_y = chunk[:, 2]

                x = self._clean_text(x)
                token_x, lemma_x = self._lemmatize(x)
                token_ids = self._map_tokens(x, token_x)

                sample_index = self._write_chunk(token_x, lemma_x, token_ids, y, orig_y, x, sample_index)    

                bar.set_description("(preprocess) Processing chunk {} of {}".format(i+1, len(chunks)))
                bar.update(1)
    

    def run(self):
        self._set_dir()
        self._main_loop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage data for training")

    parser.add_argument("--part_n", type=int, help="Which chunk to stage")

    args = parser.parse_args()

    af.chunckdist_n_checker(args.part_n)
    
    preprocess = Preprocess(args.part_n, progress_bar=True)

    preprocess.run()