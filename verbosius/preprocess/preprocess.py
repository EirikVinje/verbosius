import re
import pickle
import os
import gzip
import warnings

from bs4 import BeautifulSoup
import spacy

import config



class Preprocess:
    def __init__(self, partion_n):

        self.chunking_dir = os.path.join(config.root, "chunking")
        self.preprocess_dir = os.path.join(config.root, "preprocess")
        self.partition = f"part_{partion_n}"
        
            
    def __call__(self):
        pass


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
        docs = nlp.pipe(x, n_process=4) 

        tokens = []
        lemmas = []

        for doc in docs:

            tokens.append([token.text for token in doc])
            lemmas.append([token.lemma_.lower() for token in doc])

        return tokens, lemmas
    

    def _map_tokens(self, x, token_x):

        ids = []

        for i in range(len(x)):
            
            id = []
            topw = 0
            topt = 0
            
            while True:

                if topt >= len(token_x[i]) or topw >= len(x[i]):
                    break

                elif token_x[i][topt] == x[i][topw]:
                    id.append(topw)
                    topw += 1 
                    topt += 1

                elif token_x[i][topt] in x[i][topw]:
                    top_len = len(token_x[i][topt])  
                    id.append(topw)
                    topt += 1

                    while True:

                        if top_len == len(x[i][topw]):
                            topw += 1
                            break
                        
                        elif x[i][topw].find(token_x[i][topt], top_len) == top_len:
                            id.append(topw)
                            top_len += len(token_x[i][topt])
                            topt += 1
                
                else:
                    id.append(-1)
                    topt += 1

            ids.append(id)

        return ids


    def _write_part(self, token_x, lemma_x, token_id_x, y, orig_y, x):

        train_data = []
    
        for i in range(len(y)):
            
            instance = {
                        "token_x": token_x[i],
                        "lemma_x": lemma_x[i],
                        "y": y[i],
                        "orig_x" : x[i],
                        "token_id_x": token_id_x[i],
                        "orig_y": orig_y[i]}

            train_data.append(instance)

        n = len(os.listdir(self.preprocess_dir))
        chunk_dir = os.path.join(self.preprocess_dir, f"chunk_{n}_.pkl")

        with gzip.open(chunk_dir, "wb") as f:
            pickle.dump(train_data, f)
    

    def main_loop(self):

        chunks = os.listdir(os.path.join(self.chunking_dir, self.partition))
        sorted_chunks = sorted(chunks, key=lambda x: int(x.split("_")[2]))

        for chunkname in sorted_chunks:

            chunk = self._load_chunk(chunkname)

            