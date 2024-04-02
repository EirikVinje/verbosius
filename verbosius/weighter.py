from copy import deepcopy
from typing import Tuple
import argparse
import shutil
import pickle 
import gzip
import glob
import os
import gc


from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from trainingdata.helper_functions import make_weighted_data, write_train_chunk, write_error_chunk, set_directory, make_eval
import trainingdata.helper_functions as gen_data
import green_tsetlin as gt
import config as config
import utils.arg_funcs as af



class Weighter:
    def __init__(self, part_n : int, progress_bar : bool = False, force_write : bool = False) -> None:

        self.preprocess_dir = os.path.join(config.root, "preprocess")
        self.weighter_dir = os.path.join(config.root, "weighter")
        self.partition = f"part_{part_n}"

        self.progress_bar = progress_bar

        self.force_write = force_write


    def _set_dir(self) -> None:

        pre_part_dir = os.path.join(self.preprocess_dir, self.partition)

        if not os.path.exists(pre_part_dir):
            assert ValueError(f"Partition {self.partition} in {self.preprocess_dir} does not exist")

        part_dir = os.path.join(self.weighter_dir, self.partition)
        if not os.path.exists(part_dir):
            os.mkdir(part_dir)
        
        elif self.force_write:
            shutil.rmtree(part_dir)
            os.mkdir(part_dir)

        else:
            assert False, f"partion {part_dir} already exists in {self.weighter_dir}"

        remove_e_chunks = os.path.join(pre_part_dir, "*e.pkl")
        e_chunks = glob.glob(remove_e_chunks)

        for chunk in e_chunks:
            os.remove(chunk)

    
    def _bag_of_words(self, train_x, train_y, error_params : bool = False) -> np.ndarray:
    
        train_y = np.array(train_y, dtype=np.uint32)
        
        vectorizer = CountVectorizer(max_features=config.CV_MAX_FEATURES,
                                    max_df=config.MAX_DF if not error_params else config.ERROR_MAX_DF, 
                                    min_df=config.MIN_DF if not error_params else config.ERROR_MIN_DF,
                                    ngram_range=config.N_GRAM_RANGE,
                                    binary=True,
                                    dtype=np.uint8,
                                    stop_words=config.STOPWORDS)
        
        train_x_bin = vectorizer.fit_transform([" ".join(x) for x in train_x])

        feature_names = vectorizer.get_feature_names_out()

        return train_x_bin, feature_names

    
    def _select_k_best(self, bin_x, y, feature_names, error_params : bool = False) -> np.ndarray:

        SKB = SelectKBest(score_func=config.SKB_score_func, k=config.MAX_FEATURES if not error_params else config.ERROR_MAX_FEATURES)

        SKB.fit(bin_x, y)
        feature_names = SKB.get_feature_names_out(input_features=feature_names)
        
        bin_x = SKB.transform(bin_x).toarray()
        
        return bin_x, feature_names


    def _generate_ruleset(self, bin_x, y, error_params : bool = False) -> gt.RulePredictor:

        y = np.array(y, dtype=np.uint32)
        bin_x = bin_x.astype(np.uint8)

        tm = gt.TsetlinMachine(n_literals=bin_x.shape[1], 
                            n_clauses=config.NUMBER_OF_CLAUSES if not error_params else config.ERROR_NUMBER_OF_CLAUSES, 
                            n_classes=config.NUM_TM_LABELS,
                            s=config.S if not error_params else config.ERROR_S,
                            n_literal_budget=config.LITERAL_BUDGET if not error_params else config.ERROR_LITERAL_BUDGET)

        copy_bin_x = deepcopy(bin_x)
        copy_y = deepcopy(y)

        c_train_x, c_val_x, c_train_y, c_val_y = train_test_split(copy_bin_x, copy_y, test_size=0.2, random_state=config.seed)

        tm.set_train_data(c_train_x, c_train_y)
        tm.set_test_data(c_val_x, c_val_y)
        
        trainer = gt.Trainer(threshold=config.T if not error_params else config.ERROR_T, 
                            n_epochs=config.TM_EPOCHS, 
                            seed=config.seed, 
                            n_jobs=config.N_JOBS, 
                            early_exit_acc=config.EARLY_STOP_ACC,
                            progress_bar=False)

        trainer.train(tm)    

        ruleset = gt.RulePredictor()
        fm = list(range(bin_x.shape[1]))
        ruleset.create_from_state(tm.get_state(), fm)
        
        return ruleset


    def _do_weighting(self, train_data, feature_names, ruleset) -> Tuple[list[dict], list[dict]]:
    
        true_x_idx = []
        false_x_idx = []
        explanations = []

        for idx, inst in enumerate(train_data):
        
            bin_x = inst["bin_x"]
            y = inst["y"]
        
            prediction = ruleset.predict(bin_x, explain=False)
            
            expl = ruleset.explain(bin_x, [0, 1, 2])
            
            if prediction == 2:
                expl  = expl[2] - (expl[1] + expl[0])
            
            if prediction == 1:
                expl = expl[1] - (expl[2] + expl[0])
            
            if prediction == 0:
                expl = np.array([0 for _ in range(len(feature_names))])

            explanations.append(expl)
            
            votes = ruleset._inference.get_votes()
            n_votes = votes[prediction]
            
            if y == prediction and n_votes > 0: 
                true_x_idx.append([n_votes, idx])
            
            else:
                false_x_idx.append(idx)
        
        true_x_idx = np.array(true_x_idx)
        
        percentile_25 = np.percentile(true_x_idx[:, 0], 25)

        is_75_percentile = np.where(true_x_idx[:, 0] >= percentile_25)[0]
        is_25_percentile = np.where(true_x_idx[:, 0] < percentile_25)[0]

        true_x = true_x_idx[is_75_percentile]
        true_x = list(true_x[:, 1:3])

        is_25_percentile = true_x_idx[is_25_percentile]
        is_25_percentile = list(is_25_percentile[:, 1])

        false_x_idx.extend(is_25_percentile)
        false_x = false_x_idx

        true_data = []
        false_data = []

        for idx, inst in enumerate(train_data):
            
            y = inst["y"]
            token_x = inst["token_x"]
            orig_y = inst["orig_y"]

            lemma_x = inst["lemma_x"]
            token_ids_x = inst["token_ids_x"]
            
            x = inst["x"]
            sample_index = inst["sample_index"]


            if idx in true_x:

                expl = explanations[idx]
                
                vocabulary = {feature_names[i]: expl[i] for i in range(len(feature_names))}
            
                newtokens_x, weights_x = self._weight_tokens(lemma_x, token_x, vocabulary, token_ids_x)

                labels = self._label_tokens(y, weights_x)

                true_inst = {"sample_index" : sample_index,
                            "token_x" : newtokens_x,
                            "weights" : weights_x,
                            "y" : y,
                            "labels" : labels,
                            "orig_y" : orig_y}

                true_data.append(true_inst)
                

            elif idx in false_x:
                
                false_inst = {"sample_index" : sample_index,
                            "y" : y,
                            "lemma_x" : lemma_x,
                            "token_x" : token_x,
                            "token_ids_x" : token_ids_x,
                            "orig_y" : orig_y,
                            "x" : x}

                false_data.append(false_inst)
        

        return true_data, false_data
    

    def _weight_tokens(self, lemma_x, token_x, vocabulary, token_ids_x) -> Tuple[list, list]:
    
        weights = np.zeros(len(lemma_x))
        for i, lemma in enumerate(lemma_x):

            unigram = lemma if i < len(lemma_x) else None
            bigram = "{} {}".format(lemma, lemma_x[i+1]) if i+1 < len(lemma_x) else None
            trigram = "{} {} {}".format(lemma, lemma_x[i+1], lemma_x[i+2]) if i+2 < len(lemma_x) else None
            
            if trigram in vocabulary.keys() and trigram is not None:
                
                tri_w = vocabulary[trigram] * 1/3
                weights[i] += tri_w
                weights[i+1] += tri_w
                weights[i+2] += tri_w

            if bigram in vocabulary.keys() and bigram is not None:
                
                bi_w = vocabulary[bigram] * 1/2 
                weights[i] += bi_w
                weights[i+1] += bi_w

            if unigram in vocabulary.keys():
                
                uni_w = vocabulary[unigram]
                weights[i] += uni_w

        new_toks, new_weights = self._connect_tokens(token_x, weights, token_ids_x)
        
        return new_toks, new_weights


    def _connect_tokens(self, tokens, weights, token_map) -> Tuple[list, list]:

        new_toks = []
        new_weights = []
        pre_id = None

        for i, token in enumerate(tokens):
            
            if pre_id is not None:

                curr_id = token_map[i]

                if curr_id == pre_id:
                    new_toks[-1] += token
                    new_weights[-1] += weights[i]
                    pre_id = token_map[i]
                
                else:
                    new_toks.append(token)
                    new_weights.append(weights[i])
                    pre_id = token_map[i]

            else:
                new_toks.append(token)
                new_weights.append(weights[i])
                pre_id = token_map[i]

        return new_toks, new_weights


    def _label_tokens(self, y, weights) -> list:

        if y == 2:
            labels = [2 if x > 0 else 1 if x < 0 else 0 for x in weights]
        
        elif y == 1:
            labels = [1 if x > 0 else 2 if x < 0 else 0 for x in weights]

        elif y == 0:
            labels = [0 for x in weights]
        
        return labels


    def _write_chunk(self, data, n) -> None:

        filepath = os.path.join(self.weighter_dir, self.partition, f"chunk_{n}_.pkl")
        with gzip.open(filepath, "wb") as file:
            pickle.dump(data, file)


    def _write_e_chunk(self, data, n) -> None:
        
        filepath = os.path.join(self.preprocess_dir, self.partition, f"chunk_{n+100000}_e.pkl")
        with gzip.open(filepath, "wb") as file:
            pickle.dump(data, file)


    def _main_loop(self) -> None:

        part_dir = os.path.join(self.preprocess_dir, self.partition)
        dir_len = len(os.listdir(part_dir)) * 2
        
        error_params = False

        with tqdm(total=dir_len, disable=self.progress_bar is False) as bar:
            bar.set_description("(weighter) Processing chunk 1 of {}".format(dir_len))
        
            for n in range(dir_len):
                
                dir = sorted(os.listdir(part_dir), key=lambda x: int(x.split("_")[1]))

                if n > len(dir):
                    assert False, "Something went wrong here."

                chunkname = os.path.join(part_dir, dir[n])

                with gzip.open(chunkname, "rb") as f:
                    chunk = pickle.load(f)

                error_params = False if chunkname[-5] != "e" else True

                lemma_x = [instance["lemma_x"] for instance in chunk]
                y = [instance["y"] for instance in chunk]
                
                bin_x, feature_names = self._bag_of_words(lemma_x, y, error_params=error_params)
                bin_x, feature_names = self._select_k_best(bin_x, y, feature_names, error_params=error_params)
                ruleset = self._generate_ruleset(bin_x, y, error_params=error_params)

                for i in range(len(chunk)):
                    chunk[i]["bin_x"] = bin_x[i]

                true_data, false_data = self._do_weighting(chunk, feature_names, ruleset)

                self._write_chunk(true_data, n)
                
                if not error_params:
                    self._write_e_chunk(false_data, n)

                bar.set_description("(weighter) Processing chunk {} of {}".format(n+1, dir_len))

                true_data = None
                false_data = None
                ruleset = None
                bin_x = None
                chunk = None

                gc.collect()

                bar.update(1)


    def run(self):

        self._set_dir()
        self._main_loop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stage trainingdata to transformer")

    parser.add_argument("--part_n", type=int, help="Set size for individual batch, must be greater than 0. Default value is 10000")

    args = parser.parse_args()

    af.chunckdist_n_checker(args.part_n)
    
    weighter = Weighter(args.part_n, progress_bar=True)

    weighter.run()