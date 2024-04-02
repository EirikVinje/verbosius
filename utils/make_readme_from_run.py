import os
import argparse
import json
from datetime import datetime

import utils.config as config
import performance as md
import utils.arg_funcs as af


def make_readme_from_run(dataset, chunkdist_n):

    root = config.root
    user = config.user

    models_folder = os.path.join(root, dataset, "models")
    if not os.path.exists(models_folder):
        assert False, f"Models folder {models_folder} does not exist, please check your input"

    model_folder = os.path.join(models_folder, f"{dataset}_model_dist_{chunkdist_n}")
    if not os.path.exists(model_folder):
        assert False, f"Model folder {model_folder} does not exist, please check your input"

    model_path = os.path.join(model_folder, "model")
    if not os.path.exists(model_path):
        assert False, f"Model {model_path} does not exist, please check your input"
    
    model_acc = md.model_accuracy(dataset, model_path)

    timedelta_path = os.path.join(model_folder, "time.json")
    if not os.path.exists(timedelta_path):
        assert False, f"Timedelta {timedelta_path} does not exist, please check your input"
    
    timedelta = json.load(open(timedelta_path, "r"))
    timedelta = timedelta["time_hours"]

    path_to_markdown = f"/home/{user}/project/labs.journal/verbosius/runs/"
    
    month = datetime.now().month
    day = datetime.now().day
    hour = datetime.now().hour
    minute = datetime.now().minute
    mardown_file = f"run_{dataset}_{chunkdist_n}_{minute}_{hour}_{day}_{month}.md"
    path_to_markdown = os.path.join(path_to_markdown, mardown_file)

    metachunks_path = os.path.join(root, dataset, "chunking", f"{dataset}_chunkdist_{chunkdist_n}", "meta.json")
    if not os.path.exists(metachunks_path):
        assert False, f"Metachunks {metachunks_path} does not exist, please check your input"

    meta = json.load(open(metachunks_path, "r"))
    chunksize = meta["train_length"]
    n_chunks = meta["chunk_amount"]

    with open(path_to_markdown, "w") as f:

        f.write(f"# Run on chunkdist : {chunkdist_n} \n")
        f.write(f"- Dataset: {dataset} \n")
        f.write(f"- Model accuracy: ${model_acc}$ \n")
        f.write(f"- Chunksize: {chunksize} \n")
        f.write(f"- Number of chunks: {n_chunks} \n")

        f.write(f"## Parameters: \n")
        f.write(f"### trainingdata \n")
        f.write(f"MAX_FEATURES : {config.MAX_FEATURES} \n")
        f.write(f"ERROR_MAX_FEATURES : {config.ERROR_MAX_FEATURES} \n")
        f.write(f"MAX_DF : {config.MAX_DF} \n")
        f.write(f"ERROR_MAX_DF : {config.ERROR_MAX_DF} \n")
        f.write(f"MIN_DF : {config.MIN_DF} \n")
        f.write(f"ERROR_MIN_DF : {config.ERROR_MIN_DF} \n")
        f.write(f"NUMBER_OF_CLAUSES : {config.NUMBER_OF_CLAUSES} \n")
        f.write(f"ERROR_NUMBER_OF_CLAUSES : {config.ERROR_NUMBER_OF_CLAUSES} \n")
        f.write(f"S : {config.S} \n")
        f.write(f"ERROR_S : {config.ERROR_S} \n")
        f.write(f"T : {config.T} \n")
        f.write(f"ERROR_T : {config.ERROR_T} \n")
        f.write(f"TM_EPOCHS : {config.TM_EPOCHS} \n")
        f.write(f"SKB_score_func : {config.SKB_score_func} \n")
        f.write(f"STOPWORDS : {config.STOPWORDS} \n")
        f.write(f"N_JOBS : {config.N_JOBS} \n")
        f.write(f"EARLY_STOP_ACC : {config.EARLY_STOP_ACC} \n")
        f.write(f"error_chunk : {config.error_chunk} \n")
        f.write(f"n_badtexts : {config.n_badtexts} \n")
        f.write(f"CV_MAX_FEATURES : {config.CV_MAX_FEATURES} \n")
        f.write(f"N_GRAM_RANGE : {config.N_GRAM_RANGE} \n")
        f.write(f"LITERAL_BUDGET : {config.LITERAL_BUDGET} \n")
        f.write(f"ERROR_LITERAL_BUDGET : {config.ERROR_LITERAL_BUDGET} \n")
        f.write(f"\n")
        f.write(f"### transformer \n")
        f.write(f"learning_rate : {config.learning_rate} \n")
        f.write(f"per_device_train_batch_size : {config.per_device_train_batch_size} \n")
        f.write(f"per_device_eval_batch_size : {config.per_device_eval_batch_size} \n")
        f.write(f"num_train_epochs : {config.num_train_epochs} \n")
        f.write(f"evaluation_strategy : {config.evaluation_strategy} \n")
        f.write(f"save_strategy : {config.save_strategy} \n")
        f.write(f"load_best_model_at_end : {config.load_best_model_at_end} \n")
        f.write(f"label_names : {config.label_names} \n")
        f.write(f"neutral_weight : {config.neutral_weight} \n")
        f.write(f"loss_weight : {config.loss_weight} \n")
        f.write(f"num_labels : {config.num_labels} \n")
        f.write(f"num_seq_labels : {config.num_seq_labels} \n")
        f.write(f"\n")
        f.write(f"### model \n")
        f.write(f"model_name_ : {config.model_name_} \n")
        f.write(f"device : {config.device} \n")

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--chunkdist_n", type=int, required=True)

    args = parser.parse_args()

    af.dataset_checker(args.dataset)
    af.chunckdist_n_checker(args.chunkdist_n)

    make_readme_from_run(args.dataset, args.chunkdist_n)