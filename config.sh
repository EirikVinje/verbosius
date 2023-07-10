#!/bin/bash

# *************** PREPROSESSING *************** 
preproc_input_path=/home/tobxtra/data/verbosius/imdb/
preproc_output_path=/home/tobxtra/data/verbosius/store_imdb_pickle/

dataset=imdb
batch_size=12500
batch_amount=2
use_test_set=True
shuffle=True
seed=42


# *************** GENERATE TRAINING DATA ***************
traindat_input_path=/home/tobxtra/data/verbosius/store_imdb_pickle/
traindat_output_path=/home/tobxtra/data/verbosius/store_imdb_trainingdata/

batchdist_n=0

# TM PARAMS
MAX_DF=0.7
MIN_DF=10
MAX_FEATURES=5000
N_GRAM_RANGE=(1, 2)
NUMBER_OF_CLAUSES=8000
LITERAL_BUDGET=10
S=42.69
T=1000
TM_EPOCHS=10

