from transformers import AutoTokenizer


# *************** PREPROSESSING *************** 
preproc_input_path="/home/bigtech/data/verbosius/imdb/"
preproc_output_path="/home/bigtech/data/verbosius/store_imdb_pickle/"

dataset="rottentomatoes"
batch_size=8530
batch_amount=1
use_test_set=True
batch_size_test = 1066
shuffle=True
seed=42


# *************** GENERATE TRAINING DATA ***************
traindat_input_path="/home/bigtech/data/verbosius/store_imdb_pickle/"
traindat_output_path="/home/bigtech/data/verbosius/store_imdb_trainingdata/"

batchdist_n=3

# TM PARAMS
MAX_DF=0.5263
MIN_DF=21
MAX_FEATURES=5000
N_GRAM_RANGE=(1, 3)
NUMBER_OF_CLAUSES=500 #5853
LITERAL_BUDGET=8
S=3.4606
T=9225
TM_EPOCHS=1
EARLY_STOP_ACC=0.86
STOPWORDS=None #"english"

#TOKENSTUFF
model_name_ = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_, add_prefix_space=True)
device="cpu"
