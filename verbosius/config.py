# *************** PREPROSESSING *************** 
preproc_input_path="/home/tobxtra/data/verbosius/imdb/"
preproc_output_path="/home/tobxtra/data/verbosius/store_imdb_pickle/"

dataset="imdb"
batch_size=25000
batch_amount=1
use_test_set=True
shuffle=True
seed=42


# *************** GENERATE TRAINING DATA ***************
traindat_input_path="/home/tobxtra/data/verbosius/store_imdb_pickle/"
traindat_output_path="/home/tobxtra/data/verbosius/store_imdb_trainingdata/"

batchdist_n=0

# TM PARAMS
MAX_DF=0.5263
MIN_DF=21
MAX_FEATURES=5000
N_GRAM_RANGE=(1, 2)
NUMBER_OF_CLAUSES=5853
LITERAL_BUDGET=8
S=3.4606
T=9225
TM_EPOCHS=7
