from transformers import AutoTokenizer
import os

# *************** PREPROSESSING *************** 

root = os.path.expanduser("~/data/verbosius")
preproc_input_path= os.path.join(root, "imdb_raw")
preproc_output_path= os.path.join(root, "imdb/preprocess")

dataset="imdb"
batch_size=500
batch_amount=4
use_test_set=True
batch_size_test = 250
shuffle=True
seed=42


# *************** GENERATE TRAINING DATA ***************

traindat_input_path= os.path.join(root, "imdb/preprocess")
traindat_output_path= os.path.join(root, "imdb/trainingdata")

batchdist_n=0

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


# *************** TRANSFORMER ***************

device="cpu"
model_name_ = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_, add_prefix_space=True, device=device)

batchdist_range = (0,5)
final_input_dir = os.path.join(root, "imdb/trainingdata")
final_output_dir = os.path.join(root, "imdb/models")

learning_rate = 2e-5
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
num_train_epochs = 1
weight_decay = 0.01
evaluation_strategy = "epoch"
save_strategy = "epoch"
warmup_steps = 500
load_best_model_at_end = True
eval_accumulation_steps = 16
label_names = ['labels', 'sentiment']
neutral_weight = 0.5
loss_weight = 0.5
num_labels = 3
num_seq_labels = 2
