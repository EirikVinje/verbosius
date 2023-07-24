from transformers import AutoTokenizer
import os

# *************** PREPROSESSING *************** 

root = os.path.expanduser("~/data/verbosius")
preproc_input_path= os.path.join(root, "imdb")
preproc_output_path= os.path.join(root, "store_imdb_pickle")

dataset="imdb"
batch_size=100
batch_amount=3
use_test_set=True
batch_size_test = 30
shuffle=True
seed=42


# *************** GENERATE TRAINING DATA ***************

traindat_input_path= os.path.join(root, "store_imdb_pickle")
traindat_output_path= os.path.join(root, "store_imdb_trainingdata")

batchdist_n=2


# TM PARAMS
MAX_DF=0.6
MIN_DF=15
MAX_FEATURES=5000
N_GRAM_RANGE=(1, 2)
NUMBER_OF_CLAUSES=500 #5853
LITERAL_BUDGET=8
S=3.4606
T=9225
TM_EPOCHS=4
EARLY_STOP_ACC=0.9
STOPWORDS=None #"english"
N_JOBS = 5


# *************** TRANSFORMER ***************

device="cpu"
model_name_ = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_, add_prefix_space=True, device=device)

save_model = False

batchdist_range = "\(1,2\)"
final_input_dir = os.path.join(root, "store_imdb_trainingdata")
final_output_dir = os.path.join(root, "imdb_final_output")

learning_rate = 2e-5
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
num_train_epochs = 5
weight_decay = 0.01
evaluation_strategy = "epoch"
save_strategy = "epoch"
warmup_steps = 500
load_best_model_at_end = True
eval_accumulation_steps = 16
label_names = ['labels', 'sentiment']
neutral_weight = 0.5 #
loss_weight = 0.5 #
num_labels = 3
num_seq_labels = 2
