from transformers import AutoTokenizer
import os

# *************** GLOBAL ***************
seed = 42

# *************** PREPROSESSING *************** 

batch_size=8544
batch_amount=1
use_test_set=True
batch_size_test = 2210
shuffle=True
seed=42
test_size=0.2
val_size=0.2
validation=True

# *************** GENERATE TRAINING DATA ***************

batchdist_n=0

MAX_DF=0.6
MIN_DF=5
MAX_FEATURES=5000
N_GRAM_RANGE=(1, 2)
NUMBER_OF_CLAUSES=9000
LITERAL_BUDGET=8
S=3.4606
T=9225
TM_EPOCHS=5
EARLY_STOP_ACC=0.86
STOPWORDS=None
N_JOBS = 5

error_chunk=True
n_badtexts=2000

# *************** TRANSFORMER ***************

device="cuda"
model_name_ = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_, add_prefix_space=True, device=device)

learning_rate = 1.539e-5
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
num_train_epochs = 5
weight_decay = 0.01
evaluation_strategy = "epoch"
save_strategy = "epoch"
warmup_steps = 500
load_best_model_at_end = True
eval_accumulation_steps = 16
label_names = ['labels', 'sentiment']
neutral_weight = 0.3897 
loss_weight = 0.4154 
num_labels = 3
num_seq_labels = 2
