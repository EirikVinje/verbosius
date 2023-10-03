from transformers import AutoTokenizer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
import os

# *************** GLOBAL ***************

seed = 69

# *************** PREPROSESSING *************** 

shuffle=True
seed=42
test_size=0.2
val_size=0.2
validation=True

# *************** GENERATE TRAINING DATA ***************

MAX_DF=0.9
MIN_DF=10
MAX_FEATURES=100
CV_MAX_FEATURES=30000
N_GRAM_RANGE=(1, 2)
NUMBER_OF_CLAUSES=50
LITERAL_BUDGET=10
S=10
T=100
TM_EPOCHS=15
EARLY_STOP_ACC=0.86
STOPWORDS=None
N_JOBS = 5

SKB_score_func = chi2
error_chunk=True
n_badtexts=2000

ERROR_MAX_FEATURES=100
ERROR_NUMBER_OF_CLAUSES=50
ERROR_S = 10
ERROR_T = 100
ERROR_LITERAL_BUDGET = 10
ERROR_MAX_DF = 0.9
ERROR_MIN_DF = 10

# *************** TRANSFORMER ***************

device="cuda"
model_name_ = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_, add_prefix_space=True, device=device)

learning_rate = 1.539e-5
per_device_train_batch_size = 32
per_device_eval_batch_size = 32
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
