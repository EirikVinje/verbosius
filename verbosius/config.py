from transformers import AutoTokenizer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
import os

# *************** GLOBAL ***************

seed = 69

# *************** PREPROSESSING *************** 

shuffle=True
test_size=0.2
val_size=0.2
validation=True

# *************** GENERATE TRAINING DATA ***************

MAX_FEATURES = 1750
MAX_DF = 0.7086319286046587
MIN_DF = 22
NUMBER_OF_CLAUSES = 4000
S = 17.7
T = 5000
TM_EPOCHS = 15
ERROR_MAX_FEATURES = 700
ERROR_NUMBER_OF_CLAUSES = 3200
ERROR_S = 25.61065
ERROR_T = 1750
ERROR_MAX_DF = 0.437663961421369
ERROR_MIN_DF = 30
SKB_score_func = mutual_info_classif

STOPWORDS = None
N_JOBS = 5
EARLY_STOP_ACC=1.0
error_chunk=True
n_badtexts=2000
CV_MAX_FEATURES=40000
N_GRAM_RANGE=(1, 2)
LITERAL_BUDGET=6
ERROR_LITERAL_BUDGET = 6

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
neutral_weight = 0.0001 
loss_weight = 5.0
num_labels = 3
num_seq_labels = 2

# *************** TESTING ***************
#MAX_DF = 0.9
#MIN_DF = 2
#MAX_FEATURES = 50
#NUMBER_OF_CLAUSES = 100
#TM_EPOCHS = 2
#ERROR_MAX_FEATURES = 50
#ERROR_NUMBER_OF_CLAUSES = 100
#ERROR_MAX_DF = 0.9
#ERROR_MIN_DF = 2
