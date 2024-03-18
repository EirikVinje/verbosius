import os
from transformers import AutoTokenizer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

# *************** INIT ***************

seed = 69

dataset = "amazon"
chunkdist_n = 6165
user = os.environ.get("USER")
root = f"/home/{user}/data/verbosius/" 
chunk_size = 8000
chunk_amount = 75
TM_EPOCHS = 50
num_train_epochs = 10

num_labels = 3
num_seq_labels = 5
NUM_TM_LABELS = 3


# *************** PREPROSESSING *************** 

shuffle=True
test_size=0.2
val_size=0.2
validation=False

# *************** GENERATE TRAINING DATA ***************

MAX_FEATURES = 1000
MAX_DF = 0.54
MIN_DF = 9
NUMBER_OF_CLAUSES = 6300
S = 28.4
T = 4600
ERROR_MAX_FEATURES = 900
ERROR_NUMBER_OF_CLAUSES = 3700
ERROR_S = 26.2
ERROR_T = 4800
ERROR_MAX_DF = 0.51
ERROR_MIN_DF = 9
SKB_score_func = f_classif

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
evaluation_strategy = "epoch"
save_strategy = "epoch"
load_best_model_at_end = True
label_names = ['labels', 'sentiment']

per_device_eval_batch_size = 8
per_device_train_batch_size = 8
learning_rate =  1.9729419811296342e-05
neutral_weight = 0.0419
loss_weight = 8.2

# *************** TESTING ***************





