import os
from transformers import AutoTokenizer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

# *************** INIT ***************

seed = 42

user = os.environ.get("USER")
root = f"/home/{user}/data/verbosius/amazon" 
TM_EPOCHS = 1
num_train_epochs = 1

num_tok_labels = 3 # transformer
num_seq_labels = 5 # transformer
NUM_TM_LABELS = 3 # TM

# *************** GENERATE TRAINING DATA *************** #

ERROR_MAX_DF = 0.67
ERROR_MAX_FEATURES = 900
ERROR_MIN_DF = 9
ERROR_NUMBER_OF_CLAUSES = 3700
ERROR_S = 26.200000000000003
ERROR_T = 4800
MAX_DF = 0.54
MAX_FEATURES = 1000
MIN_DF = 9
NUMBER_OF_CLAUSES = 6300
S = 28.400000000000002
SKB_score_func = f_classif
STOPWORDS = None
T = 4600

N_JOBS = 5
EARLY_STOP_ACC=1.0
CV_MAX_FEATURES=60000
N_GRAM_RANGE=(1, 2)
LITERAL_BUDGET=6
ERROR_LITERAL_BUDGET = 6

# *************** TRANSFORMER *************** #

device="cuda"
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", add_prefix_space=True, device=device)
evaluation_strategy = "epoch"
save_strategy = "epoch"
load_best_model_at_end = True
label_names = ['labels', 'sentiment']

trainer_batch_size = 8

learning_rate = 1.9729419811296342e-05
neutral_weight = 0.0419 
loss_weight = 8.2






