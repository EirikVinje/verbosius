from transformers import AutoTokenizer
import os

# *************** PREPROSESSING *************** 


dataset="sst5"
root = os.path.expanduser("~/data/verbosius")
preproc_input_path= os.path.join(root, f"{dataset}/{dataset}_raw")
preproc_output_path= os.path.join(root, f"{dataset}/preprocess")

batch_size=8544
batch_amount=1
use_test_set=True
batch_size_test = 2210
shuffle=True
seed=42


# *************** GENERATE TRAINING DATA ***************

traindat_input_path= os.path.join(root, f"{dataset}/preprocess")
traindat_output_path= os.path.join(root, f"{dataset}/trainingdata")

batchdist_n=0

MAX_DF=0.9
MIN_DF=1
MAX_FEATURES=10000
N_GRAM_RANGE=(1, 2)
NUMBER_OF_CLAUSES=9000
LITERAL_BUDGET=8
S=3.4606
T=9225
TM_EPOCHS=4
EARLY_STOP_ACC=0.86
STOPWORDS=None
N_JOBS = 5


# *************** TRANSFORMER ***************

device="cuda"
model_name_ = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_, add_prefix_space=True, device=device)

save_model = True

batchdist_range = "\(1,2\)"
final_input_dir = os.path.join(root, f"{dataset}/trainingdata")
final_output_dir = os.path.join(root, f"{dataset}/models")


learning_rate = 1.539e-5
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
num_train_epochs = 1
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
