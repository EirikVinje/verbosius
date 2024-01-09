import os


# ********** Global ********** #
user = os.environ.get("USER")
dataset = "amazon"
chunkdist_n = 696969696
rootdir = f"/home/{user}/data/verbosius/{dataset}/"

# *********** Chunker ********** #

input_raw = f"/home/{user}/data/verbosius/{dataset}/"
output_chunk = f"/home/{user}/data/verbosius/{dataset}/chunking/"
chunk_size = 8000
chunk_amount = 125

# *********** Preprocess ********** #
input_chunk = f"/home/{user}/data/verbosius/{dataset}/chunking/"
output_preproc = f"/home/{user}/data/verbosius/{dataset}/preprocess/"

# *********** Trainingdata ********** #
input_preproc = f"/home/{user}/data/verbosius/{dataset}/preprocess/"
output_traindata = f"/home/{user}/data/verbosius/{dataset}/trainingdata/"

# *********** Transformer ********** #
input_traindata = f"/home/{user}/data/verbosius/{dataset}/trainingdata"
model_output = f"/home/{user}/data/verbosius/{dataset}/models"

# *********** accuracy score ********** #
input_testdata = f"/home/{user}/data/verbosius/{dataset}/chunking"

# *********** XAI score ********** #
input_xai_val_model = f"/home/{user}/data/verbosius/{dataset}/models"
model_name = f"{dataset}_model_dist_{chunkdist_n}"
