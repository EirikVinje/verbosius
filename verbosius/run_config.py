import os


# ********** Global ********** #
dataset = "imdb"
user = os.environ.get("USER")
chunkdist_n = 5555

# *********** Chunker ********** #

input_raw = f"/home/{user}/data/verbosius/imdb/"
output_chunk = f"/home/{user}/data/verbosius/imdb/chunking/"
chunk_size = 8000
chunk_amount = 5

# *********** Preprocess ********** #
input_chunk = f"/home/{user}/data/verbosius/imdb/chunking/"
output_preproc = f"/home/{user}/data/verbosius/imdb/preprocess/"

# *********** Trainingdata ********** #
input_preproc = f"/home/{user}/data/verbosius/imdb/preprocess/"
output_traindata = f"/home/{user}/data/verbosius/imdb/trainingdata/"

# *********** Transformer ********** #
input_traindata = f"/home/{user}/data/verbosius/imdb/trainingdata"
input_testdata = f"/home/{user}/data/verbosius/imdb/chunking"
model_output = f"/home/{user}/data/verbosius/imdb/models"

# *********** XAI score ********** #
input_xai_val_model = f"/home/{user}/data/verbosius/imdb/models"
model_name = f"{dataset}_model_dist_{chunkdist_n}"
batch_size = 32
