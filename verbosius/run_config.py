import os


# ********** Global ********** #
dataset = "imdb"
user = os.environ.get("USER")
chunkdist_n = 92384576


# *********** Chunker ********** #

input_raw = f"/home/{user}/data/verbosius/imdb/"
output_chunk = f"/home/{user}/data/verbosius/imdb/chunking/"
chunk_size = 100
chunk_amount = 3

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

