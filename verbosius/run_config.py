import os


# ********** Global ********** #
dataset = "imdb"
user = os.environ.get("USER")


# *********** Chunker ********** #
input_raw = f"/home/{user}/data/verbosius/imdb/testing/"
output_chunk = f"/home/{user}/data/verbosius/imdb/testing/chunking/"
chunk_size = 100
chunk_amount = 3

# *********** Preprocess ********** #
input_chunk = f"/home/{user}/data/verbosius/imdb/testing/chunking/"
output_preproc = f"/home/{user}/data/verbosius/imdb/testing/preprocess/"

# *********** Trainingdata ********** #
input_preproc = f"/home/{user}/data/verbosius/imdb/testing/preprocess/"
output_traindata = f"/home/{user}/data/verbosius/imdb/testing/trainingdata/"

# *********** Transformer ********** #
input_traindata = f"/home/{user}/data/verbosius/imdb/testing/trainingdata"
input_testdata = f"/home/{user}/data/verbosius/imdb/testing/chunking"
model_output = f"/home/{user}/data/verbosius/imdb/testing/models"

