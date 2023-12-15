#!/bin/bash

# *************** Global *************** 

dataset="$(python -c 'import run_config; print(run_config.dataset)')"
chunkdist_n="$(python -c 'import run_config; print(run_config.chunkdist_n)')"

# *********** Chunking ********** #
input_raw="$(python -c 'import run_config; print(run_config.input_raw)')" 
output_chunk="$(python -c 'import run_config; print(run_config.output_chunk)')"
chunk_size="$(python -c 'import run_config; print(run_config.chunk_size)')"
chunk_amount="$(python -c 'import run_config; print(run_config.chunk_amount)')"

# *********** Preprocess ********** #
input_chunk="$(python -c 'import run_config; print(run_config.input_chunk)')"
output_preproc="$(python -c 'import run_config; print(run_config.output_preproc)')"

# *********** Trainingdata ********** #
input_preproc="$(python -c 'import run_config; print(run_config.input_preproc)')"
output_traindata="$(python -c 'import run_config; print(run_config.output_traindata)')"

# *********** Transformer ********** #
input_traindata="$(python -c 'import run_config; print(run_config.input_traindata)')"
input_testdata="$(python -c 'import run_config; print(run_config.input_testdata)')"
model_output="$(python -c 'import run_config; print(run_config.model_output)')"

# *********** XAI score ********** #
input_xai_val_model="$(python -c 'import run_config; print(run_config.input_xai_val_model)')" 
model_name="$(python -c 'import run_config; print(run_config.model_name)')"
