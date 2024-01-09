#!/bin/bash

source $PWD/config.sh

python verbosius/chunking/stage_chunks.py --dataset $dataset --input $input_raw --output $output_chunk --chunk_size $chunk_size --chunk_amount $chunk_amount --chunkdist_n $chunkdist_n 
python verbosius/preprocessing/stage_preprocess.py --dataset $dataset --input $input_chunk --output $output_preproc --chunkdist_n $chunkdist_n
CUDA_VISIBLE_DEVICES=1 python verbosius/trainingdata/stage_trainingdata.py --dataset $dataset --input $input_preproc --output $output_traindata --chunkdist_n $chunkdist_n
