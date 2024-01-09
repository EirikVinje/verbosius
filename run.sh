#!/bin/bash

source $PWD/config.sh

echo " "
echo "chunkdist: $chunkdist_n"
echo "dataset: $dataset"
echo "Chunk size: $chunk_size"
echo "Chunk amount: $chunk_amount"
echo " "

# python verbosius/chunking/stage_chunks.py --dataset $dataset --chunk_size $chunk_size --chunk_amount $chunk_amount --chunkdist_n $chunkdist_n 
# python verbosius/preprocessing/stage_preprocess.py --dataset $dataset --chunkdist_n $chunkdist_n
# CUDA_VISIBLE_DEVICES=1 python verbosius/trainingdata/stage_trainingdata.py --dataset $dataset  --chunkdist_n $chunkdist_n
# CUDA_VISIBLE_DEVICES=1 python verbosius/xai_transformer/stage_transformer.py --dataset $dataset --chunkdist_n $chunkdist_n 
# CUDA_VISIBLE_DEVICES=1 python verbosius/make_readme_from_run.py --dataset $dataset --chunkdist_n $chunkdist_n
