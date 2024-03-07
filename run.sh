#!/bin/bash

# source $PWD/config.sh

chunkdist_n=2000
dataset="amazon"
chunk_size=8000
chunk_amount=3
size="big"

echo " "
echo "Running $size $dataset"
echo " "
echo "chunkdist_id: $chunkdist_n"
echo "Chunk size: $chunk_size"
echo "Chunk amount: $chunk_amount"
echo " "

# python verbosius/chunking/stage_chunks.py --dataset "$dataset" --chunk_size "$chunk_size" --chunk_amount "$chunk_amount" --chunkdist_n "$chunkdist_n" --size "$size" &&
# python verbosius/preprocess/stage_preprocess.py --dataset "$dataset" --chunkdist_n "$chunkdist_n" &&

python verbosius/trainingdata/stage_trainingdata.py --dataset "$dataset" --chunkdist_n "$chunkdist_n"
CUDA_VISIBLE_DEVICES=0 python verbosius/xai_transformer/stage_transformer.py --dataset $dataset --chunkdist_n $chunkdist_n


# CUDA_VISIBLE_DEVICES=1 python verbosius/xai_transformer/model_accuracy.py --dataset $dataset --chunkdist_n $chunkdist_n --size $size
# CUDA_VISIBLE_DEVICES=0 python verbosius/xai_transformer/custom_trainloop.py --dataset $dataset --chunkdist_n $chunkdist_n 

