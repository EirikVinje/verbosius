#!/bin/bash

# source $PWD/config.sh

chunkdist_n=1
dataset="amazon"
chunk_size=8000
chunk_amount=1
size="small"

echo " "
echo "chunkdist: $chunkdist_n"
echo "dataset: $dataset"
echo "Chunk size: $chunk_size"
echo "Chunk amount: $chunk_amount"
echo "size : $size"
echo " "

python verbosius/chunking/stage_chunks.py --dataset $dataset --chunk_size $chunk_size --chunk_amount $chunk_amount --chunkdist_n $chunkdist_n --size $size 
python verbosius/preprocessing/stage_preprocess.py --dataset $dataset --chunkdist_n $chunkdist_n
python verbosius/trainingdata/stage_trainingdata.py --dataset $dataset  --chunkdist_n $chunkdist_n

chunkdist_n=2
dataset="amazon"
chunk_size=8000
chunk_amount=1
size="small"

echo " "
echo "chunkdist: $chunkdist_n"
echo "dataset: $dataset"
echo "Chunk size: $chunk_size"
echo "Chunk amount: $chunk_amount"
echo " "

python verbosius/chunking/stage_chunks.py --dataset $dataset --chunk_size $chunk_size --chunk_amount $chunk_amount --chunkdist_n $chunkdist_n --size $size 
python verbosius/preprocessing/stage_preprocess.py --dataset $dataset --chunkdist_n $chunkdist_n
python verbosius/trainingdata/stage_trainingdata.py --dataset $dataset  --chunkdist_n $chunkdist_n


# CUDA_VISIBLE_DEVICES=0 python verbosius/xai_transformer/stage_transformer.py --dataset $dataset --chunkdist_n $chunkdist_n 
# CUDA_VISIBLE_DEVICES=1 python verbosius/xai_transformer/model_accuracy.py --dataset $dataset --chunkdist_n $chunkdist_n
