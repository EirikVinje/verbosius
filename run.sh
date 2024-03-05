#!/bin/bash

# source $PWD/config.sh

chunkdist_n=9999
dataset="amazon"
chunk_size=5000
chunk_amount=3
size="small"


echo " "
echo "Running $size $dataset"
echo " "
echo "chunkdist_id: $chunkdist_n"
echo "Chunk size: $chunk_size"
echo "Chunk amount: $chunk_amount"
echo " "

set -e

handle_error() {
    echo "An error occurred. Exiting..."
    exit 1
}

trap 'handle_error' ERR

python verbosius/chunking/stage_chunks.py --dataset "$dataset" --chunk_size "$chunk_size" --chunk_amount "$chunk_amount" --chunkdist_n "$chunkdist_n" --size "$size" &&
python verbosius/preprocess/stage_preprocess.py --dataset "$dataset" --chunkdist_n "$chunkdist_n" &&

python verbosius/trainingdata/stage_trainingdata.py --dataset "$dataset" --chunkdist_n "$chunkdist_n" &&
if [ $? -ne 0 ]; then
    rm -rf /home/bigtech/data/verbosius/amazon/preprocess/${dataset}_chunkdist_${chunkdist_n}/*e.pkl 
    echo "Deleted error chunks in preprocess files"
else
    echo "Normal exit"
fi

# CUDA_VISIBLE_DEVICES=0 python verbosius/xai_transformer/stage_transformer.py --dataset $dataset --chunkdist_n $chunkdist_n 
# CUDA_VISIBLE_DEVICES=1 python verbosius/xai_transformer/model_accuracy.py --dataset $dataset --chunkdist_n $chunkdist_n

