#!/bin/bash -e

# source $PWD/config.sh

chunkdist_n=6165
dataset="amazon"
chunk_size=8000
chunk_amount=75
size="big"

model_name="model_t"
checkpoint=0


python verbosius/chunking/stage_chunks.py --dataset $dataset --chunk_size $chunk_size --chunk_amount $chunk_amount --chunkdist_n $chunkdist_n --size $size 
python verbosius/preprocess/stage_preprocess.py --dataset $dataset --chunkdist_n $chunkdist_n 
python verbosius/trainingdata/stage_trainingdata.py --dataset $dataset --chunkdist_n $chunkdist_n
python verbosius/train_eval_tokenize/stage_train_eval_tokenize.py --dataset $dataset --chunkdist_n $chunkdist_n
CUDA_VISIBLE_DEVICES=0 python verbosius/xai_transformer/stage_transformer_old.py --dataset $dataset --chunkdist_n $chunkdist_n
CUDA_VISIBLE_DEVICES=1 python verbosius/xai_transformer/model_accuracy.py --chunkdist_n $chunkdist_n --model_name $model_name --checkpoint $checkpoint

# CUDA_VISIBLE_DEVICES=0 python verbosius/xai_transformer/stage_transformer.py --dataset $dataset --chunkdist_n $chunkdist_n


