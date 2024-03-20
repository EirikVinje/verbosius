#!/bin/bash -e

# source $PWD/config.sh

part_n=6165
n_chunks=2
size="small"

model_name="model_t"
checkpoint=0


# python verbosius/chunking/chunker.py --n_chunks $n_chunks --part_n $part_n --size $size 

python verbosius/preprocess/preprocess.py --part_n $part_n 

# python verbosius/trainingdata/weighter.py --part_n $part_n


# python verbosius/train_eval_tokenize/stage_train_eval_tokenize.py --dataset $dataset --chunkdist_n $chunkdist_n
# CUDA_VISIBLE_DEVICES=0 python verbosius/xai_transformer/stage_transformer_old.py --dataset $dataset --chunkdist_n $chunkdist_n
# CUDA_VISIBLE_DEVICES=1 python verbosius/xai_transformer/model_accuracy.py --chunkdist_n $chunkdist_n --model_name $model_name --checkpoint $checkpoint

# CUDA_VISIBLE_DEVICES=0 python verbosius/xai_transformer/stage_transformer.py --dataset $dataset --chunkdist_n $chunkdist_n


