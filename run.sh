#!/bin/bash -e

part_n=6165
n_chunks=25
size="small" 

python verbosius/chunking/chunker.py --n_chunks $n_chunks --part_n $part_n --size $size 
python verbosius/preprocess/preprocess.py --part_n $part_n 
python verbosius/trainingdata/weighter.py --part_n $part_n
python verbosius/train_eval_tokenize/trainingdata.py --part_n $part_n

model_name="model_02_04"
checkpoint=0

CUDA_VISIBLE_DEVICES=0 python verbosius/xai_transformer/transformer.py --part_n $part_n --model_name $model_name 

# CUDA_VISIBLE_DEVICES=0 python verbosius/xai_transformer/performance.py --model_name $model_name --size $size 

