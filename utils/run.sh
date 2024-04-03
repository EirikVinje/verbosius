#!/bin/bash -e

part_n=6165
n_chunks=2
size="small" 

# python verbosius/chunker.py --n_chunks $n_chunks --part_n $part_n --size $size 
# python verbosius/preprocess.py --part_n $part_n 
# python verbosius/weighter.py --part_n $part_n
# python verbosius/trainingdata.py --part_n $part_n

model_name="model_02_04"
checkpoint=0

CUDA_VISIBLE_DEVICES=0 python verbosius/transformer.py --part_n $part_n --model_name $model_name
# CUDA_VISIBLE_DEVICES=0 python verbosius/performance.py --model_name $model_name --size $size 




