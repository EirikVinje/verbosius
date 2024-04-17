#!/bin/bash -e

part_n=3939
ds_path="/home/bigtech/data/verbosius/amazon/nc_2"
fw=1
pb=1

python verbosius/preprocess.py --part_n $part_n --ds_path $ds_path --fw $fw --pb $pb
python verbosius/weighter.py --part_n $part_n --fw $fw --pb $pb
python verbosius/trainingdata.py --part_n $part_n --fw $fw --pb $pb

model_name="model_02_04"
checkpoint=0

# CUDA_VISIBLE_DEVICES=0 python verbosius/transformer.py --part_n $part_n --model_name $model_name
# CUDA_VISIBLE_DEVICES=0 python verbosius/performance.py --model_name $model_name --size $size 

# CUDA_VISIBLE_DEVICES=1 python hyperparam/transformer_hp.py



