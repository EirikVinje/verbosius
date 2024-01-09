#!/bin/bash

source $PWD/config.sh

CUDA_VISIBLE_DEVICES=1 python verbosius/xai_transformer/stage_transformer.py --dataset $dataset --input_traindata $input_traindata --model_output $model_output --chunkdist_n $chunkdist_n 
CUDA_VISIBLE_DEVICES=1 python verbosius/make_readme_from_run.py --dataset $dataset --model_path $model_output --chunkdist_n $chunkdist_n

# python verbosius/xai_validation/stage_validation.py --model_path $input_xai_val_model --model_name $model_name --batch_size_pred $batch_size
