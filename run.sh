#!/bin/bash

source $PWD/config.sh

# python verbosius/chunking/stage_chunks.py --dataset $dataset --input $input_raw --output $output_chunk --chunk_size $chunk_size --chunk_amount $chunk_amount --chunkdist_n $chunkdist_n 
# python verbosius/preprocessing/stage_preprocess.py --dataset $dataset --input $input_chunk --output $output_preproc --chunkdist_n $chunkdist_n
# python verbosius/trainingdata/stage_trainingdata.py --dataset $dataset --input $input_preproc --output $output_traindata --chunkdist_n $chunkdist_n
# python verbosius/tokenclassifier/stage_tokenclassifier.py --dataset $dataset --input_traindata $input_traindata --input_testdata $input_testdata --model_output $model_output --chunkdist_n $chunkdist_n 
python verbosius/xai_transformer/stage_transformer.py --dataset $dataset --input_traindata $input_traindata --input_testdata $input_testdata --model_output $model_output --chunkdist_n $chunkdist_n 
# python verbosius/xai_validation/stage_validation.py --model_path $input_xai_val_model --model_name $model_name --batch_size_pred $batch_size

# python verbosius/chunking/stage_chunks.py --dataset imdb --input /home/bigtech/data/verbosius/imdb/testing --output /home/bigtech/data/verbosius/imdb/testing/chunking --chunk_size 200 --chunk_amount 3 --chunkdist_n 999 
# python verbosius/preprocessing/stage_preprocess.py --dataset imdb --input /home/bigtech/data/verbosius/imdb/testing/chunking --output /home/bigtech/data/verbosius/imdb/testing/preprocessing --chunkdist_n 999
# python verbosius/trainingdata/stage_trainingdata.py --dataset imdb --input /home/bigtech/data/verbosius/imdb/testing/preprocessing --output /home/bigtech/data/verbosius/imdb/testing/trainingdata --chunkdist_n 999
# python verbosius/xai_transformer/stage_transformer.py --dataset imdb --train_val_input /home/bigtech/data/verbosius/imdb/testing/trainingdata --test_input /home/bigtech/data/verbosius/imdb/testing/chunking --model_output /home/bigtech/data/verbosius/imdb/testing/model --save_model True --chunkdist_n 999
