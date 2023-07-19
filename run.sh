#!/bin/bash

source $PWD/config.sh

#python verbosius/preprocessing/stager.py --dataset $dataset --batch_size $batch_size --batch_amount $batch_amount --input $preproc_input_path --output $preproc_output_path --use_test_set $use_test_set --batch_size_test $batch_size_test --shuffle $shuffle --seed $seed
python verbosius/trainingdata/stage_trainingdata.py --dataset $dataset --batchdist_n $batchdist_n --input $traindat_input_path --output $traindat_output_path
python verbosius/exp_transformer/stage_transformer.py --dataset $dataset --input $final_input_path --output $final_output_path --n_batchdist $batchdist_range --save_model true

