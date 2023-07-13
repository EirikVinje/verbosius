#!/bin/bash

source /home/bigtech/projects/verbosius/config.sh


python verbosius/preprocessing/stager.py --dataset $dataset --batch_size $batch_size --batch_amount $batch_amount --input $preproc_input_path --output $preproc_output_path --use_test_set $use_test_set --shuffle $shuffle --seed $seed
python verbosius/trainingdata/stage_trainingdata.py --dataset $dataset --batchdist_n $batchdist_n --input $traindat_input_path --output $traindat_output_path
