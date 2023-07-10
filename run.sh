#!/bin/bash


python verbosius/preprocessing/stager.py --dataset imdb --batch_size 100 --batch_amount 10 --input /home/tobxtra/data/verbosius/imdb/ --output /home/tobxtra/data/verbosius/store_imdb_pickle/ --use_test_set True --shuffle True --seed 42
python verbosius/trainingdata/stage_trainingdata.py --dataset imdb --batchdist_n 0 --input /home/tobxtra/data/verbosius/store_imdb_pickle/ --output /home/tobxtra/data/verbosius/store_imdb_trainingdata/
