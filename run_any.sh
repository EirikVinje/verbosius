#!/bin/bash -e

# python verbosius/chunker.py  --n_chunks 5 --n_reads 1000000
# python verbosius/chunker.py  --n_chunks 10 --n_reads 1000000 
# python verbosius/chunker.py  --n_chunks 40 --n_reads 10000000   

# CUDA_VISIBLE_DEVICES=0 python hyperparam/transformer_hp.py --n_chunks 5 --part_n 234 --n_trials 20 --study_name "transformer_search_nc_5"
# CUDA_VISIBLE_DEVICES=0 python hyperparam/transformer_hp.py --n_chunks 10 --part_n 234 --n_trials 20 --study_name "transformer_search_nc_10"
CUDA_VISIBLE_DEVICES=0 python hyperparam/transformer_hp.py --n_chunks 40 --part_n 567 --n_trials 100 --study_name "transformer_search_nc_40"