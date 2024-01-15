#!/bin/bash

echo " "
echo "Running hp search "
echo " "

CUDA_VISIBLE_DEVICES=0 python verbosius/hyperparam/TM_hp.py
# CUDA_VISIBLE_DEVICES=1 python verbosius/hyperparam/transformer_hp.py 