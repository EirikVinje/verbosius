#!/bin/bash

# Most parameters are input in verbosius/config.py, as this allows for easy
# use in python scripts. This file is used to convert the python variables
# to bash variables, so that they can be used in run.sh for when you run the pipeline
# ONLY add variables here if they are not used in python scripts, else keep them
# in config.py, and add conversion here, as done for the other variables.

# *************** PREPROSESSING *************** 

preproc_input_path="$(python -c 'import config; print(config.preproc_input_path)')"
preproc_output_path="$(python -c 'import config; print(config.preproc_output_path)')"

dataset="$(python -c 'import config; print(config.dataset)')"
batch_size="$(python -c 'import config; print(config.batch_size)')"
batch_amount="$(python -c 'import config; print(config.batch_amount)')"
use_test_set="$(python -c 'import config; print(config.use_test_set)')"
batch_size_test="$(python -c 'import config; print(config.batch_size_test)')"
shuffle="$(python -c 'import config; print(config.shuffle)')"
seed="$(python -c 'import config; print(config.seed)')"


# *************** GENERATE TRAINING DATA ***************
traindat_input_path="$(python -c 'import config; print(config.traindat_input_path)')"
traindat_output_path="$(python -c 'import config; print(config.traindat_output_path)')"

batchdist_n="$(python -c 'import config; print(config.batchdist_n)')"

batchdist_range="$(python -c 'import config; print(config.batchdist_range)')"

final_input_path="$(python -c 'import config; print(config.final_input_dir)')"
final_output_path="$(python -c 'import config; print(config.final_output_dir)')"
save_model="$(python -c 'import config; print(config.save_model)')"
