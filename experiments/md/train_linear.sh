#!/bin/bash

source ../base_config.sh

# export CUDA_VISIBLE_DEVICES=$1
export _TYPER_STANDARD_TRACEBACK=1

file=$(basename "$0")
filename="${file%%.*}"
output_dir="$results_dir/$filename-$current_date/"

for cmodel in $linear_cls_models; do
  eval python $py_script\
    --output-dir=$output_dir\
    --no-save-checkpoints\
    --hyperopt\
    --hyperopt-trials=$neural_trials\
    --seeds=$seeds\
    --train-path=$train_path\
    --val-path=$val_path\
    --test-path=$test_path\
    --retrieval-dataset-path=$retrieval_dataset_path\
    --model-type=$cmodel\
    --retrieval-model-name-or-path=$en_ret_models":"$mul_ret_models\
    --retrieval-weight=$retrieval_weights\
    --eval-attribute-cols=$eval_attribute_cols
done