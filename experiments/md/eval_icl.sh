#!/bin/bash

source ../base_config.sh

# export CUDA_VISIBLE_DEVICES=$1
export _TYPER_STANDARD_TRACEBACK=1

file=$(basename "$0")
filename="${file%%.*}"
output_dir="$results_dir/$filename-$current_date/"

for llm in $llms; do
    eval python $llm_py_script\
    --output-dir=$output_dir\
    --seeds=$seeds\
    --train-path=$train_path\
    --test-path=$test_path\
    --model-name-or-path=$llm
done