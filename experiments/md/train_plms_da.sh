#!/bin/bash

source ../base_config.sh

# export CUDA_VISIBLE_DEVICES=$1
export _TYPER_STANDARD_TRACEBACK=1

file=$(basename "$0")
filename="${file%%.*}"
output_dir="$results_dir/$filename-$current_date/"

cmodel="FacebookAI/xlm-roberta-base"

neural_trials=20
neural_epochs=1

synthetic_paths="data/da/llm_gen/100k.tsv data/da/llm_gen/200k.tsv data/da/llm_gen/400k.tsv"
for spath in $synthetic_paths; do
  eval python $py_script\
    --output-dir=$output_dir\
    --no-save-checkpoints\
    --hyperopt\
    --hyperopt-trials=$neural_trials\
    --epochs=$neural_epochs\
    --seeds=$seeds\
    --train-path=$spath\
    --val-path=$val_path\
    --test-path=$test_path\
    --retrieval-dataset-path=$retrieval_dataset_path\
    --classification-model-name-or-path=$cmodel\
    --eval-attribute-cols=$eval_attribute_cols
done