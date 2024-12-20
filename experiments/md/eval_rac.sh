#!/bin/bash

source ../base_config.sh

# export CUDA_VISIBLE_DEVICES=$1
export _TYPER_STANDARD_TRACEBACK=1

file=$(basename "$0")
filename="${file%%.*}"
output_dir="$results_dir/$filename-$current_date/"

best_ft_model_path=""
best_knn_model_path=""

python simple_experiments/retrieval_augmented_classification.py \
    --train-path=$train_path \
    --test-path=$test_path \
    --output-dir=$output_dir \
    --knn-model-path=$best_knn_model_path \
    --knn-featurizer="tf-idf" \
    --transformer-model-name-or-path=$best_ft_model_path