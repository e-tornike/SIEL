#!/bin/bash

source ../base_config.sh

# export CUDA_VISIBLE_DEVICES=$1
export _TYPER_STANDARD_TRACEBACK=1

file=$(basename "$0")
filename="${file%%.*}"
output_dir="$results_dir/$filename-$current_date/"

mul_ret_models="intfloat/multilingual-e5-large"
IFS=':' read -ra cmodels <<< $mul_ret_models

for cmodel in "${cmodels[@]}"; do
    for model_type in $cluster_cls_models; do
        eval python $py_script\
            --output-dir=$output_dir\
            --no-save-checkpoints\
            --save-best-model\
            --hyperopt\
            --hyperopt-trials=$linear_trials\
            --seeds="42"\
            --train-path=$train_path\
            --val-path=$val_path\
            --test-path=$test_path\
            --retrieval-dataset-path=$retrieval_dataset_path\
            --model-type=$model_type\
            --classification-model-name-or-path=$cmodel\
            --retrieval-model-name-or-path=$mul_ret_models\
            --retrieval-weight=$retrieval_weights\
            --eval-attribute-cols=$eval_attribute_cols
    done
done

for model_type in $cluster_cls_models; do
    eval python $py_script\
        --output-dir=$output_dir\
        --no-save-checkpoints\
        --save-best-model\
        --hyperopt\
        --hyperopt-trials=$linear_trials\
        --seeds="42"\
        --train-path=$train_path\
        --val-path=$val_path\
        --test-path=$test_path\
        --retrieval-dataset-path=$retrieval_dataset_path\
        --model-type=$model_type\
        --no-cluster-embedding-vectorizer\
        --classification-model-name-or-path="tfidf"\
        --eval-attribute-cols=$eval_attribute_cols
done