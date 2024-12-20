#!/bin/bash

source ../base_config.sh

# export CUDA_VISIBLE_DEVICES=$1
export _TYPER_STANDARD_TRACEBACK=1

file=$(basename "$0")
filename="${file%%.*}"
output_dir="$results_dir/$filename-$current_date/"

mul_ret_models="intfloat/multilingual-e5-large"
IFS=':' read -ra cmodels <<< $mul_ret_models

sim_data_paths="/data/gsim/400k.tsv /data/llm_gen/400k.tsv"

for sim_path in $sim_data_paths; do
    for cmodel in "${cmodels[@]}"; do
        for model_type in $cluster_cls_models; do
            eval python $py_script\
                --output-dir=$output_dir\
                --no-save-checkpoints\
                --save-best-model\
                --hyperopt\
                --hyperopt-trials=$linear_trials\
                --n-folds=$linear_folds\
                --cross-validation\
                --seeds="42"\
                --langs="en,de"\
                --zero-shot-langs=""\
                --train-path=$project_dir$sim_path\
                --val-path=$val_path\
                --test-path=$test_path\
                --retrieval-dataset-path=$retrieval_dataset_path\
                --model-type=$model_type\
                --classification-model-name-or-path=$cmodel\
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
            --n-folds=$linear_folds\
            --cross-validation\
            --seeds="42"\
            --langs="en,de"\
            --zero-shot-langs=""\
            --train-path=$project_dir$sim_path\
            --val-path=$val_path\
            --test-path=$test_path\
            --retrieval-dataset-path=$retrieval_dataset_path\
            --model-type=$model_type\
            --no-cluster-embedding-vectorizer\
            --classification-model-name-or-path="tfidf"\
            --eval-attribute-cols=$eval_attribute_cols
    done
done