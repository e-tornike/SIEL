#!/bin/bash

source ../base_config.sh

# export CUDA_VISIBLE_DEVICES=$1
export _TYPER_STANDARD_TRACEBACK=1

file=$(basename "$0")
filename="${file%%.*}"
output_dir="$results_dir/$filename-$current_date/"

linear_cmodel="linear_svm"
linear_folds=1
linear_trials=10

eval python $py_script\
  --output-dir=$output_dir\
  --no-save-checkpoints\
  --hyperopt\
  --hyperopt-trials=$linear_trials\
  --n-folds=$linear_folds\
  --cross-validation\
  --seeds=$seeds\
  --train-path=$train_path\
  --val-path=$val_path\
  --test-path=$test_path\
  --retrieval-dataset-path=$retrieval_dataset_path\
  --model-type=$linear_cmodel\
  --eval-attribute-cols=$eval_attribute_cols

spath="data/sild/diff_train_synonyms.tsv"
eval python $py_script\
  --output-dir=$output_dir\
  --no-save-checkpoints\
  --hyperopt\
  --hyperopt-trials=$linear_trials\
  --n-folds=$linear_folds\
  --cross-validation\
  --seeds=$seeds\
  --train-path=$spath\
  --val-path=$val_path\
  --test-path=$test_path\
  --retrieval-dataset-path=$retrieval_dataset_path\
  --model-type=$linear_cmodel\
  --eval-attribute-cols=$eval_attribute_cols

spath="data/da/gsim/400k.tsv"
eval python $py_script\
  --output-dir=$output_dir\
  --no-save-checkpoints\
  --hyperopt\
  --hyperopt-trials=$linear_trials\
  --n-folds=$linear_folds\
  --cross-validation\
  --seeds=$seeds\
  --train-path=$spath\
  --val-path=$val_path\
  --test-path=$test_path\
  --retrieval-dataset-path=$retrieval_dataset_path\
  --model-type=$linear_cmodel\
  --eval-attribute-cols=$eval_attribute_cols

spath="data/da/llm_gen/400k.tsv"
eval python $py_script\
  --output-dir=$output_dir\
  --no-save-checkpoints\
  --hyperopt\
  --hyperopt-trials=$linear_trials\
  --n-folds=$linear_folds\
  --cross-validation\
  --seeds=$seeds\
  --train-path=$spath\
  --val-path=$val_path\
  --test-path=$test_path\
  --retrieval-dataset-path=$retrieval_dataset_path\
  --model-type=$linear_cmodel\
  --eval-attribute-cols=$eval_attribute_cols