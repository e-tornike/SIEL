#!/bin/bash

project_dir=""
results_dir="$project_dir/results"
current_date=$(date '+%Y-%m-%d_%H-%M')

py_script="$project_dir/sil/train_md.py"
llm_py_script="$project_dir/sil/run_icl.py"
ed_script="$project_dir/sil/run_ed.py"

train_path=$project_dir"/data/sild/diff_train.tsv"
val_path=$project_dir"/data/sild/diff_val.tsv"
test_path=$project_dir"/data/sild/diff_test.tsv"

sim_data_paths=$project_dir"/data/da/llm_gen/400k.tsv"

rand_train_path=$project_dir"/data/sild/rand_train.tsv"
rand_val_path=$project_dir"/data/sild/rand_val.tsv"
rand_test_path=$project_dir"/data/sild/rand_test.tsv"

en_cls_models="google-bert/bert-base-uncased FacebookAI/roberta-base allenai/scibert_scivocab_uncased allenai/specter KM4STfulltext/SSCI-BERT-e2 distilbert/distilbert-base-uncased"
mul_cls_models="google-bert/bert-base-multilingual-uncased FacebookAI/xlm-roberta-base FacebookAI/xlm-roberta-large"
linear_cls_models="logistic_regression linear_svm svm"
cluster_cls_models="knn"

llms="mistralai/Mistral-7B-Instruct-v0.2"

seeds="42,1234,1337,9999,98765"

neural_folds=1
neural_trials=20
neural_epochs=20

linear_folds=10
linear_trials=100

eval_attribute_cols="lang,type_short,subtype_short"