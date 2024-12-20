from sklearn_linear import LinearModelTrainer
from sklearn_cluster import ClusterModelTrainer
from metrics import compute_metrics, compute_fine_grained_metrics
from data_utils import clean_df, preprocess_function, write_json

import typer
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict
import torch
from lightning.fabric.utilities.seed import seed_everything
import os
import time
from datetime import datetime


def nested_defaultdict(layer):
    def new_layer():
        return defaultdict(layer)
    return new_layer


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-12, 1e-8),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
    }


def optuna_objective(metrics):
    return metrics["eval_BinaryF1Score()"]


def train_test_split_with_dropped_instances(df, col_name, test_size, seed, _idxs=None):
    if _idxs:
        _idxs = df.loc[_idxs].dropna(subset=col_name).index.tolist()
        _labels = df.loc[_idxs]["label"].tolist()
        train_idxs, test_idxs, _, _ = train_test_split(_idxs, _labels, test_size=test_size, random_state=seed, stratify=_labels)
    else:
        _df = df.dropna(subset=col_name)
        _idxs = _df.index.tolist()
        _labels = _df["label"].tolist()
        train_idxs, test_idxs, _, _ = train_test_split(_idxs, _labels, test_size=test_size, random_state=seed, stratify=_labels)

    # add syonym instances to train
    train_source_uuids = df.loc[train_idxs][col_name].tolist()
    train_source_idxs = df[df["uuid"].isin(train_source_uuids)].index.tolist()
    assert set(train_idxs).intersection(train_source_idxs) == set()
    train_idxs = train_idxs + train_source_idxs

    # remove synonym instances from evaluation and replace with source instances
    test_source_uuids = df.loc[test_idxs][col_name].tolist()
    test_source_idxs = df[df["uuid"].isin(test_source_uuids)].index.tolist()
    assert set(train_idxs).intersection(test_source_idxs) == set()
    test_idxs = test_source_idxs

    return train_idxs, test_idxs


def kfold_train_test_split_with_dropped_instances(df, col_name, seed, n_folds, shuffle):
    _df = df.dropna(subset=col_name)
    idxs = _df.index.tolist()
    labels = _df["label"].tolist()
    
    skf = StratifiedKFold(n_folds, random_state=seed, shuffle=shuffle)

    train_test_splits = []
    for train_idxs, test_idxs in skf.split(idxs, labels):
        # add syonym instances to train
        train_source_uuids = df.loc[train_idxs][col_name].tolist()
        train_source_idxs = df[df["uuid"].isin(train_source_uuids)].index.tolist()
        assert set(train_idxs).intersection(train_source_idxs) == set()
        train_idxs = train_idxs + train_source_idxs

        # remove synonym instances from evaluation and replace with source instances
        test_source_uuids = df.loc[test_idxs][col_name].tolist()
        test_source_idxs = df[df["uuid"].isin(test_source_uuids)].index.tolist()
        assert set(train_idxs).intersection(test_source_idxs) == set()
        test_idxs = test_source_idxs

        train_test_splits.append((train_idxs, test_idxs))

    return train_test_splits


def main(
    model_type: str = "transformer",
    train_path: str = "",
    val_path: str = "",
    test_path: str = "",
    additional_data_path: str = "",
    zero_shot_langs: str = "de",
    classification_model_name_or_path: str = "distilbert-base-uncased",
    additional_data_multiplier: int = 1,
    num_labels: int = 2,
    batch_size: int = 32,
    max_length: int = 64,
    cross_validation: bool = False,
    n_folds: int = 10,
    test_size: float = 0.1,
    text_col: str = "sentence",
    label_col: str = "is_variable",
    eval_attribute_cols: str = "lang,type_short,subtype_short",
    shuffle: bool = True,
    seeds: str = "42",
    langs: str = "en",
    epochs: int = 5,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    fp16: bool = False,
    output_dir: str = "",
    hyperopt: bool = False,
    hyperopt_trials: int = 2,
    metric_for_best_model: str = "BinaryF1Score()",
    save_checkpoints: bool = False,
    save_total_limit: int = 1,
    save_log_history: bool = True,
    load_best_model_at_end: bool = False,
    save_best_model: bool = False,
    drop_synonym_instances: bool = False,
    synonym_source_col: str = "source_uuid",
    load_in_8bit: bool = False,
    cluster_embedding_vectorizer: bool = True,
    ):
    seeds = seeds.split(",")
    langs = langs.split(",")
    zero_shot_langs = zero_shot_langs.split(",")
    eval_attribute_cols = eval_attribute_cols.split(",")

    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    if model_type in ["logistic_regression", "svm", "linear_svm"]:
        output_dir = os.path.join(output_dir, f"linear={model_type.replace('_', '')}-trained_"+timestamp)
    elif model_type in ["knn", "rnn", "kmeans"]:
        output_dir = os.path.join(output_dir, f"cluster={model_type.replace('_', '')}-trained_"+timestamp)
    else:
        output_dir = os.path.join(output_dir, f"{classification_model_name_or_path.replace('/', '--')}-finetuned_"+timestamp)
    os.makedirs(output_dir, exist_ok=True)

    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(classification_model_name_or_path, num_labels=num_labels, load_in_8bit=load_in_8bit)
        return model
    
    if train_path:
        df = pd.read_csv(train_path, sep="\t").rename(columns={text_col: "text", label_col: "label"})
        df = df[df["lang"].isin(langs)].reset_index()
        df = clean_df(df)
        zero_shot_df = df[df["lang"].isin(zero_shot_langs)].reset_index()
        zero_shot_df = clean_df(zero_shot_df)
    if val_path:
        val_df = pd.read_csv(val_path, sep="\t").rename(columns={text_col: "text", label_col: "label"})
        val_df = val_df[val_df["lang"].isin(langs)].reset_index()
        val_df = clean_df(val_df)
    if test_path:
        test_df = pd.read_csv(test_path, sep="\t").rename(columns={text_col: "text", label_col: "label"})
        test_df = clean_df(test_df)

        if output_dir:
            test_df.to_csv(os.path.join(output_dir, "test.tsv"), index=False, sep="\t")

    if cross_validation and n_folds > 1:
        df = pd.concat([df, val_df]).reset_index()

    seed_results = defaultdict(lambda: defaultdict(list))
    seed_configs = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        seed = int(seed)
        seed_everything(seed)

        tokenizer = AutoTokenizer.from_pretrained(classification_model_name_or_path, use_fast=True)
        if "pad_token" not in tokenizer.special_tokens_map:
            tokenizer.pad_token = tokenizer.eos_token

        args = TrainingArguments(
            os.path.join(output_dir, f"seed={seed}"),
            save_strategy="no",  # this will be overwritten later
            save_total_limit=save_total_limit,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            metric_for_best_model=metric_for_best_model,
            push_to_hub=False,
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            fp16=fp16,
        )

        if cross_validation and n_folds > 1:
            if drop_synonym_instances:
                train_test_splits = kfold_train_test_split_with_dropped_instances(df, synonym_source_col, seed, n_folds, shuffle) 
            else:
                skf = StratifiedKFold(n_folds, random_state=seed, shuffle=shuffle)
                
                idxs = df.index.tolist()
                labels = df["label"].tolist()
                train_test_splits = skf.split(idxs, labels)
        else:
            if drop_synonym_instances:
                train_idxs, test_idxs = train_test_split_with_dropped_instances(df, synonym_source_col, test_size*2, seed)                
            else:
                idxs = df.index.tolist()
                labels = df["label"].tolist()
                train_idxs, test_idxs, _, _ = train_test_split(idxs, labels, test_size=test_size*2, random_state=seed, stratify=labels)

            train_test_splits = [(train_idxs, test_idxs)]

        fold_results = defaultdict(list)
        fold_configs = {}
        for i, (train_idx, val_idx) in enumerate(train_test_splits):
            local_output_dir = os.path.join(args.output_dir, f"fold={i}")
            os.makedirs(local_output_dir, exist_ok=True)

            if cross_validation and n_folds > 1:
                val_data = df.loc[val_idx]
            else:
                train_idx = train_idx + val_idx
                val_data = val_df

            train_data = df.loc[train_idx]
            if additional_data_path:
                additional_df = pd.read_csv(additional_data_path, sep="\t").rename(columns={text_col: "text", label_col: "label"})
                additional_df = clean_df(additional_df)
                additional_df = additional_df[additional_df["lang"].isin(langs)].reset_index()
                additional_data_size = min(train_data.shape[0] * additional_data_multiplier, additional_df.shape[0])
                train_data = pd.concat([train_data, additional_df.sample(additional_data_size)])
                train_data = train_data.sample(frac=1).reset_index(drop=True)

            test_data = test_df

            print(train_data.shape[0], val_data.shape[0], test_data.shape[0])

            if model_type in ["logistic_regression", "svm", "linear_svm"]:
                trainer = LinearModelTrainer(
                    train_data,
                    val_data,
                    compute_metrics,
                    tokenizer=tokenizer,
                    model_type=model_type,
                    metric_for_best_model=metric_for_best_model,
                    n_warmup_steps=10,
                    seed=seed,
                    hyperopt=hyperopt,
                    hyperopt_trials=hyperopt_trials,
                )
                trainer.train()
            elif model_type in ["knn", "rnn", "kmeans"]:
                _encoder = None
                if cluster_embedding_vectorizer:
                    _encoder = SentenceTransformer(classification_model_name_or_path, device="cuda")

                trainer = ClusterModelTrainer(
                    train_data,
                    val_data,
                    compute_metrics,
                    tokenizer=tokenizer,
                    model_type=model_type,
                    metric_for_best_model=metric_for_best_model,
                    n_warmup_steps=10,
                    seed=seed,
                    hyperopt=hyperopt,
                    hyperopt_trials=hyperopt_trials,
                    encoder=_encoder,
                    batch_size=batch_size,
                )
                trainer.train()
            else:
                dataset = DatasetDict({
                    "train": Dataset.from_pandas(train_data), 
                    "val": Dataset.from_pandas(val_data), 
                    "test": Dataset.from_pandas(test_data), 
                    "zero_shot_test": Dataset.from_pandas(zero_shot_df)
                })
                encoded_dataset = dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length})

                trainer = Trainer(
                    model_init=model_init,
                    args=args,
                    train_dataset=encoded_dataset["train"],
                    eval_dataset=encoded_dataset["val"],
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                )

                if hyperopt:
                    best_trial = trainer.hyperparameter_search(direction="maximize", backend="optuna", hp_space=optuna_hp_space, n_trials=hyperopt_trials, compute_objective=optuna_objective)
                    for n, v in best_trial.hyperparameters.items():
                        setattr(trainer.args, n, v)

                # set save arguments after hyperparameter search
                trainer.args.save_strategy = "epoch" if save_checkpoints else "no"
                trainer.args.load_best_model_at_end = load_best_model_at_end
                trainer.args.save_total_limit = save_total_limit
                trainer.train()

                if save_log_history:
                    log_history_df = pd.DataFrame(trainer.state.log_history)
                    log_history_df.to_csv(os.path.join(local_output_dir, "trainer_log_history.tsv"), index=False, sep="\t")

            results = trainer.evaluate()
            if save_best_model:
                model_path = os.path.join(local_output_dir, "final_model")
                os.makedirs(model_path, exist_ok=True)
                trainer.save_model(model_path)
            
            for k,v in results.items():
                fold_results["val_"+k].append(v)
            
            if isinstance(trainer, LinearModelTrainer) or isinstance(trainer, ClusterModelTrainer):
                preds, labels, _ = trainer.predict(test_data)
                if zero_shot_df.shape[0] > 0:
                    zero_preds, zero_labels, _ = trainer.predict(zero_shot_df)
                else:
                    zero_preds, zero_labels = [], []
            else:
                preds, labels, _ = trainer.predict(encoded_dataset["test"])
                if zero_shot_df.shape[0] > 0:
                    zero_preds, zero_labels, _ = trainer.predict(encoded_dataset["zero_shot_test"])
                else:
                    zero_preds, zero_labels = [], []

            pred_scores = compute_metrics([preds, labels])
            for k,v in pred_scores.items():
                fold_results["test_"+str(k)].append(v)
            test_data["pred"] = np.argmax(preds, axis=1)
            softmax = torch.nn.Softmax(dim=1)
            pred_probs = softmax(torch.Tensor(preds))
            test_data["pred_scores"] = pred_probs.tolist()

            if eval_attribute_cols:
                fine_grained_pred_scores = compute_fine_grained_metrics(test_data, eval_attribute_cols, target_col="label", pred_col="pred_scores")

                for k,v in fine_grained_pred_scores.items():
                    for m,s in v.items():
                        fold_results["test_"+str(k)+":"+m].append(s)

            if isinstance(zero_preds, np.ndarray):
                zero_pred_scores = compute_metrics([zero_preds, zero_labels])
                for k,v in zero_pred_scores.items():
                    fold_results["zero_test_"+str(k)].append(v)

            os.makedirs(local_output_dir, exist_ok=True)

            config_dict = {}
            config_dict["run"] = {
                    "seed": seed,
                    "classification_model_name_or_path": classification_model_name_or_path,
                    }
            fold_configs[str(i)] = config_dict

            columns = ["uuid"] + [c for c in test_data.columns if "pred" in c]
            filename = "test_preds.tsv"
            for col in test_data.columns:
                if "_scores" in col:
                    test_data[col] = test_data[col].apply(lambda x: ";".join([str(s) for s in x]))
            test_data[columns].to_csv(os.path.join(local_output_dir, filename), index=False, sep="\t")

            if zero_preds:
                zero_shot_df["pred"] = np.argmax(zero_preds, axis=1)
                softmax = torch.nn.Softmax(dim=1)
                zero_pred_probs = softmax(torch.Tensor(zero_preds))
                zero_shot_df["pred_scores"] = zero_pred_probs.tolist()
                filename = "zero_test_preds.tsv"
                for col in zero_shot_df.columns:
                    if "_scores" in col:
                        zero_shot_df[col] = zero_shot_df[col].apply(lambda x: ";".join([str(s) for s in x]))
                zero_shot_df.to_csv(os.path.join(local_output_dir, filename), index=False, sep="\t")

        for k,v in fold_results.items():
            seed_results[k][str(seed)] = v
        
        for k,v in fold_configs.items():
            seed_configs[k][str(seed)] = v
            
    if output_dir:
        config_path = os.path.join(output_dir, "config.json")
        write_json(config_path, seed_configs)

        results_path = os.path.join(output_dir, "results.json")
        write_json(results_path, seed_results)
        print(f"Wrote results to file: {results_path}")


if __name__ == "__main__":
    typer.run(main)