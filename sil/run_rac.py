from data_utils import clean_df, preprocess_function, write_json
from metrics import compute_metrics, compute_fine_grained_metrics

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import skops.io as sio
import typer
from datetime import datetime
import time
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from lightning.fabric.utilities.seed import seed_everything
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


def load_transformer_model(model_name_or_path, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    return model, tokenizer


def load_knn_model(model_path):
    model = sio.load(model_path, trusted=True)
    return model


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


def load_vectorizer():
        params = {
                "lowercase": True,
                "max_df": 1.0,
                "min_df": 1,
                "sublinear_tf": False,
                "smooth_idf": True,
        }
        return TfidfVectorizer(**params)


def main(
    train_path: str = "./data/sild/diff_train.tsv",
    test_path: str = "./data/sild/diff_test.tsv",
    transformer_model_name_or_path: str = "",
    knn_model_path: str = "",
    knn_featurizer: str = "",
    seed: int = 42,
    text_col: str = "sentence",
    label_col: str = "is_variable",
    eval_attribute_cols: str = "lang,type_short,subtype_short",
    output_dir: str = "",
    num_labels: int = 2,
    batch_size: int = 8,
    alpha: float = 0.5,
    max_length: int = 64,
    ):
    seed_everything(seed)
    eval_attribute_cols = eval_attribute_cols.split(",")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    if knn_featurizer:
        output_dir = os.path.join(output_dir, f"{knn_featurizer.replace('/', '_')}_{timestamp}")
    else:
        output_dir = os.path.join(output_dir, f"norac-{timestamp}")
    
    train_df = pd.read_csv(train_path, sep="\t").rename(columns={text_col: "text", label_col: "label"})
    train_df = clean_df(train_df)
    test_df = pd.read_csv(test_path, sep="\t").rename(columns={text_col: "text", label_col: "label"})
    test_df = clean_df(test_df)
    y_true = test_df["label"].tolist()  # (n,)

    y_pred = []  # (n,2)
    if transformer_model_name_or_path:
        hf_model, hf_tokenizer = load_transformer_model(transformer_model_name_or_path, num_labels)
        dataset = DatasetDict({"test": Dataset.from_pandas(test_df)})
        encoded_dataset = dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": hf_tokenizer, "max_length": max_length}, remove_columns=dataset["test"].column_names)
        data_collator = DataCollatorWithPadding(hf_tokenizer, max_length=max_length, padding="max_length")
        dataloader = DataLoader(encoded_dataset['test'], batch_size=batch_size, collate_fn=data_collator)

        hf_model.eval()
        hf_model.to(device)
        for batch in tqdm(dataloader):
            with torch.no_grad():
                outputs = hf_model(**batch.to(device)).logits.detach().cpu().numpy()
                scores = softmax(outputs).tolist()
                y_pred.extend(scores)

        y_pred = np.asarray(y_pred)

        test_df["hf_pred"] = np.argmax(y_pred, axis=1)
        test_df["hf_pred_scores"] = y_pred.tolist()

    if knn_model_path and knn_featurizer:
        knn_model = load_knn_model(knn_model_path)

        if knn_featurizer == "tf-idf":
            vectorizer = load_vectorizer()
            vectorizer.fit(train_df["text"].tolist())
            X = vectorizer.transform(test_df["text"].tolist())
        else:
            encoder = SentenceTransformer(knn_featurizer, device="cuda")
            X = encoder.encode(test_df["text"].tolist(), batch_size=batch_size, show_progress_bar=False)
        
        predict_proba = getattr(knn_model, "predict_proba", None)
        if callable(predict_proba):
            _y_pred = torch.Tensor(knn_model.predict_proba(X))
        else:
            _y_pred = torch.Tensor(knn_model.predict(X))
            _y_pred = np.column_stack((1-_y_pred, _y_pred))

        _y_pred = np.asarray(_y_pred)

        if isinstance(y_pred, np.ndarray):
            if y_pred.shape == _y_pred.shape:
                print(y_pred.shape, _y_pred.shape)
                y_pred = (alpha * y_pred) + ((1 - alpha) * _y_pred)
                assert np.allclose(np.sum(y_pred, axis=1), 1.0, atol=1e-8), "Sum of probabilities should be 1."
            else:
                raise ValueError(f"Predictions of different shapes: {y_pred.shape} and {_y_pred.shape}")
        else:
            y_pred = _y_pred

        test_df["knn_pred"] = np.argmax(_y_pred, axis=1)
        test_df["knn_pred_scores"] = _y_pred.tolist()

    if isinstance(y_pred, np.ndarray):
        test_df["pred"] = np.argmax(y_pred, axis=1)
        test_df["pred_scores"] = y_pred.tolist()

        results = compute_metrics([y_pred, y_true])
        fine_grained_results = compute_fine_grained_metrics(test_df, eval_attribute_cols, target_col="label", pred_col="pred_scores")

        combined_results = defaultdict(list)
        for k,v in results.items():
            print(k,v)
            combined_results["test_"+str(k)].append(v)
        for k,v in fine_grained_results.items():
            for m,s in v.items():
                combined_results["test_"+str(k)+":"+m].append(s)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            test_pred_path = os.path.join(output_dir, "preds.tsv")
            test_df.to_csv(test_pred_path, index=False, sep="\t")
            
            results_path = os.path.join(output_dir, "results.json")
            write_json(results_path, combined_results)
            print(f"Wrote results to file: {results_path}")
    else:
        print("No predictions were computed.")


if __name__ == "__main__":
    typer.run(main)