from pydantic import BaseModel
from enum import Enum
import outlines
import pandas as pd
from tqdm import tqdm
import random
from torchmetrics import F1Score, Precision, Recall
import torch
import numpy as np
import typer
from lightning.fabric.utilities.seed import seed_everything
from collections import defaultdict
import os
import json
from datetime import datetime
import time


class Classes(int, Enum):
    _0 = 0
    _1 = 1


class Sentence(BaseModel):
    label: Classes


def write_json(path, data):
    with open(path, "w") as outfile:
        json.dump(data, outfile)


def compute_metrics(pred, argmax=False):
    f1 = F1Score(task="binary")
    r = Recall(task="binary")
    p = Precision(task="binary")
    f1micro = F1Score(average="micro", task="multiclass", num_classes=2)
    rmicro = Recall(average="micro", task="multiclass", num_classes=2)
    pmicro = Precision(average="micro", task="multiclass", num_classes=2)

    metrics = [f1, f1micro, r, rmicro, p, pmicro]

    if argmax:
        preds = np.argmax(preds, axis=1)

    scores = {}
    for m in metrics:
        preds, labels = pred
        score = m(preds=torch.Tensor(preds), target=torch.Tensor(labels))
        scores[str(m)] = float(score)
    return scores


def compute_fine_grained_metrics(df, attribute_columns, target_col, pred_col):
    results = {}

    for acol in attribute_columns:
        for name,group in df.groupby(by=acol):
            group_preds = torch.tensor(group[pred_col].tolist())
            group_labels = torch.tensor(group[target_col].tolist())
            pred_scores = compute_metrics([group_preds, group_labels])
            results[acol+":"+name] = pred_scores

    return results


def load_model(
        model_name_or_path,
        device,
        load_in_4bit=False,
        load_in_8bit=False,
        use_flash_attention_2=False,
    ):
        model_kwargs = {"load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit, "use_flash_attention_2": use_flash_attention_2}
        return outlines.models.transformers(model_name_or_path, device=device, model_kwargs=model_kwargs)


def predict(
        model,
        train_df,
        test_df,
        text_col,
        label_col,
        in_context_examples = 10,
        rng = None,
        max_tokens = 500,
        debug = False,
    ):
    prompt = "<s>[INST]"

    if in_context_examples > 0:
        prompt = "Background: Here are examples of sentences and their associated binary class labels (either 0 or 1, where 1 means that a survey item is mentioned in the sentence):\n"
    
        samples = []
        for label in [0, 1]:
            sample_df = train_df[train_df[label_col] == label].sample(int(in_context_examples/2))
            for i in range(sample_df.shape[0]):
                samples.append((sample_df.iloc[i][text_col], label))
    
        random.shuffle(samples)
        for sentence,label in samples:
            prompt += f"{sentence} (label: {label})\n"

    prompt += "Instruction: Classify the following sentence into the binary class 0 or 1, where 1 means that a survey item is mentioned in the sentence: "

    preds = {}
    failed_ids = []
    for i in tqdm(range(test_df.shape[0])):
        sid = test_df.iloc[i]["uuid"]
        sentence = test_df.iloc[i][text_col]

        try:
            result = outlines.generate.json(model, Sentence)(prompt+sentence+"[/INST] (label: ", max_tokens=max_tokens, rng=rng)
            
            preds[sid] = int(result.label)
        except:
            failed_ids.append(sid)
        
        if debug:
            break

    return preds, failed_ids


def load_data(path, label_col):
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=label_col)
    df[label_col] = df[label_col].apply(lambda x: int(x))
    return df


def main(
        model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
        train_path: str = "",
        test_path: str = "",
        in_context_examples: int = 10,
        text_col: str = "sentence",
        label_col: str = "is_variable",
        seeds: str = "42",
        max_tokens: int = 500,
        output_dir: str = "",
        eval_attribute_cols: str = "lang,type_short,subtype_short",
        debug: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_flash_attention_2: bool = False,
    ):
    if debug:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = seeds.split(",")
    eval_attribute_cols = eval_attribute_cols.split(",")

    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir, f"{model_name_or_path.replace('/', '--')}_"+timestamp)
    os.makedirs(output_dir, exist_ok=True)

    train_df = load_data(train_path, label_col).rename(columns={text_col: "text", label_col: "label"})
    test_df = load_data(test_path, label_col).rename(columns={text_col: "text", label_col: "label"})
    test_df.index = test_df["uuid"]

    model = load_model(model_name_or_path, device, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, use_flash_attention_2=use_flash_attention_2)

    seed_results = defaultdict(lambda: defaultdict(list))
    for seed in seeds:
        seed = int(seed)
        seed_everything(seed)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        preds_dict, _failed_ids = predict(model, train_df, test_df, "text", "label", in_context_examples, rng=rng, max_tokens=max_tokens, debug=debug)

        print(preds_dict)

        test_df["pred"] = pd.Series(preds_dict)
        final_test_df = test_df.dropna(subset=["label", "pred"])
        final_test_df["pred"] = final_test_df["pred"].astype(int)

        preds = final_test_df["pred"].tolist()
        labels = final_test_df["label"].tolist()
        pred_scores = compute_metrics([preds, labels])

        for k,v in pred_scores.items():
            seed_results["test_"+str(k)][str(seed)].append(v)

        if eval_attribute_cols:
            fine_grained_pred_scores = compute_fine_grained_metrics(test_df, eval_attribute_cols, target_col="label", pred_col="pred")

            for k,v in fine_grained_pred_scores.items():
                for m,s in v.items():
                    seed_results["test_"+str(k)+":"+m][str(seed)].append(s)

    if output_dir:
        results_path = os.path.join(output_dir, "results.json")
        write_json(results_path, seed_results)
        print(f"Wrote results to file: {results_path}")


if __name__ == "__main__":
    typer.run(main)
