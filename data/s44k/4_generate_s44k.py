import json
import os
import random

import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path


log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f'{log_dir / Path(__file__).stem}.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

seed = 42
random.seed(seed)
np.random.seed(seed)


def load_json(path):
    assert ".json" == path[-5:]
    data = {}
    with open(path, "r") as fp:
        data = json.load(fp)
    return data

root_dir = "./3_jsons_paragraphs/"
files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

sild_paths = ["../sild/diff_train.tsv", "../sild/diff_val.tsv", "../sild/diff_test.tsv"]
sild_df = pd.concat([pd.read_csv(p, sep="\t") for p in sild_paths])
sild_document_ids = [str(i) for i in sild_df["doc_id"].unique().tolist()]

all_sentences = []
for f in tqdm(files):
    pid = os.path.basename(f).split(".")[0]
    if (
        pid not in sild_document_ids
    ):  # ensure that only sentences are included from publications that are not in the SILD dataset
        paragraphs = load_json(f)

        sentences = []
        for i, d in paragraphs.items():
            sentences.extend(d["sentences"])

        all_sentences.extend([(pid, s) for s in sentences])

train_sample_size = 0.9
all_sentences_ids = range(len(all_sentences))
train_ids = random.sample(round(train_sample_size*len(all_sentences)))
val_ids = [i for i in all_sentences_ids if i not in train_ids]
train_sentences = [all_sentences[i] for i in train_ids]
val_sentences = [all_sentences[i] for i in val_ids]

with open("./texts_train.tsv", "w") as f:
    for s in train_sentences:
        f.write(s+"\n")

with open("./texts_val.tsv", "w") as f:
    for s in val_sentences:
        f.write(s+"\n")