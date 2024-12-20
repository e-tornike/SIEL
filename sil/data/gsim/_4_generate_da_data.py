import jsonlines
import uuid
from lingua import Language, LanguageDetectorBuilder
import random
import numpy as np
import json
from tqdm import tqdm
import os
import typer
import pandas as pd
import logging
from pathlib import Path


log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f'{log_dir / Path(__file__).stem}.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

seed = 42
random.seed(seed)
np.random.seed(seed)

def detect_language(text, detector):
    if text.replace(" ", ""):
        pred = detector.detect_language_of(text)
        if pred == Language.ENGLISH:
            return "en"
        elif pred == Language.GERMAN:
            return "de"
        else:
            return "other"
        
languages = [Language.ENGLISH, Language.GERMAN]
lang_detector = LanguageDetectorBuilder.from_languages(*languages).build()

data_type = "meta-pairs"

if data_type == "llm-gen":
    path = "/home/tornike/Coding/phd/sosci-simlearn/data/filtered_meta_llm-gen_20240113-210643_train.jsonl"  # llm-gen
    output_dir = "/home/tornike/Coding/phd/simple-experiments/data/synthetic_samples_llm-gen/"
    sample_sizes = [1000, 10000, 100000, 200000, 400000, 800000]
elif data_type == "meta-pairs":
    path = "/home/tornike/Coding/phd/sosci-simlearn/data/filtered_meta_pairs_20240113-122949_train.jsonl"  # meta pairs
    output_dir = "/home/tornike/Coding/phd/simple-experiments/data/synthetic_samples_meta-pairs/"
    sample_sizes = [1000, 10000, 100000, 200000, 400000]

data = []
with jsonlines.open(path, "r") as reader:
    for obj in reader:
        data.append(obj)

positive_rows = []
for d in tqdm(data):
    row = {}
    row["id"] = str(uuid.uuid4())
    if len(d["sentence_1"]) > len(d["sentence_2"]):
        row["sentence"] = d["sentence_1"]
    else:
        row["sentence"] = d["sentence_2"]
    row["is_variable"] = 1
    row["lang"] = detect_language(row["sentence"], lang_detector)
    row["variable"] = d["variable_id"]

    if row["lang"] in ["en", "de"]:
        positive_rows.append(row)

print(len(positive_rows))

def load_json(path):
    assert ".json" == path[-5:]
    data = {}
    with open(path, "r") as fp:
        data = json.load(fp)
    return data

def references_research_data(meta):
    for k, _ in meta.items():
        if "related_research" in k:
            return True
    return False

# mapping_path = "/home/tornike/Coding/phd/sosci-data-pipeline/meta_mapping.json"
mapping_path = "./data/gsim/metadata.json"
mapping = load_json(mapping_path)
ids = [pid for pid, meta in mapping.items() if not references_research_data(meta)]
print(f"Length of mapping: {len(mapping)}\nLength of PIDs: {len(ids)}")

root_dir = "./data/s44k/pdfs_paragraphs_json/"
files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

all_sentences = []

for f in tqdm(files):
    pid = os.path.basename(f).split(".")[0]
    if pid not in ids:
        paragraphs = load_json(f)

        sentences = []
        for i, d in paragraphs.items():
            sentences.extend(d["sentences"])

        all_sentences.extend([(pid, s) for s in sentences])

negative_rows = []
for source_id,s in tqdm(all_sentences):
    row = {}
    row["id"] = str(uuid.uuid4())
    row["sentence"] = s
    row["is_variable"] = 0
    row["lang"] = detect_language(s, lang_detector)
    row["variable"] = ""
    row["source_id"] = source_id

    if row["lang"] in ["en", "de"]:
        negative_rows.append(row)

print(len(negative_rows))

os.makedirs(output_dir, exist_ok=True)

for size in sample_sizes:
    positives = random.sample(positive_rows, int(size/2))
    negatives = random.sample(negative_rows, int(size/2))

    data = positives + negatives
    random.shuffle(data)

    size = str(size)
    size = size[:size.rfind("000")]+"k"
    output_path = os.path.join(output_dir, f"{size}.tsv")
    pd.DataFrame(data).to_csv(output_path, sep="\t", index=False)