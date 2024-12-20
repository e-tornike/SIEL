import typer
import jsonlines
import itertools
import os
import datetime
import time
from tqdm import tqdm
import re


def load_jsonl(path):
    assert os.path.isfile(path)

    with jsonlines.open(path, "r") as reader:
        data = []
        for obj in reader:
            data.append(obj)
    return data


def write_jsonl(data, path):
    with jsonlines.open(path, "w") as writer:
        writer.write_all(data)


def clean_str(input_string):
    # Remove HTML tags
    no_html = re.sub(r'<.*?>', '', input_string)
    return no_html


def make_variable_meta_pairs(meta_id, meta, keys, ignore_pairs):
    pairs = []
    pairs_dict = []

    key_pairs = itertools.combinations(keys, 2)

    for k1,k2 in key_pairs:
        if [k1,k2] not in ignore_pairs and [k2,k1] not in ignore_pairs:
            k1_content = meta.get(k1, "")
            k2_content = meta.get(k2, "")

            if isinstance(k1_content, list) and k1_content:
                k1_content = " ".join(k1_content)
            if isinstance(k2_content, list) and k2_content:
                k2_content = " ".join(k2_content)
            
            if k1_content:
                k1_content = clean_str(k1_content)
            if k2_content:
                k2_content = clean_str(k2_content)

            if k1_content and k2_content and k1_content != k2_content:
                if len(k1_content.split(" ")) < 3 or len(k2_content.split(" ")) < 3:
                    continue
                
                pairs_dict.append({"source": meta_id.split("_")[0], "variable_id": meta_id, "sentence_1": k1_content, "sentence_2": k2_content})
                pairs.append((k1, k2))
    return pairs_dict    


def main(
    data_path: str = "/home/tornike/Coding/phd/inception-pre-annotation/sosci-data-pipeline/meta_08-12-23_20-40-00.jsonl",
    # meta_keys: str = "variable_label,question_text,question_text_en,sub_question,item_category,topic,topic_en",
    meta_keys: str = "variable_label,question_text,item_category,topic",
    ignore_meta_key_pairs: str = "sub_question:item_category,question_text:item_category",
    output_dir: str = "/home/tornike/Coding/phd/sosci-simlearn/data",
    ):
    meta_keys = meta_keys.split(",")
    ignore_meta_key_pairs = [k.split(":") for k in ignore_meta_key_pairs.split(",")]

    data = load_jsonl(data_path)

    full_meta_pairs = []
    for d in tqdm(data):
        variables = d.get("variables", {})
        for variable_id, variable_meta in variables.items():
            if "version" in variable_id.lower():
                continue
            meta_pairs = make_variable_meta_pairs(variable_id.replace("exploredata-", ""), variable_meta, meta_keys, ignore_meta_key_pairs)
            full_meta_pairs.extend(meta_pairs)
            # for p in meta_pairs:
            #     print(p)
            # break
        # break

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_dir, "meta_pairs_"+timestamp+".jsonl")
        write_jsonl(full_meta_pairs, output_path)

if __name__ == "__main__":
    typer.run(main)