import typer
import json
import jsonlines
import itertools
import os
import datetime
import time
from tqdm import tqdm
import re
import openai
import random


def load_json(path):
    assert os.path.isfile(path)
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_jsonl(path):
    assert os.path.isfile(path)
    with jsonlines.open(path, "r") as reader:
        data = []
        for obj in reader:
            data.append(obj)
    return data


def write_jsonl(data, path):
    with jsonlines.open(path, "a") as writer:
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


def generate_text(prompt, model, max_tokens=64, temperature=0.8):
    completion = openai.chat.completions.create(
        model=model, 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return completion.choices[0].message.content


def batch_generate_text(prompts, model, max_tokens=64, temperature=0.8):
    messages = []
    for p in prompts:
        messages.append({"role": "user", "content": p})

    completion = openai.chat.completions.create(
        model=model, 
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return completion


def main(
    data_path: str = "./data/gsim/survey_items.jsonl",
    synthetic_data_path: str = "./data/llm-gen/pairs.json",
    meta_keys: str = "variable_label,question_text,item_category,topic",
    ignore_meta_key_pairs: str = "sub_question:item_category,question_text:item_category",
    output_dir: str = "./data/llm-gen/",
    temperatures: str = "0.8",
    model: str = "vicuna-7b-v1.5",
    max_tokens: int = 128,
    prompt: str = "Write a sentence in [LANG] describing that the following survey item was used in the paper by paraphrasing the survey item.\n\n",
    base_url: str = "http://localhost:8000/v1/",
    generated_data_path: str = "",
    ):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "llm_generated_pairs.jsonl")

    openai.api_key = "EMPTY"
    openai.base_url = base_url

    meta_keys = meta_keys.split(",")
    ignore_meta_key_pairs = [k.split(":") for k in ignore_meta_key_pairs.split(",")]
    temperatures = temperatures.split(",")

    data = load_jsonl(data_path)
    random.shuffle(data)

    gen_ids = []
    if generated_data_path:
        gen_data = load_jsonl(generated_data_path)
        gen_ids = [d["variable_id"] for d in gen_data]
        output_path = generated_data_path

    if synthetic_data_path:
        synthetic_data = load_json(synthetic_data_path)

    count = 0
    for i,d in enumerate(tqdm(data)):
        variables = d.get("variables", {})
        rd_meta_pairs = []
        for variable_id, variable_meta in tqdm(variables.items()):
            if "version" in variable_id.lower():
                continue
                
            try:
                meta_id = variable_id.replace("exploredata-", "")
                if meta_id in gen_ids:
                    continue
                
                if "de" in variable_meta.get("question_lang", ""):
                    lang = "German"
                else:
                    lang = "English"

                prompt = prompt.replace("[LANG]", lang)

                meta_pairs = []
                gen_texts = []

                for temp in temperatures:
                    text = ""
                    for k in meta_keys:
                        content = variable_meta.get(k, "")
                        if isinstance(content, list):
                            content = " ".join(content)

                        if content:
                            text += content + " "
                    if text == "" or text.replace(" ", "") == "" or len(text.split(" ")) < 3:
                        continue

                    if synthetic_data_path:
                        synthetic_content = synthetic_data[meta_id]
                        sentences = [synthetic_content, text]
                        random.shuffle(sentences)
                        sent_1, sent_2 = sentences
                        pair = {"source": meta_id.split("_")[0], "variable_id": meta_id, "sentence_1": sent_1, "sentence_2": sent_2, "temperature": temp}
                    else:
                        generated_content = generate_text(prompt+text, model, max_tokens, temp)
                        if generated_content in gen_texts:
                            continue
                        else:
                            gen_texts.append(generated_content)
                        pair = {"source": meta_id.split("_")[0], "variable_id": meta_id, "sentence_1": text, "sentence_2": generated_content, "temperature": temp}

                    meta_pairs.append(pair)
                    count += 1

                rd_meta_pairs.extend(meta_pairs)
            except Exception:
                print(f"Failed to generated data for variable: {variable_id}")

        if output_dir:
            print("Saved to path:", output_path)
            print(f"Number of pairs: {count}")
            write_jsonl(rd_meta_pairs, output_path)

if __name__ == "__main__":
    typer.run(main)