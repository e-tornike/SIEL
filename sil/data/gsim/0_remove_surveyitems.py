from tqdm import tqdm
import os
import json
import jsonlines

def load_jsonl(path):
    assert os.path.isfile(path)

    with jsonlines.open(path, "r") as reader:
        data = []
        for obj in reader:
            data.append(obj)
    return data

sp_train_path = "/home/tornike/Coding/phd/SIL/data/sp/filtered_meta_llm-gen_20240113-210643_train.jsonl"
sp_val_path = "/home/tornike/Coding/phd/SIL/data/sp/filtered_meta_llm-gen_20240113-210643_val.jsonl"
sp_data = load_jsonl(sp_train_path) + load_jsonl(sp_val_path)
sp_data = {d["variable_id"]: d for d in sp_data}

data_path: str = "/home/tornike/Coding/phd/inception-pre-annotation/sosci-data-pipeline/filtered_meta_08-12-23_20-40-00.jsonl"
data = load_jsonl(data_path)

meta_keys: str = "variable_label,question_text,item_category,topic"
meta_keys = meta_keys.split(",")

output_sp_data = {}  # {variable_id: "synthetic sentence"}

for i,d in enumerate(tqdm(data)):
    variables = d.get("variables", {})
    for variable_id, variable_meta in tqdm(variables.items()):
        if "version" in variable_id.lower():
            continue
        meta_id = variable_id.replace("exploredata-", "")
        if meta_id in sp_data:
            try:
                text = ""
                for k in meta_keys:
                    content = variable_meta.get(k, "")
                    if isinstance(content, list):
                        content = " ".join(content)

                    if content:
                        text += content + " "
                # if text == "" or text.replace(" ", "") == "" or len(text.split(" ")) < 3:
                #     continue

                spd = sp_data[meta_id]
                s1 = spd["sentence_1"]
                s2 = spd["sentence_2"]

                if s1 == text:
                    s = s2
                elif s2 == text:
                    s = s1
                else:
                    print(f"Found no match for {meta_id} with text: '{text}'")
                    continue
                
                output_sp_data[meta_id] = s

            except Exception:
                print(f"Failed for variable with id: {variable_id}")

output_path = "./data/sp/synthetic_sentences.json"
with open(output_path, "w") as f:
    json.dump(output_sp_data, f)