import numpy as np
import json


def clean_df(df):
    for col, typ in df.dtypes.items():
        if typ == np.float64:
            df[col] = df[col].fillna(0)
            df[col] = df[col].astype(float)

        elif typ == np.int64:
            df[col] = df[col].fillna(0)
            df[col] = df[col].astype(int)

        df[col] = df[col].fillna("")
    if "label" in df:
        df["label"] = df["label"].astype(int)
    return df


def preprocess_function(examples, tokenizer, max_length=64):
    return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")


def write_json(path, data):
    with open(path, "w") as outfile:
        json.dump(data, outfile)