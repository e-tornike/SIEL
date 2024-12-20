import json
import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from lingua import Language, LanguageDetectorBuilder
from bs4 import BeautifulSoup
import time
import datetime

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
        elif pred == Language.FRENCH:
            return "fr"
        elif pred == Language.ITALIAN:
            return "it"
        elif pred == Language.SPANISH:
            return "es"
        else:
            return "other"


def load_json(path):
    assert ".json" == path[-5:]
    data = {}
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def references_research_data(meta) -> bool:
    for k,_ in meta.items():
        if "related_research" in k:
            return True
    return False


def clean_variable_question(vq: str) -> str:
    vq = vq.strip() # remove leading and trailing whitespace

    for term in ['kalometer:', 'ZA-Studiennummer', 'Rangplatz', 'Erhebungstag', 'ONLY IN', '-----']:
        if term in vq:
            continue

    if '<br/>1.' in vq: # ignore everything after a break followed by a '1.' (which is typically a list of answers)
        vq = vq.split('<br/>1.')[0]
    if '...<br/>' in vq: # ignore everything after three dots followed by a break (which is typically a list of answers)
        vq = vq.split('...<br/>')[0]+'...'
    if '?<br/>' in vq:  # ignore everything after question mark followed by a break (which is typically a list of answers)
        vq = vq.split('?<br/>')[0]+'?'

    if ')<br/>' in vq: # ignore everything before the first paranthesis before a break (which is typically information for the interviewer)
        vq = ''.join(vq.split(')<br/>')[1:])
    
    if '<br/>(' in vq: # ignore everything after a paranthesis after a break (which is typically information for the interviewer)
        vq = vq.split('<br/>(')[0]

    for pattern in ['Frage:) <br/>', 'Frage:) ', 'Frage:', 'NENNT) ', 'CODE 1 IN ']:
        if pattern in vq:
            vq = ''.join(vq.split(pattern)[1:])

    for pattern in ['?  0.', '?  <FALLS']:
        if pattern in vq:
            vq = vq.split(pattern)[0]+'?'

    vq = vq.replace('-<br/>', '')
    vq = vq.replace('<br/>', ' ')

    if '.' in vq.split(' ')[0]: # ignore first word if it contains a dot (which is typically a question number)
        vq = ' '.join(vq.split(' ')[1:])

    if '<Vollständiger Fragetext' in vq: # ignfore everything after '<Vollständiger Fragetext' (which is typically a list of answers)
        vq = vq.split('<Vollständiger Fragetext')[0]

    if '<VOLLSTÄNDIGER FRAGENTEXT' in vq: # ignfore everything after '<VOLLSTÄNDIGER FRAGENTEXT' (which is typically a list of answers)
        vq = vq.split('<VOLLSTÄNDIGER FRAGENTEXT')[0]
    
    if vq and vq[-1] == ']':
        vq = vq[:-1]

    vq = vq.replace('  ', ' ')
    vq = vq.replace('---', '')

    return vq


def get_unique_variables(questions, names):
    uquestions = []
    unames = []

    for q,n in zip(questions, names):
        if q not in uquestions and n not in unames:
            uquestions.append(q)
            unames.append(n)
    
    return uquestions, unames


def get_variables(metadata_path, exclude_ids=[], clean_question=True, clean_html=True, lang="de"):
    metadata = load_json(metadata_path)
    print(f"Variable metadata size before excluding IDs: {len(metadata)}")
    print(f"Size of excluding IDs: {len(exclude_ids)}")

    variable_questions = []
    variable_names = []
    excluded_questions = []

    c = 0
    t = 0

    question_str = "question_text_en" if lang == "en" else "question_text"

    for mid,variables_dict in tqdm(metadata.items()):
        exclude = False
        if mid in exclude_ids:
            exclude = True
        for vn,meta in variables_dict.items():
            t += 1
            if question_str in meta:
                c += 1
                vq = meta[question_str]
                if clean_question:
                    vq = clean_variable_question(vq)
                if len(vq.replace(" ", "")) > 0 and len(vq.split()) > 2:
                    if clean_html:
                        vq = BeautifulSoup(vq, "lxml").text
                    if exclude:
                        excluded_questions.append(vq)
                    else:
                        variable_questions.append(vq)
                        variable_names.append(vn)
            
    print(f"Variables with questions {c}/{t}")

    return variable_questions, variable_names, excluded_questions
    

languages = [Language.ENGLISH, Language.GERMAN]
lang_detector = LanguageDetectorBuilder.from_languages(*languages).build()

valid_languages = sorted(["de", "en"])

mapping_path = "./meta_mapping.json"
mapping = load_json(mapping_path)
ids = [pid for pid,meta in mapping.items() if not references_research_data(meta)]
print(f"Length of mapping: {len(mapping)}\nLength of PIDs: {len(ids)}")

root_dir = "./ssoar_sosci_pdfs/pdfs_paragraphs_json/"
files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

all_sentences = []
c = 0
cn = 0
for f in tqdm(files):
    pid = os.path.basename(f).split(".")[0]
    if pid not in ids:  # this ensures that only sentences are included from publications that don't reference a research dataset
        paragraphs = load_json(f)

        sentences = []
        for i,d in paragraphs.items():
            sentences.extend(d["sentences"])

        all_sentences.extend([(pid,s) for s in sentences])
        cn += 1
    else:
        c += 1

print(f"Documents that reference research data: {c}")
print(f"Documents that don't reference research data: {cn}")

print(f"Length of all sentences: {len(all_sentences)}")

# Exclude these IDs which are used during training
_df = pd.read_csv("./data/sild/vadis-prolific-3_project_2023-12-09_1251_12:53:08.tsv", sep='\t')
exclude_ids = _df["research_data"].dropna().apply(lambda x: x.split(";")).tolist()
exclude_ids = [rd for rds in exclude_ids for rd in rds]
exclude_ids = sorted(list(set(exclude_ids)))
exclude_ids = [f"https://search.gesis.org/searchengine?q=_id:{e}" for e in exclude_ids]
print(f"Excluding {len(exclude_ids)} IDs.")

variables_path = "./variables_metadata.json"
all_questions, all_names, excluded_questions = [], [], []
for lang in valid_languages:
    _questions, _names, _excluded_questions = get_variables(variables_path, exclude_ids, clean_question=True, clean_html=True, lang=lang)
    all_questions.extend(_questions)
    all_names.extend(_names)
    excluded_questions.extend(_excluded_questions)
assert len(all_questions) == len(all_names)
print(f"Length of all variable questions: {len(all_questions)}")

all_questions, all_names = get_unique_variables(all_questions, all_names)
print(f"Length of unique questions: {len(all_questions)}")

finish = False
prev_sample_size = 0

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')

for init_sample_size in [50,100,200,400,800,1600,3200,6400,12800,25600,51200,102400]:
    # Set seed to get a different shuffle order of sentences
    random.seed(init_sample_size+13)
    np.random.seed(init_sample_size+13)

    print(f"Working on sample size: {init_sample_size}")
    sentences_size = min(len(all_sentences), init_sample_size)
    print(f"Sentences size: {sentences_size}")

    variables_size = min(len(all_questions), init_sample_size)
    print(f"Variables size: {variables_size}")

    sample_size = min(sentences_size, variables_size)
    if sample_size <= prev_sample_size:
        break

    print(f"Real sample size: {sample_size}")

    prev_top10 = all_sentences[:10]
    random.shuffle(all_sentences)
    assert prev_top10 != all_sentences[:10]

    random_variable_idxs = list(range(len(all_questions)))
    prev_top10 = random_variable_idxs[:10]
    random.shuffle(random_variable_idxs)
    assert prev_top10 != random_variable_idxs[:10]

    random_questions,random_names = zip(*[(all_questions[i],all_names[i]) for i in random_variable_idxs])
    assert len(random_questions) == len(random_names)

    false_data = []
    c = 0
    raw_sentences = []
    for pid,s in all_sentences:
        if len(false_data) >= sample_size:
            break
        s_lang = detect_language(s, lang_detector)  # do this again for each sentence
        if s_lang in valid_languages and s not in raw_sentences:
            false_data.append({"id": pid, "sentence": s, "is_variable": 0, "lang": s_lang, "variable": ""})
            raw_sentences.append(s)

    print(f"False data size: {len(false_data)}")
    
    true_data = []
    raw_questions = []
    for vq,vn in zip(random_questions,random_names):
        if len(true_data) >= sample_size:
            break
        vq_lang = detect_language(vq, lang_detector)
        if vq_lang in valid_languages and vq not in raw_questions and vq not in excluded_questions:
            true_data.append({"id": vn, "sentence": vq, "is_variable": 1, "lang": vq_lang, "variable": vn})
            raw_questions.append(vq)

    print(f"True data size: {len(true_data)}")

    if len(false_data) != len(true_data):
        sample_size = min(len(false_data), len(true_data))
        false_data = random.sample(false_data, k=sample_size)
        true_data = random.sample(true_data, k=sample_size)
    
    output_dir = f"./sosci-data-pipeline/data/{timestamp}/{'_'.join(valid_languages)}/{sample_size*2}"
    os.makedirs(output_dir, exist_ok=True)

    joined_df = pd.DataFrame(false_data+true_data)
    joined_df = joined_df.sample(frac=1).reset_index(drop=True)
    output_path = os.path.join(output_dir, f"joined_sentences_{sample_size*2}.tsv")
    joined_df.to_csv(output_path, index=False, sep="\t")

    if joined_df.shape[0] != joined_df.drop_duplicates(subset=["sentence"]).shape[0]:
        print(f"File contains duplicate sentences: {output_path}")

    prev_sample_size = sample_size

    if sample_size < init_sample_size:
        break