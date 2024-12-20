import requests
import os
from collections import defaultdict
import json
import jsonlines
from tqdm import tqdm
import traceback
from lxml import etree
import typer
import logging
from pathlib import Path
from bs4 import BeautifulSoup


log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f'{log_dir / Path(__file__).stem}.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def load_json(path):
    assert ".json" == path[-5:]
    assert os.path.isfile(path)
    data = {}
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def load_jsonl(path):
    assert ".jsonl" == path[-6:]
    assert os.path.isfile(path)
    data = []
    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            data.append(obj)
    return data

def load_text_lines(path):
    assert ".txt" == path[-4:]
    assert os.path.isfile(path)
    with open(path, "r") as f:
        lines = f.read().split("\n")
    return lines


def get_metadata(urls):
    metadata = []
    variablesless_urls = []

    for url in tqdm(urls):
        data = {}
        data['url'] = url

        title = ''
        date = ''
        countries = []
        abstract = []
        methodology = []
        topics = []
        variables = []
        publications = []
        codebook_url = ''
        codebook_secured = True

        try:
            response = requests.get(url)
            content = json.loads(response.content)

            if content["hits"]["hits"] == []:
                continue

            hit = content["hits"]["hits"][0]

            if '_source' in hit:
                try:
                    # Extract text
                    if 'title' in hit['_source']:
                        title = hit['_source']['title']
                except Exception:
                    pass
                
                try:
                    # Extract date
                    if 'date_recency' in hit['_source']:
                        date = hit['_source']['date_recency']
                except Exception:
                    pass
                
                try:
                    # Extract countries_iso
                    if 'countries_iso' in hit['_source']:
                        countries = hit['_source']['countries_iso']
                except Exception:
                    pass
                
                try:
                    # Extract abstract_en
                    if 'abstract_en' in hit['_source']:
                        abstract = hit['_source']['abstract_en']
                except Exception:
                    pass
                
                try:
                    # Extract methodology_collection_en
                    if 'methodology_collection_en' in hit['_source']:
                        methodology = hit['_source']['methodology_collection_en']
                except Exception:
                    pass
                
                try:
                    # Extract topic_en
                    if 'topic_en' in hit['_source']:
                        topics = hit['_source']['topic_en']
                except Exception:
                    pass
                
                try:
                    # Extract related_variables
                    if 'related_variables' in hit['_source']:
                        variables = hit['_source']['related_variables']
                        if isinstance(variables, list):
                            variables = [v['id'] for v in variables]
                except Exception:
                    logging.info(f"Could not load related_variables for {url}")
                
                try:
                    # Extract related_publication
                    if 'related_publication' in hit['_source']:
                        publications = hit['_source']['related_publication']
                        if isinstance(publications, list):
                            publications = [v['id'] for v in publications]
                except Exception:
                    pass

                try:
                    # Extract codebook url
                    if "links_codebook" in hit["_source"]:
                        if "url" in hit["_source"]["links_codebook"][0]:
                            codebook_url = hit["_source"]["links_codebook"][0]["url"]
                        if "secured" in hit["_source"]["links_codebook"][0]:
                            codebook_secured = (
                                False
                                if hit["_source"]["links_codebook"][0][
                                    "secured"
                                ].lower()
                                == "false"
                                else True
                            )
                except Exception:
                    pass

        except Exception as e:
            logging.warn(e)

        data['title'] = title
        data['date_recency'] = date
        data['countries'] = countries
        data['abstract'] = abstract
        data['methodology'] = methodology
        data['topics'] = topics
        data['variables'] = variables
        if len(variables) == 0:
            logging.info("No variables for {url}")
            variablesless_urls.append(url)
        data['publications'] = publications
        data['codebook'] = codebook_url
        data['codebook_secured'] = codebook_secured
        metadata.append(data)

    logging.info(f"Found {len(variablesless_urls)} URLs with no variables out of {len(urls)} total URLs.")

    return metadata, variablesless_urls


def contains_dataset(research_data_meta, metadata):
    exists = False
    for nm in metadata:
        if research_data_meta["url"] in nm["url"]:
            exists = True
    return exists


def get_new_metadata(metadata, output_path=None, new_metadata_keys=None):
    variable_base_url = "http://search.gesis.org/searchengine?q=_id:"
    failed_urls = defaultdict(list)

    if not new_metadata_keys:
        new_metadata_keys = []

    for m in tqdm(metadata):
        if m["url"] in new_metadata_keys:
            logging.debug(f"This research dataset already contains an entiry: {m['url']}")
            continue

        _variables = []
        for v in tqdm(m["variables"]):
            _url = variable_base_url + v

            var_meta = {}

            r = requests.get(_url)
            if r.status_code == 200:
                content = json.loads(r.content)
                var_meta = content
            else:
                logging.debug(f"Failed for: {_url}")
                failed_urls[m["url"]].append(_url)
                continue

            _variables.append({v: var_meta})

        m["variables"] = _variables
        new_metadata_keys.append(m["url"])
        if len(_variables) == 0:
            logging.debug("0 variables found for URL:"+m["url"])
        
        if output_path:
            with jsonlines.open(output_path, "a") as writer:
                writer.write(m)

    new_metadata = load_jsonl(output_path)
    os.remove(output_path)

    return new_metadata


def get_categories(html):
    table = etree.HTML(html).find("body/table")
    rows = iter(table)
    categories = []
    for row in rows:
        values = [col.text for col in row]
        categories.append(values[1])

    return ";".join([c for c in categories if c and c.lower() != "wertelabel"])


def get_highlighted_item_in_table(html):
    soup = BeautifulSoup(html, 'html.parser')
    highlighted_rows = soup.find_all('td', class_='highlight_item_td')[1:]
    content = " ".join([cell.text.strip() for row in highlighted_rows for cell in row.find_all(['td', 'span'])])
    return content


def format_research_data_variables(metadata, output_path=None):
    if ".jsonl" not in output_path and ".json" in output_path:
        output_path = output_path.replace(".json", ".jsonl")

    output = {}

    for m in tqdm(metadata):
        _url = m["url"]
        year = m["date_recency"].split("-")[0]
        try:
            _variables_raw = m["variables"]
            _variables = {}
            if len(_variables_raw) > 0:
                for v in _variables_raw:
                    assert len(v) == 1
                    for var_id, v_data in v.items():
                        if v_data["hits"]["total"]["value"] == 0:
                            _variables[var_id] = {}
                            continue

                        assert v_data["hits"]["total"]["value"] == 1
                        assert len(v_data["hits"]["hits"]) == 1

                        v_meta_raw = v_data["hits"]["hits"][0]["_source"]

                        valid_keys = [
                            "title",
                            "totle_en",
                            "variable_label",
                            "variable_name",
                            "variable_text",
                            "question_text",
                            "question_text_en",
                            "question_id",
                            "question_label",
                            "question_lang",
                            "sub_question",
                            "item_categories",
                            "answer_categories",
                            "variable_code_list",
                            "topic",
                            "topic_en",
                            "question_type1",
                            "question_type2",
                            "analysis_unit",
                        ]

                        v_meta = {"year": year}
                        for k in valid_keys:
                            if k in v_meta_raw:
                                if k in ["item_categories", "answer_categories"]:
                                    v_meta[k[:-3]+"y"] = get_highlighted_item_in_table(v_meta_raw[k])
                                    v_meta[k] = get_categories(v_meta_raw[k])
                                else:
                                    v_meta[k] = v_meta_raw[k]

                        _variables[var_id] = v_meta

            if output_path and _variables:
                with jsonlines.open(output_path, "a") as writer:
                    writer.write({"url": _url, "variables": _variables})
        except Exception:
            traceback.print_stack()
            raise Exception(f"Error during processing of metadata for variable: {_url}")

    return output


def get_variable_metadata(ids, output_path):
    root = "https://search.gesis.org/searchengine?q=_id:"
    all_urls = []

    all_urls = [root+i for i in ids]
    all_urls = list(set(all_urls))
    logging.info(f"Number of unique research data URLs: {len(all_urls)}")
    
    new_metadata_path = output_path.split(".")[0]+"_raw.jsonl"

    logging.info("Generating all metadata...")
    metadata, _ = get_metadata(all_urls)
    logging.info("Loading new metadata...")
    new_metadata = get_new_metadata(metadata, new_metadata_path)
        
    variable_lengths = [len(m["variables"]) for m in new_metadata]
    logging.info(f"Total variables: {sum(variable_lengths)}")
    logging.info("Formatting metadata...")
    format_research_data_variables(new_metadata, output_path)


def main(
        survey_ids_path: str = "./data/gsim/survey_ids.txt",
        output_path: str = "./data/gsim/survey_items.json",
    ):
    output_path = os.path.abspath(output_path)

    survey_ids = load_text_lines(survey_ids_path)

    logging.info("Downloading survey item metadata...")
    get_variable_metadata(survey_ids, output_path=output_path)
    logging.info("Done.")


if __name__ == "__main__":
    typer.run(main)