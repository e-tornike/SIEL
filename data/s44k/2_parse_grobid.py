import os
import json
from doc2json.grobid2json import tei_to_json
import logging
from tqdm import tqdm
from pathlib import Path


log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f'{log_dir / Path(__file__).stem}.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

root = "./1_xmls/"
assert os.path.exists(root)

files = [os.path.join(root, f) for f in os.listdir(root)]

output_dir = "./2_jsons/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

failed_files = []

for f in tqdm(files):
    output_path = os.path.join(output_dir, os.path.basename(f).split(".")[0]+".json")
    if os.path.exists(output_path):
        continue

    try:
        paper = tei_to_json.convert_tei_xml_file_to_s2orc_json(f)
    except Exception:
        failed_files.append(f)
        continue

    res_json = paper.release_json()

    with open(output_path, "w") as fp:
        json.dump(res_json, fp)

logging.info(f"Succeeded/failed: {len(files)-len(failed_files)}/{len(files)}")