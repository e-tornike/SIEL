import os
from bs4 import BeautifulSoup
import requests
import logging
from tqdm import tqdm
from pathlib import Path


log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f'{log_dir / Path(__file__).stem}.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


input_file = "./document_ids.txt"
assert os.path.exists(input_file)

output_dir = "./0_pdfs/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

downloaded_urls = []
downloaded_urls_path = "./downloaded_ids.txt"
if os.path.exists(downloaded_urls_path):
    with open(downloaded_urls_path, "r") as infile:
        downloaded_urls = list(set(infile.read().split("\n")))

with open(input_file, "r") as infile:
    ids = infile.read().split("\n")

urls = ["https://www.ssoar.info/ssoar/handle/document/"+i for i in ids]

successful_urls = []
failed_urls = []
for url in tqdm(urls):
    outfile_path = os.path.join(output_dir, os.path.basename(url)+".pdf")
    if downloaded_urls:
        if url in downloaded_urls:
            continue  # skip if already in downloaded_urls
    else:
        if os.path.exists(outfile_path):
            continue  # skip if already downloaded
    
    try:
        res_page = requests.get(url)
    except requests.exceptions.RequestException:
        logging.debug(f"Failed to open url: {url}")
        failed_urls.append(url)

    if res_page.status_code == 200 and res_page.content:
        try:
            raw_html = res_page.content
            soup = BeautifulSoup(raw_html, features="html.parser")
            pdf_item = soup.find("div", {"class": "item-page-field-wrapper table word-break"})
            pdf_url = pdf_item.find("a").get("href")

            res = requests.get(pdf_url)
            
            with open(outfile_path, 'wb') as outfile:
                outfile.write(res.content)
            successful_urls.append(url)
        except Exception:
            logging.debug(f"Failed parsing for url: {url}")
            failed_urls.append(url)
    else:
        failed_urls.append(url)

with open(downloaded_urls_path, "a") as outfile:
    for url in successful_urls:
        outfile.write(url+"\n")

with open("./failed_urls.txt", "w") as outfile:
    for url in failed_urls:
        outfile.write(url+"\n")

logging.info(f"Succeeded at downloading {len(successful_urls)} files. Failed at downloading {len(failed_urls)} files.")
logging.info(f"Files were saved to: {output_dir}")