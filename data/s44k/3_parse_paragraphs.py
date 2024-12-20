import os
import time
import orjson
import typer
import pysbd
import logging
from tqdm import tqdm
from pathlib import Path
from lingua import Language, LanguageDetectorBuilder


log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f'{log_dir / Path(__file__).stem}.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def clean_text(text):
    text = text.replace("¬ ", "")
    text = text.replace("  ", " ")
    text = text.replace("\\", "")
    text = text.replace("-</td></tr><tr><td>", "")  # PDF grobid-specific line-breaks
    text = text.replace("</td></tr><tr><td>", " ")  # inject space after line-breaks
    return text


def get_pysbd_lang(language):
    if language == Language.ENGLISH:
        return "en"
    elif language == Language.FRENCH:
        return "fr"
    elif language == Language.GERMAN:
        return "de"
    elif language == Language.ITALIAN:
        return "it"
    elif language == Language.RUSSIAN:
        return "ru"
    elif language == Language.SPANISH:
        return "es"
    else:
        return ""


def get_segmenter(text):
    languages = [Language.ENGLISH, Language.GERMAN, Language.ITALIAN, Language.RUSSIAN, Language.SPANISH, Language.FRENCH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    pysbd_segmenters = {"en": pysbd.Segmenter(language="en", clean=True), "fr": pysbd.Segmenter(language="fr", clean=True), "de": pysbd.Segmenter(language="de", clean=True), "it": pysbd.Segmenter(language="it", clean=True), "ru": pysbd.Segmenter(language="ru", clean=True), "es": pysbd.Segmenter(language="es", clean=True)}

    language = detector.detect_language_of(text)
    lang = get_pysbd_lang(language)
    if lang == "":
        return "", lang
    else:
        return pysbd_segmenters[lang], lang


def split_into_paragraphs(paper_json):
    paragraphs = {}

    for i,p in enumerate(paper_json["pdf_parse"]["body_text"]):
        i = str(i)
        text = p["text"]
        text = clean_text(text)
        if text != "":
            segmenter, lang = get_segmenter(text)
            if segmenter == "":
                # TODO: apply cleaning (segmenter also has a cleaner)
                paragraphs[i] = {"sentences": [text], "lang": lang}
            else:
                paragraphs[i] = {"sentences": segmenter.segment(text), "lang": lang}
    
    for j,(_,p) in enumerate(paper_json["pdf_parse"]["ref_entries"].items()):  # TODO: how to fit sentences (including wrongly recognized ones) from figures into the right context?
        j = "FIG"+str(len(paper_json["pdf_parse"]["body_text"])+j)
        for text in [p["content"], p["text"]]:
            text = clean_text(text)
            if text != "":
                segmenter, lang = get_segmenter(text)
                if segmenter == "":
                    paragraphs[j] = {"sentences": [text], "lang": lang}
                else:
                    paragraphs[j] = {"sentences": segmenter.segment(text), "lang": lang}

    return paragraphs


def parse(
    input_dir: str = "./2_jsons/",
    output_dir: str = "./3_jsons_paragraphs/",
    overwrite: bool = True,
    last_modified_days: int = None,
    ):
    assert os.path.isdir(input_dir)
    logging.info('Started.')
    files = [f for f in Path(input_dir).iterdir()]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped_files = 0
    failed_files = 0
    created_files = 0

    for f in tqdm(files):
        output_file = output_dir / f.name

        if not overwrite:  # skip if the file already exists
            if output_file.is_file():
                skipped_files += 1
                continue

        if last_modified_days:  # skip if the file has been modified within the past N days
            if output_file.is_file():
                if (time.time() - (86400*last_modified_days)) < output_file.stat().st_mtime:
                    skipped_files += 1
                    continue

        try:
            with open(f, "rb") as f:
                paper_json = orjson.loads(f.read())
        except Exception:
            logging.debug(f'Unable to read file: {f}')  # skip if the file cannot be opened
            failed_files += 1
            continue
        
        try:
            paragraphs = split_into_paragraphs(paper_json)

            with open(output_file, "wb") as f:
                f.write(orjson.dumps(paragraphs))
            created_files += 1
        except Exception:
            logging.debug(f'Failed generating paragraphs for file: {f}')  # skip if the file cannot be processed
            failed_files += 1

    logging.info(f'Skipped files: {skipped_files}')
    logging.info(f'Failed files: {failed_files}')
    logging.info(f'Created files: {created_files}')
    logging.info('Finished.')

if __name__ == '__main__':
    # parse()
    typer.run(parse)