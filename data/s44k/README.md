# SSOAR 44k (S44k) Dataset

## Dataset Description

### Dataset Summary

This dataset contains full texts from 44,741 [SSOAR]() publications. The texts can be used to pretrain language models (e.g., [SSOAR-XLM-R-base]()). The majority of the publications are in English and German.

### Supported Tasks

This dataset supports [masked language modeling](https://huggingface.co/tasks/fill-mask) and [text generation](https://huggingface.co/tasks/text-generation) tasks.

### Languages

The dataset covers English, German, ..., publications, written in scientific language.

## Dataset Structure and Splits

The dataset is a line-separated text file with each line containing a sentence from a publication. The dataset is divided into training and validation data, where the former contains X lines and the latter X lines.

## Dataset Creation

### Curation Rationale

Research in the social sciences has a long history of using surveys to understand societies by collecting information from individuals (e.g., to measure the influence of environmental factors on life satisfaction). Latent concepts, which do not appear in surveys explicitly, are first defined and then operationalized using one or more questions from surveys, which are a collection of statements or questions, called *survey items*. Due to a lack of standardization in citation practices, unique identifiers are not provided when describing the survey items used in a study. Without such identifiers, finding relevant work based on survey items of interest, which is desired by social scientists, is challenging. Consequently, there is a need for identifying used survey items to improve access to research along the FAIR principles (findable, accessible, interoperable, re-usable).

### Source Data

The data originates from 100 openly accessible social science publications from [SSOAR](https://www.gesis.org/ssoar) that are also indexed in [GESIS Search](https://search.gesis.org/).

#### Initial Data Collection and Normalization

Publications were first downloaded from SSOAR and the full text was extracted using [GROBID](https://github.com/kermitt2/grobid). Documents that could not properly be parsed were filtered out. To avoid data contamination, we additionally removed the 100 documents in [SILD](../sild/README.md). The document IDs of the resulting publications are listed in [document_ids.txt](/document_ids.txt). The IDs can be used to re-create the dataset using the pipeline script provided.

#### Who are the source language producers?

The publications in the dataset are written by social scientists and collected in the [SSOAR](https://www.gesis.org/ssoar) database.

#### Preparing the Data

To prepare the data, install [Docker](). The data can then be created using the following commands:

Install grobid (here via Docker) and then run the server:
```bash
./scripts/setup_grobid.sh
```

Download PDF files from SSOAR usign a list of IDs ([document_ids.txt](/document_ids.txt)):
```bash
python 1_download_pdfs.py
```

Process the PDF files (make sure that the grobid server is running):
```bash
./scripts/extract_texts.sh
```

Parse the grobid output XML files:
```bash
python 2_parse_grobid.py
```

Parse paragraphs from the XML files:
```bash
python 3_parse_paragraphs.py
```

Generate the S44k dataset
```bash
python 4_generate_s44k.py
```

## Considerations for Using the Data

### Social Impact of Dataset

The dataset can be used to pretrain language models on social science texts. 

The use of automatic entity linking tools comes with the risk of providing wrong or missing links. Wrong links could result in inaccurate conclusions, whereas missing links could overlook underrepresented surveys/topics or long-tail entities.

### Discussion of Biases

The publications in SSOAR mostly cover social science within Europe or have a European perspective. As such, the texts may only cover certain values that cannot generalize to other cultures.

## Additional Information

### Dataset Curators

The document IDs were collected by Tornike Tsereteli (University of Mannheim).

### Licensing Information

The publication content (i.e., the sentences) follow the same license as provided by the original source (see the source URLs for each document).