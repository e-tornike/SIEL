## GESIS Survey Item Metadata (GSIM) Knowledge Base
This directory contains the scripts to reproduce the GSIM knowledge base.

Download survey items from GESIS usign a list of IDs ([survey_ids.txt](../../../data/gsim/survey_ids.txt)):

```
python 1_download_gsim.py
```

Using GSIM, metadata pairs and LLM-generated pairs can be created using the following scripts.

For metadata pairs:
```
python 2_generate_pairs.py
```

For [LLM-Gen](../../../data/llm-gen/README.md):
```
python 3_generate_llm_pairs.py
```