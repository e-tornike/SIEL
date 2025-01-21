[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.18914.svg)](http://dx.doi.org/10.5281/zenodo.11397370)

# Enriching Social Science Research via Survey Item Linking (SIL)

This repository is the official implementation of [Enriching Social Science Research via Survey Item Linking (2024)]().

![A figure showing the pipeline for Survey Item Linking](./images/sil-pipeline.svg)

## Requirements

To install requirements, use either [poetry](https://python-poetry.org/) or [pip](https://pip.pypa.io/en/stable/):

```setup
poetry install
poetry install --only data_s44k # if you want to reproduce the S44k dataset
poetry install --only data_gsim # if you want to reproduce the GSIM dataset
```

```setup
pip install -r requirements/requirements.txt
pip instlal -r requirements/data_s44k.txt # if you want to reproduce the S44k dataset
pip install -r requirements/data_gsim.txt # if you want to reproduce the GSIM dataset
```

> [!IMPORTANT]
> After installing the requirements, supplemenraty datasets can be re-created (except for [SILD](/data/sild/README.md), which should be downloaded from [here](https://forms.gle/EFx3qDfUGN9XnJWT9) and placed into the `/data/sild/` directory) by following the instructions for each ([GSIM](/data/gsim/README.md), [LLM-Gen](/data/llm_gen/README.md), or [S44k](/data/s44k/README.md)).
> SILD is archived on [Zenodo](http://dx.doi.org/10.5281/zenodo.11397370).

## Experiments

To run the experiments in the paper, run the following commands:

```train
bash ./experiments/md/pretrain.slurm  # continue pretraining PLMs on S44k
bash ./experiments/md/train_linear.sh  # train linear classifiers on SILD
bash ./experiments/md/train_linear_da.sh  # train linear classifiers using data augmentation
bash ./experiments/md/train_plms.sh  # fine-tune PLMs on SILD
bash ./experiments/md/train_plms_da.sh  # fine-tune PLMS using data augmentation
bash ./experiments/md/train_knn.sh  # train kNN on SILD
bash ./experiments/md/eval_rac.sh  # combine the best PLM w/ the best kNN
bash ./experiments/md/eval_icl.sh  # evaluate In-Context Learning
bash ./experiments/ed/eval_bm25.sh  # evaluate BM25
bash ./experiments/ed/eval_plms.sh  # evaluate PLMs (including sentence transformers)
bash ./experiments/ed/train_sosse.sh  # train SoSSE models by fine-tuning sentence-transformers on GSIM and LLM-Gen
bash ./experiments/ed/eval_sosse.sh  # evaluate SoSSE models
```

## Models

> [!IMPORTANT]
> The models will be uploaded to [HuggingFace Hub](https://huggingface.co/models) soon!

You can download multilingual pretrained models for the social science domain:

- [SSOAR-XLM-R-base](https://huggingface.co/e-tornike/ssoar-xlm-roberta-base) is pre-trained on S44k using masked language modeling (MLM), a batch size of 8, and a sequence length of 512 tokens. 

You can download multilingual fine-tuned models for MD on SILD, which used a batch size of 32 and a sequence length of 64, here:

- [XLM-R-base-SILD](https://huggingface.co/e-tornike/xlm-roberta-base-sild) is fine-tuned on SILD using ... . 
- [XLM-R-large-SILD](https://huggingface.co/vadis/xlm-roberta-large-finetuned-sild-md) is fine-tuned on SILD using ... . 
- [SSOAR-XLM-R-base-SILD](https://huggingface.co/e-tornike/ssoar-xlm-roberta-base-sild) is pre-trained on S44k and then fine-tuned on SILD using ... . 

You can download multilingual fine-tuned models for ED on LLM-Gen, which used a batch size of 1024 and a sequence length of 512, here:

- [SoSSE-mE5-base](https://huggingface.co/e-tornike/sosse-multilingual-e5-base) is fine-tuned on LLM-Gen using ... .

## Results

Our model achieves the following performance on:

### [Mention Detection (MD) on SILD](https://paperswithcode.com/)

| Model name         | F1-binary (English) | F1-binary (German) | F1-binary (Total)
| ------------------ |---------------- | -------------- | -------------- |
| [XLM-R-base-SILD](https://huggingface.co/e-tornike/xlm-roberta-base-sild)   |     58.5%         |      53.9%       | 57.1% |
| [SSOAR-XLM-R-base-SILD](https://huggingface.co/e-tornike/ssoar-xlm-roberta-base-sild)   |     60.7%         |      61.8%       | 61.0% |
| [XLM-R-large-SILD](https://huggingface.co/e-tornike/xlm-roberta-large-sild)   |     61.4%         |      65.1%       | 62.6% |

### [Entity Disambiguation (ED) on SILD](https://paperswithcode.com/)

| Model name              | MAP@10 (English)  | MAP@10 (German) |
| ----------------------- | ----------------- | --------------- |
| [mE5-base](https://huggingface.co/intfloat/multilingual-e5-base) (baseline)            |     57.9%         |      65.6%      |
| [SoSSE-mE5-base](https://huggingface.co/e-tornike/sosse-multilingual-e5-base)   |     63.2%         |      68.1%      |

## Licensing Information

Dataset licensing can be found under the respective directories ([SILD](/data/sild/README.md), [GSIM](/data/gsim/README.md), [LLM-Gen](/data/llm-gen/README.md), or [S44k](/data/s44k/README.md)). This work (including the models and the annotations) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
