# Survey Item Linking Dataset (SILD)

## Dataset Description

- **Homepage:** [Add homepage URL here if available (unless it's a GitHub repository)]()
- **Repository:** [If the dataset is hosted on github or has a github homepage, add URL here]()
- **Paper:** [If the dataset was introduced by a paper or there was a paper written describing the dataset, add URL here (landing page for Arxiv paper preferred)]()
- **Leaderboard:** [If the dataset supports an active leaderboard, add link here]()
- **Point of Contact:** [If known, name and email of at least one person the reader can contact for questions about the dataset.]()

### Dataset Summary

This dataset contains a collection of texts from publications from a broad range of social science domains (e.g., economics, politics, psychology, etc.). The texts are annotated with labels for [Survey Item Linking (SIL)](), an [Entity Linking (EL)](https://en.wikipedia.org/wiki/Entity_linking) task. SIL is divided into two sub-tasks: Mention Detection (MD), a binary text classification task, and Entity Disambiguation (ED), a sentence similarity task. Sentences that mention survey items are labeled with the IDs of entities from a knowledge base ([GSIM]()). SILD contains 20,454 sentences in English and German from 100 publications.

### Supported Tasks

This dataset supports [text classification](https://huggingface.co/tasks/text-classification) and [sentence similarity](https://huggingface.co/tasks/sentence-similarity) tasks.

- `text-classification`: The dataset can be used to train a model for Mention Detection (MD), which consists in identifying sentences within a document that explicitly or implicitly mention survey items. Success on this task is typically measured by achieving a high [F1-score](https://huggingface.co/spaces/evaluate-metric/f1). The [XLM-R-large](https://huggingface.co/docs/transformers/model_doc/xlm-roberta) model currently achieves a score of 62.6% in the binary case.
- `sentence-similarity`: The dataset can be used to train a model for Entity Disambiguation (ED), which consists in matching a sentence that mentions a survey item to a set of descriptions of entities in a knowledge base. Success on this task is typically measured by achieving a high [Recall@k](https://huggingface.co/metrics/recall), [MAP@k](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision), or [nDCG@k](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Discounted_cumulative_gain), where the top-*k* most-similar entities are evaluated. The [mE5-large](https://huggingface.co/intfloat/multilingual-e5-large) model currently achieves a Recall@10 of 80%.

### Languages

The dataset covers 84 English and 16 German publications, written in scientific language.

## Dataset Structure

### Data Instances

An example instances in the dataset contains a sentence, a binary label indicating if the sentence mentions a survey item, a language tag, a list of entity IDs, a list of entity types and subtypes corresponding to the entities, a document ID, a list of survey IDs, and a UUID. An example instance is shown below:

```
{
  'sentence': 'The survey included four indicators of socioeconomic position, namely education, subjective social status, difficulty in paying bills and occupation.',
  'label': 1,
  'lang': en,
  'entities': ['ZA4977_Varv328', 'ZA4977_Varv356', 'ZA4977_Varv355', 'ZA4977_Varv181', 'Unk'],
  'type': ['Explicit', 'Implicit', 'Explicit', 'Explicit', 'Implicit'],
  'subtype': ['Paraphrase', 'Paraphrase', 'Paraphrase', 'Paraphrase', 'Paraphrase'],
  'document_id': '74709',
  'surveys': ['ZA4977'],
  'uuid': 'c02330cd-cbd2-4df5-be16-56630626bea0',
  ...
}
```

The entity types and subtypes directly correspond to the entities (e.g.,, the first entity `ZA4977_Varv328` has an `Explicit` type and a `Paraphrase` subtype). The entity IDs can be used to lookup the corresponding descriptions in the knowledge base (GSIM). The survey IDs can be used to filter the possible candidate surveys in GSIM. In practice, this has a significant impact on the performance. Other attributes include concepts defined in the sentence and annotation confidence scores.


### Data Fields

The fields present in the dataset are the following:

- `sentence`: a string containing a sentence that is used as the input for each task
- `label`: a binary label that is used as the output for text classification (MD) (either `0` or `1`)
- `lang`: a string containing the language tag of the sentence (either `en` or `de`)
- `entities`: a list of strings that contain the entity IDs that are mentioned in the sentence, which are used as the output for sentence similarity (ED)
- `type`: a list of strings that contain the entity types for each entity in `entities`
- `subtype`: a list of strings that contain the entity subtypes for each entity in `entities`
- `document_id`: a string containing the ID of the publication where the sentence was extracted from
- `surveys`: a list of strings containing the survey IDs that are referenced in the publication, which are used to filter the possible candidate entities in GSIM
- `uuid`: a string containing the unique ID for each instance
- `concepts`: a list of strings containing the concepts defined in the sentence
- `concept_confidence`: an integer on a likerd scale of 1 to 5 that contains the confidence score the concepts
- `confidence`: an integer on a likerd scale of 1 to 5 that contains the confidence score for the annotations of the instance
- `sha256`: a string containing the SHA256 hash of the sentence (this is only present for instances that have a restrictive license, see the usage section for more details)

### Data Splits

The dataset is divided into two equally sized configurations that contain different document samples: `Diff` and `Rand`. `Diff` contains more challenging documents in the test set, which were selected manually, whereas `Rand` contains a random sample of documents. Both configurations share the same test set for German. Each configuration is divided into three dataset splits:

Dataset statistics for `SILD-Diff` are shown below:

|                         | Train | Validation | Test (EN) | Test (DE)
|-------------------------|-------:|----------:|----------:|----------:|
| Sentences (+)           |    397 |        75 |       204 |       107 |
| Sentences (-)           | 10,367 |     1,950 |     4,772 |     2,582 |
| Survey items            |    944 |       168 |       547 |       162 |
| Unique survey items     |    697 |        88 |       434 |       126 |
| Surveys                 |     45 |         8 |        45 |        16 |
| Papers                  |     50 |        10 |        24 |        16 |

## Dataset Creation

### Curation Rationale

Research in the social sciences has a long history of using surveys to understand societies by collecting information from individuals (e.g., to measure the influence of environmental factors on life satisfaction). Latent concepts, which do not appear in surveys explicitly, are first defined and then operationalized using one or more questions from surveys, which are a collection of statements or questions, called *survey items*. Due to a lack of standardization in citation practices, unique identifiers are not provided when describing the survey items used in a study. Without such identifiers, finding relevant work based on survey items of interest, which is desired by social scientists, is challenging. Consequently, there is a need for identifying used survey items to improve access to research along the FAIR principles (findable, accessible, interoperable, re-usable).

### Source Data

The data originates from 100 openly accessible social science publications from [SSOAR](https://www.gesis.org/ssoar) that are also indexed in [GESIS Search](https://search.gesis.org/).

#### Initial Data Collection and Normalization

Many documents in GESIS Search are linked with the surveys they cite. We downloaded over 4,000 such publications and filtered out surveys that are not linked to survey items. Finally, we manually selected 100 documents that mentioned at least one survey item in the text.

#### Who are the source language producers?

The publications in the dataset are written by social scientists and collected in the [SSOAR](https://www.gesis.org/ssoar) database.

### Annotations

Each publication is annotated at the sentence-level with survey items that it mentions.

#### Annotation process

The annotations were carried out on the [INCEpTION](https://github.com/inception-project/inception) platform. Two annotators labeled 12 documents, which were used to compute the inter-annotate agreement (Cohen's $\kappa$ = 0.66 for MD and Krippendorff's $\alpha$ = 0.43 for ED). One of the annotators additionally labeled 88 more. The annotators were first trained on a different set of documents over a number of rounds. On average, it took around 30 minutes to annotate a full document. The guideline is available [here](https://github.com/e-tornike/SIL/).

#### Who are the annotators?

The data was annotated in full by Tornike Tsereteli (a PhD student at the University of Mannheim). Jan Hendrik Blaß (a master's student in political science at University of Mannheim) was fairly compensated for annotating 12 documents.

### Personal and Sensitive Information

The dataset contains openly-accessible texts from scientific publications. No personal or sensitive information is known to the curators of the dataset.

## Considerations for Using the Data

### Social Impact of Dataset

The dataset can be used to facilitate the interlinking of research data in the social sciences, and, as a result, the access to research along the FAIR principles (findable, accessible, interoperable, re-usable). This could improve the quality and reproducibility of research. 

The use of automatic entity linking tools comes with the risk of providing wrong or missing links. Wrong links could result in inaccurate conclusions, whereas missing links could overlook underrepresented surveys/topics or long-tail entities.

### Discussion of Biases

The publications in this dataset may introduce a bias towards a specific set of topics or subfields. Each publication on SSOAR is tagged with keywords that summarize the fine-grained topics (e.g., public opinion) and classifications that cover broader topics (e.g., political process). The similarity between the distributions of the publications in the dataset and the entire SSOAR collection, based on the Jensen-Shannon divergence, is moderate (0.32) for keywords and high (0.11) for classifications. However, not all topics are covered in the dataset. As a result, the models trained on the data may underperform for certain topics.

## Additional Information

### Dataset Curators

The data was collected by Tornike Tsereteli (University of Mannheim).

### Licensing Information

The publication content (i.e., the sentences) follow the same license as provided by the original source (see the source URLs for each document in the [document_licenses.tsv]() file). The annotations are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### Citation Information

The dataset is archived under the following DOI:

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.18914.svg)](http://dx.doi.org/10.5281/zenodo.11397370)

### Contributions

Thanks to [@e-tornike](https://github.com/e-tornike) for adding this dataset.