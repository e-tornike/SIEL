# GESIS Survey Item Metadata (GSIM)

## Dataset Description

### Dataset Summary

This dataset contains surveys and their associated survey items (including their metadata), which collectively make up the GESIS Survey Item Metadata (GSIM) knowledge base. GSIM entails 1,571 surveys and 524,154 survey items.

### Supported Tasks and Leaderboards

This dataset supports is used as a knowledge base for the tasks in [SIL](../sild/README.md).

### Languages

The dataset mostly includes English and German texts.

## Dataset Structure

### Data Instances

An example instance in the dataset contains a survey URL and a list of associated survey items. Each survey item additionally contain rich metadata, such as the year, the title, the question language, the question content, and more. An example istance is shown below:

```
{
  "year": "1961", 
  "title": "V8 - MEINUNG UEBER ENGLAND", 
  "variable_label": "MEINUNG UEBER ENGLAND", 
  "variable_name": "V8", 
  "question_text": "WÜRDEN SIE BITTE DIESE KARTE BENUTZEN, UM MIR IHRE ANSICHT ÜBER VERSCHIEDENE LÄNDER ZU SAGEN.<br/>", 
  "question_text_en": "WOULD YOU PLEASE USE THIS CARD TO TELL ME YOUR VIEWS ABOUT VARIOUS COUNTRIES. <br/>WHAT KIND OF OPINION DO YOU HAVE ABOUT:<br/>", 
  "question_id": "ZA0055_Q4777_QueGri", 
  "question_label": "F.2", 
  "question_lang": "de-DE", 
  "sub_question": "F.2(C) - WELCHE MEINUNG HABEN SIE ÜBER ENGLAND?", 
  "item_category": "WELCHE MEINUNG HABEN SIE ÜBER ENGLAND?", 
  "answer_categories": "SEHR GUTE MEINUNG;GUTE MEINUNG;WEDER GUTE NOCH SCHLECHTE MEINUNG;SCHLECHTE MEINUNG;SEHR SCHLECHTE MEINUNG;KEINE MEINUNG / KEINE ANGABE", 
  "topic": ["Einstellung des B zu: - verschiedene Länder"], 
  "topic_en": ["R's position towards: - different countries"], 
  "question_type1": "MultipleQuestionItem", 
  "question_type2": "Geschlossene Frage",
  ...
}
```

### Data Fields

The fields present in the dataset are the following:

- `year`: the year the the corresponding survey was conducted
- `title`: the title of the survey item
- `variable_label`: the label of the survey item (which is often included in the title)
- `variable_name`: the name of the variable (which is often included in the title)
- `question_text`: the general question of the survey item
- `question_text_en`: the English translation of the `question_text`
- `question_id`: the unique ID of the survey item used by GESIS
- `question_lang`: the language of the `question_text`
- `sub_question`: the more specific question of the survey item
- `item_category`: the category of focus of the question/sub-question
- `topic`: the list of topics that describe the suvey item
- `topic_en`: the English translation of the `topic`
- `question_type1`: the type of question that is asked from the set `{}`
- `question_type2`: the type of question that is asked from the set `{}`

## Dataset Creation

### Curation Rationale

The tasks for [SIL](../sild/README.md) require access to a knowledge base in order to link mentions to survey items. This dataset serves as the ground truth knowledge base of entities that can be linked.

### Source Data

The data originates from openly accessible surveys that are indexed in [GESIS Search](https://search.gesis.org/).

#### Initial Data Collection and Normalization

Many documents in GESIS Search are linked with the surveys they cite. We downloaded over 4,000 such publications and filtered out surveys that are not linked to survey items. Finally, we manually selected 100 documents that mentioned at least one survey item in the text.

#### Preparing the Data

The data can be created using the following commands:
```
python /sil/data/gsim/1_download_gsim.py
```

To create sentence pairs out of the metadata, run the following command:
```
python /sil/data/gsim/3_generate_sp.py
```

## Considerations for Using the Data

### Social Impact of Dataset

The dataset can be used to facilitate the interlinking of research data in the social sciences, and, as a result, the access to research along the FAIR principles (findable, accessible, interoperable, re-usable). This could improve the quality and reproducibility of research. 

The survey items associated with surveys may be incomplete, metadata may be incorrect, and they may contain errors introduced during the processing steps. As such, carefully consider consider the conclusions made using this data.

## Additional Information

### Licensing Information

The survey item content follow the same license as provided by the original source, namely [CC0 1.0 UNIVERSAL](https://creativecommons.org/publicdomain/zero/1.0/) (see re-using metadata from "Research Data" on the [GESIS FAQ](https://search.gesis.org/faq)).