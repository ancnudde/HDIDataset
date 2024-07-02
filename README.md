# Comparison Named Entity Recognition/Generative Entities Extraction

## INTRODUCTION

This repository contains the code supporting the paper $NameOfThePaper.
It contains four folders, one for each task:

- **Named Entity Recognition** for **Drug-Drug Interactions dataset**
- **Named Entity Recognition** for **Herb-Drug Interactions dataset**
- **Generative entities extraction** for **for Drug-Drug Interactions dataset**
- **Generative entities extraction** for **for Herb-Drug Interactions dataset**

Each repo provides datasets, code for dataset processing, training (when required) and inference, configuration/prompt files and bash script to run a pipeline. 

> The pipeline uses pre-defined splits available in the "datasplits" folders and do not create new splits; this ensures consistancy with results presented in the article.

## Protocols

### Named Entity Recognition

Named entity recognition (NER) models use the Spacy library. Spacy provides pipelines for multiple tasks, incuding Named entity recognition. The pipelines used are composed as follows:
- Transformer for the embedding part: **microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext**
- NER classification based on **CRF**
- **Warmup linear scheduler with decay** for learning rate optimization, with different values of learning rate tested (1.10<sup>-5</sup>, 5.10<sup>-5</sup>, 1.10<sup>-4</sup>, 5.10<sup>-4</sup>, 1.10<sup>-3</sup>)

The complete configuration files used for training are available in the corresponding $/config$ folders.

The models where trained and tested on two disctincts datasets:
- The DDI Corpus (https://doi.org/10.1016/j.jbi.2013.07.011), a corpus of drug-drug interaction primarily focused on relation extraction but providing annotatons for named entities
- A custom Herb-Drug Interaction corpus, annotated by us. This dataset focuses not only on herbs and drugs but also on contextual information (pathologies, sex, age, herbs preparation, cohorts, ...). More information about the dataset are available in the article.

### Generative Entities Extraction

Generative Entities Extraction is performed using **small versions of Large Language Models**. In this work, we tested the performances of:
- **Mistral 7B**: TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
- **Phi3 small**: kaitchup/Phi-3-mini-4k-instruct-gptq-4bit

The models used are 4-bit **quantized** and obtained from the HuggingFace platform. All models where used sing HuggingFace **transformers** library.

The same datasets as for NER are used, but their are adapted to specificities of generative process (removed redundant occurrences, removed spans, ...; see the paper for a more detailed analysis of differences in these tasks).