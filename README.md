# Algorithmic Discrimination for NLP techniques

This repository contains the work made for my Master's Thesis on **Algorithmic Discrimination in Natural Language Processing**. 
It contains several experiments with the aim to detect **biases** based on gender, sexual orientation, ethnicity and more, 
in common algorithmic tecniques used to process natural languages (english and italian, in our case).

## Repository structure
- `data` contains the dataset files:
  - `data/bnc` is the serialized dataset of the British National Corpus, a collection of ~22k sentences in english language.
  - `data/WinoGender` contains two `.tsv` tables from the WinoGender datasets.
- `libs` contains scripts taken from external projects, optionally adapted for the case.
- `results` is the folder containing the results produced by the experiments. They can be tables (`.tsv`) os images (`.png`).
- `saved` contains serialized files of models and other _Python_ objects. If the creation of these objects is too slow, we used the Python library pickle to save a pre-computed version of them.
- `src`, finally, is the folder of the source code of the project:
  - `src/experiments` contains the main scripts for the experiments.
  - `src/models` contains the classes for ML models.
  - `src/parsers` contains the scripts used to parse datasets in the `data` folder.
  - `src/viewers` contains the scripts and classes used to produce outputs in the `results` folder.
- `main.py` is the launching script.

## Experiments
We now describe briefly how the implemented experiments work:

### Detecting gender anomalies by _Surprise_
Inspired by the paper _"How is BERT surprised? Layerwise detection of linguistic anomalies"_ by Li et al., this test aims to detect
anomalies in opposite-by-gender pairs of sentences. A metric called **surprise** is used to measure the un-likelyhood of tokens
in the BERT encoding.

Every pair of sentences contains an occupation word, such as "teacher", "doctor", "engineer", etc., and a gendered english pronoun (he/she, him/her, his/her).
Our thesis is that the _Anomaly Model_ detects a higher **surprise** when the occupation token is used in a sentence with a pronoun of a gender which is not frequently associated with it in the real world.

### Analyzing the embeddings spatial distribution
This experiments recalls a variety of papers on algorithmic discrimination studying the distribution of word embeddings in the output embedding space.
However, in our case, the BERT encoder is used to produce **contextual word embeddings**.

We tried to define standard embeddings for the occupation words, abstracting them from the context and plotting their coordinates in a reduced 2D or 3D space (via Principal Components Analysis).

### Detecting the _Gender Subspace_
_{work in progress}_

### Measuring the _Gender Bias_ for contextual embeddings in opposite-gender contexts
_{work in progress}_
