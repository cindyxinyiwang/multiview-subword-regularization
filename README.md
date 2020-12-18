# Multi-view Subword Regularization 

[**Paper**](https://arxiv.org/pdf/2003.11080.pdf) |

This repository contains the implementation for Multi-view Subword Regularization.
Our code is based on the [XTREME]() benchmark

# Introduction


# Download the data
We simply use the data downloading instruction from the official XTREME repo.

In order to run experiments on XTREME, the first step is to download the dependencies. We assume you have installed [`anaconda`](https://www.anaconda.com/) and use Python 3.7+. The additional requirements including `transformers`, `seqeval` (for sequence labelling evaluation), `tensorboardx`, `jieba`, `kytea`, and `pythainlp` (for text segmentation in Chinese, Japanese, and Thai), and `sacremoses` can be installed by running the following script:
```
bash install_tools.sh
```

The next step is to download the data. To this end, first create a `download` folder with ```mkdir -p download``` in the root of this project. You then need to manually download `panx_dataset` (for NER) from [here](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN) (note that it will download as `AmazonPhotos.zip`) to the `download` directory. Finally, run the following command to download the remaining datasets:
```
bash scripts/download_data.sh
```

Note that in order to prevent accidental evaluation on the test sets while running experiments,
we remove labels of the test data during pre-processing and change the order of the test sentences
for cross-lingual sentence retrieval.

## Wikiann named entity recognition

For named entity recognition (NER), we use data from the Wikiann (panx) dataset. 
To fine-tune a pretrained multilingual model on English using Multi-view Subword Regularization:
```
bash mvr_scripts/train_mv_panx.sh
```

## PAXS-X sentence classification

For sentence classification, we use the Cross-lingual Paraphrase Adversaries from Word Scrambling (PAWS-X) dataset. You can fine-tune a pre-trained multilingual model on the English PAWS data with the following command:
```
bash scripts/train.sh [MODEL] pawsx
```

## XNLI sentence classification

The second sentence classification dataset is the Cross-lingual Natural Language Inference (XNLI) dataset. You can fine-tune a pre-trained multilingual model on the English MNLI data with the following command:
```
bash scripts/train.sh [MODEL] xnli
```

## XQuAD, MLQA, TyDiQA-GoldP question answering

For question answering, we use the data from the XQuAD, MLQA, and TyDiQA-Gold Passage datasets.
For XQuAD and MLQA, the model should be trained on the English SQuAD training set. For TyDiQA-Gold Passage, the model is trained on the English TyDiQA-GoldP training set. Using the following command, you can first fine-tune a pre-trained multilingual model on the corresponding English training data, and then you can obtain predictions on the test data of all tasks.
```
bash scripts/train.sh [MODEL] [xquad,mlqa,tydiqa]
```

# Paper

Please cite our paper `\cite{hu2020xtreme}`.
```
@article{hu2020xtreme,
      author    = {Junjie Hu and Sebastian Ruder and Aditya Siddhant and Graham Neubig and Orhan Firat and Melvin Johnson},
      title     = {XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization},
      journal   = {CoRR},
      volume    = {abs/2003.11080},
      year      = {2020},
      archivePrefix = {arXiv},
      eprint    = {2003.11080}
}
```
Please consider including a note similar to the one below to make sure to cite all the individual datasets in your paper.

We experiment on the XTREME benchmark `\cite{hu2020xtreme}`, a composite benchmark for multi-lingual learning consisting of data from the XNLI `\cite{Conneau2018xnli}`, PAWS-X `\cite{Yang2019paws-x}`, UD-POS `\cite{nivre2018universal}`, Wikiann NER `\cite{Pan2017}`, XQuAD `\cite{artetxe2020cross}`, MLQA `\cite{Lewis2020mlqa}`, TyDiQA-GoldP `\cite{Clark2020tydiqa}`, BUCC 2018 `\cite{zweigenbaum2018overview}`, Tatoeba `\cite{Artetxe2019massively}` tasks. We provide their BibTex information as follows.
```
@inproceedings{Conneau2018xnli,
    title = "{XNLI}: Evaluating Cross-lingual Sentence Representations",
    author = "Conneau, Alexis  and
      Rinott, Ruty  and
      Lample, Guillaume  and
      Williams, Adina  and
      Bowman, Samuel  and
      Schwenk, Holger  and
      Stoyanov, Veselin",
    booktitle = "Proceedings of EMNLP 2018",
    year = "2018",
    pages = "2475--2485",
}

@inproceedings{Yang2019paws-x,
    title = "{PAWS-X}: A Cross-lingual Adversarial Dataset for Paraphrase Identification",
    author = "Yang, Yinfei  and
      Zhang, Yuan  and
      Tar, Chris  and
      Baldridge, Jason",
    booktitle = "Proceedings of EMNLP 2019",
    year = "2019",
    pages = "3685--3690",
}

@article{nivre2018universal,
  title={Universal Dependencies 2.2},
  author={Nivre, Joakim and Abrams, Mitchell and Agi{\'c}, {\v{Z}}eljko and Ahrenberg, Lars and Antonsen, Lene and Aranzabe, Maria Jesus and Arutie, Gashaw and Asahara, Masayuki and Ateyah, Luma and Attia, Mohammed and others},
  year={2018}
}

@inproceedings{Pan2017,
author = {Pan, Xiaoman and Zhang, Boliang and May, Jonathan and Nothman, Joel and Knight, Kevin and Ji, Heng},
booktitle = {Proceedings of ACL 2017},
pages = {1946--1958},
title = {{Cross-lingual name tagging and linking for 282 languages}},
year = {2017}
}

@inproceedings{artetxe2020cross,
author = {Artetxe, Mikel and Ruder, Sebastian and Yogatama, Dani},
booktitle = {Proceedings of ACL 2020},
title = {{On the Cross-lingual Transferability of Monolingual Representations}},
year = {2020}
}

@inproceedings{Lewis2020mlqa,
author = {Lewis, Patrick and Oğuz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
booktitle = {Proceedings of ACL 2020},
title = {{MLQA: Evaluating Cross-lingual Extractive Question Answering}},
year = {2020}
}

@inproceedings{Clark2020tydiqa,
author = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki},
booktitle = {Transactions of the Association of Computational Linguistics},
title = {{TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages}},
year = {2020}
}

@inproceedings{zweigenbaum2018overview,
  title={Overview of the third BUCC shared task: Spotting parallel sentences in comparable corpora},
  author={Zweigenbaum, Pierre and Sharoff, Serge and Rapp, Reinhard},
  booktitle={Proceedings of 11th Workshop on Building and Using Comparable Corpora},
  pages={39--42},
  year={2018}
}

@article{Artetxe2019massively,
author = {Artetxe, Mikel and Schwenk, Holger},
journal = {Transactions of the ACL 2019},
title = {{Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond}},
year = {2019}
}
```
