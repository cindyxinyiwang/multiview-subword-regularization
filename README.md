# Multi-view Subword Regularization 

This repository contains the implementation for [Multi-view Subword Regularization]().

```
Multi-view Subword Regularization
Xinyi Wang, Sebastian Ruder, Graham Neubig
NAACL 2021
```

Our code is based on the [XTREME](https://github.com/google-research/xtreme) benchmark

# Introduction
Multilingual pretrained models uses a single subword segmentation model on data from hundreds of languages. This often lead suboptimal subword segmentations which hinders effective cross-lingual transfer. In this paper we propose a simple and efficient subword regularization approach at **fine-tuning** time of pretrained models. It utilizes both deterministic and probabilisitc segmentations of a input and enforces the consistency between the two. 


# Main method implementation
We implement Multi-view Subword Regularization(MVR) for several different tasks. For example, the main logit of MVR for sequence tagging is [here](https://github.com/cindyxinyiwang/multiview-subword-regularization/blob/main/third_party/run_mv_tag.py#L197). 


# Download the data
We simply use the data downloading instruction from the official XTREME repo.

To install the dependencies:
```
bash install_tools.sh
```

The next step is to download the data. To this end, first create a `download` folder with ```mkdir -p download``` in the root of this project. You then need to manually download `panx_dataset` (for NER) from [here](https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN) (note that it will download as `AmazonPhotos.zip`) to the `download` directory. Finally, run the following command to download the remaining datasets:
```
bash scripts/download_data.sh
```

## Wikiann named entity recognition

For named entity recognition (NER), we use data from the Wikiann (panx) dataset. 
To fine-tune a pretrained multilingual model on English using Multi-view Subword Regularization:
```
bash mvr_scripts/train_mvr_panx.sh [MODEL]
```

## PAXS-X sentence classification

For sentence classification, we use the Cross-lingual Paraphrase Adversaries from Word Scrambling (PAWS-X) dataset. You can fine-tune a pre-trained multilingual model on the English PAWS data with the following command:
```
bash mvr_scripts/train_mvr_pawsx.sh [MODEL]
```

## XNLI sentence classification

The second sentence classification dataset is the Cross-lingual Natural Language Inference (XNLI) dataset. You can fine-tune a pre-trained multilingual model on the English MNLI data with the following command:
```
bash mvr_scripts/train_mvr_xnli.sh [MODEL]
```

## XQuAD, MLQA question answering

For question answering, we use the data from the XQuAD, MLQA Passage datasets.
For XQuAD and MLQA, the model should be trained on the English SQuAD training set. 
```
bash mvr_scripts/train_mvr_qa.sh [MODEL]
```

# Paper

Please cite our paper `\cite{wang2021multiview}`.
```
@inproceedings{wang2021multiview,
      author    = {Xinyi Wang and Sebastian Ruder and Graham Neubig},
      title     = {Multi-view Subword Regularization},
      year      = {2021},
      booktitle = {NAACL}
}
```
