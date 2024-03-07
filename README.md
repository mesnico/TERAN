# Transformer Encoder Reasoning and Alignment Network (TERAN)

## Updates

- :fire: 09/2022: The extension to this work (**ALADIN: Distilling Fine-grained Alignment Scores for Efficient Image-Text Matching and Retrieval**) has been published in proceedings of CBMI 2022. Check out [code](https://github.com/mesnico/ALADIN) and [paper](https://arxiv.org/abs/2207.14757)!

## Introduction

Code for the cross-modal visual-linguistic retrieval method from "Fine-grained Visual Textual Alignment for Cross-modal Retrieval using Transformer Encoders", accepted for publication in ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) [[Pre-print PDF](https://arxiv.org/abs/2008.05231)].

This work is an extension to our previous approach TERN accepted at ICPR 2020.

This repo is built on top of [VSE++](https://github.com/fartashf/vsepp) and [TERN](https://github.com/mesnico/TERN).

<p align="center">
  <b>Fine-grained Alignment for Precise Matching</b> <br> <br>
  <img src="figures/alignment.jpg" width="80%">
</p>

<p align="center">
  <b>Retrieval</b> <br> <br>
  <img src="figures/retrieval.png" width="80%">
</p>


## Setup

1. Clone the repo and move into it:
```
git clone https://github.com/mesnico/TERAN
cd TERAN
```

2. Setup python environment using conda:
```
conda env create --file environment.yml
conda activate teran
export PYTHONPATH=.
```

2.1 Setup minimal python environment for CUDA 10.1 using conda:
```
conda env create --file environment_min.yml
conda activate teran
export PYTHONPATH=.
```
## Get the data
Data and pretrained models be downloaded from this [OneDrive link](https://cnrsc-my.sharepoint.com/:f:/g/personal/nicola_messina_cnr_it/EnsuSFo-rG5Pmf2FhQDPe7EBCHrNtR1ujSIOEcgaj5Xrwg?e=Ger6Sl) (see the steps below to understand which files you need):

1. Download and extract the data folder, containing annotations, the splits by Karpathy et al. and ROUGEL - SPICE precomputed relevances for both COCO and Flickr30K datasets. Extract it:

```
tar -xvf data.tgz
```

2. Download the bottom-up features for both COCO and Flickr30K. We use the code by [Anderson et al.](https://github.com/peteanderson80/bottom-up-attention) for extracting them.
The following command extracts them under `data/coco/` and `data/f30k/`. If you prefer another location, be sure to adjust the configuration file accordingly.
```
# for MS-COCO
tar -xvf features_36_coco.tgz -C data/coco

# for Flickr30k
tar -xvf features_36_f30k.tgz -C data/f30k
```

## Evaluate
Extract our pre-trained TERAN models:
```
tar -xvf TERAN_pretrained_models.tgz
```

Then, issue the following commands for evaluating a given model on the 1k (5fold cross-validation) or 5k test sets.
```
python3 test.py pretrained_models/[model].pth --size 1k
python3 test.py pretrained_models/[model].pth --size 5k
```

Please note that if you changed some default paths (e.g. features are in another folder than `data/coco/features_36`), you will need to use the `--config` option and provide the corresponding yaml configuration file containing the right paths.
## Train
In order to train the model using a given TERAN configuration, issue the following command:
```
python3 train.py --config configs/[config].yaml --logger_name runs/teran
```
`runs/teran` is where the output files (tensorboard logs, checkpoints) will be stored during this training session.

## Visualization 

WIP

## Reference
If you found this code useful, please cite the following paper:

    @article{messina2021fine,
      title={Fine-grained visual textual alignment for cross-modal retrieval using transformer encoders},
      author={Messina, Nicola and Amato, Giuseppe and Esuli, Andrea and Falchi, Fabrizio and Gennaro, Claudio and Marchand-Maillet, St{\'e}phane},
      journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
      volume={17},
      number={4},
      pages={1--23},
      year={2021},
      publisher={ACM New York, NY}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
