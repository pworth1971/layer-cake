#!/bin/bash

# initialize conda environment
conda init
conda create -n python312 python=3.12
conda activate python312

#
# Ubuntu pytorch install
#
conda install pytorch torchtext cudatoolkit=11.8 -c pytorch -c nvidia

#
# MacOS pytorch install (apple silicon)
#
conda install pytorch torchtext -c pytorch

pip install scikit-learn transformers simpletransformers rdflib gensim fasttext matplotlib tabulate scipy datetime numpy pandas psutil GPUtil plotly nltk fasttext seaborn
apt update
apt install zip
