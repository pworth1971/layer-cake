#!/bin/bash

# initialize conda environment
conda init
conda install pytorch torchtext cudatoolkit=11.8 -c pytorch -c nvidia
conda create -n python312 python=3.12
conda activate python312
pip install scikit-learn transformers simpletransformers rdflib gensim fasttext matplotlib tabulate scipy datetime numpy pandas psutil GPUtil plotly nltk fasttext

apt update
apt install zip
