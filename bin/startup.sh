#!/bin/bash

conda init
conda create -n python38 python=3.8
conda activate python38
pip install scikit-learn fasttext transformers simpletransformers rdflib gensim fasttext matplotlib tabulate scipy datetime numpy pandas psutil GPUtil plotly nltk fasttext
conda install pytorch torchtext cudatoolkit=11.8 -c pytorch -c nvidia
apt update
apt install zip
