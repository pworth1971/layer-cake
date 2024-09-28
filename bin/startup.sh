#!/bin/bash

#
# install script, run each command separately from command line (terminal)
#

# ------------------------------------------------------------------------------------------
#
# initialize conda environment
conda init
conda create -n python312 python=3.12
conda activate python312
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
#
# install based on OS and environments, either CUDA pytorch (x84) or MacOS apple silicon (ARM)
#
#
# Option A: Ubuntu pytorch install
#
conda install pytorch torchtext cudatoolkit=11.8 -c pytorch -c nvidia
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
#
# Option B: MacOS pytorch install (apple silicon)
#
conda install pytorch torchtext -c pytorch
# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# core package dependencies
#
pip install scikit-learn transformers simpletransformers rdflib gensim fasttext matplotlib tabulate scipy datetime numpy pandas psutil GPUtil plotly nltk fasttext seaborn
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#
# for LlAma models
# NB you must create an account on huggingface and request access
# after which a token will be granted to you
#
pip install huggingface_hub
huggingface-cli login
(enter HF token for authentication to use LlaMa models)

#
# for rcv1 dataset
# required on Ubuntu, comes standard on MacOS unix variant
#
apt update
apt install zip
