#!/bin/bash

# Ensure the script is being run with Bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script must be run in a Bash shell."
    echo "Please restart this script using Bash: 'bash setup.sh'."
    exit 1
fi

# ----------------------------------------
# Startup script for environment setup
# Supports MacOS and Ubuntu
# ----------------------------------------

# Language models and datasets arrays
LANGUAGE_MODELS=("GloVe" "Word2Vec" "fastText" "BERT" "RoBERTa" "DistilBERT" "XLNet" "GPT2" "Llama" "DeepSeek")
DATASETS=("bbc-news" "ohsumed" "20newsgroups" "reuters21578" "arxiv" "imdb" "arxiv_protoformer" "rcv1")

# Function to detect OS type
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "ubuntu"
    else
        echo "unsupported"
    fi
}

# Function to check if conda environment is activated
check_conda_env() {
    if [[ "$CONDA_DEFAULT_ENV" != "python310" ]]; then
        echo "ERROR: The conda environment 'python310' is not activated."
        echo "Please ensure that 'conda activate python310' is successful before proceeding."
        exit 1
    fi
}

# Initialize conda environment
echo "Initializing conda..."
conda init
conda create -n python310 python=3.10 -y

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate python310

# Check if the conda environment is activated correctly
check_conda_env

# Detect OS and install PyTorch accordingly
OS=$(detect_os)

if [[ "$OS" == "macOS" ]]; then
    echo "Detected macOS (Apple Silicon). Installing PyTorch for macOS..."
    conda install pytorch=2.4.1 torchtext transformers safetensors=0.4.5 -c pytorch -y
elif [[ "$OS" == "ubuntu" ]]; then
    echo "Detected Ubuntu. Installing PyTorch with CUDA support..."
    conda install pytorch=2.4.1 torchtext transformers safetensors=0.4.5 cudatoolkit -c pytorch -c nvidia -y
else
    echo "Unsupported OS. This script only supports macOS and Ubuntu."
    exit 1
fi

# Core package dependencies
echo "Installing core package dependencies..."
pip install scikit-learn simpletransformers rdflib gensim fasttext matplotlib tabulate scipy datetime numpy pandas psutil GPUtil plotly nltk seaborn fairscale accelerate bokeh

# Prompt user to log in to Hugging Face
huggingface-cli login
echo "Please enter your Hugging Face token for authentication to use LLaMa models."

# Handle optional Ubuntu-specific setup
if [[ "$OS" == "ubuntu" ]]; then
    echo "Updating Ubuntu packages and installing zip..."
    sudo apt update
    sudo apt install zip -y
fi

# Create .vector_cache directories
echo "Creating .vector_cache directories..."
VECTOR_CACHE="../.vector_cache"
LANGUAGE_MODELS=('GloVe' 'Word2Vec' 'fastText' 'BERT' 'RoBERTa' 'DistilBERT' 'XLNet' 'GPT2' 'Llama' 'DeepSeek')

for model in "${LANGUAGE_MODELS[@]}"; do
    dir="${VECTOR_CACHE}/${model}"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created: $dir"
    else
        echo "Directory already exists: $dir"
    fi
done

# Create dataset directories
echo "Creating dataset directories..."
DATASET_DIR="../datasets"
DATASETS=('bbc-news' 'ohsumed' '20newsgroups' 'reuters21578' 'arxiv' 'imdb' 'arxiv_protoformer' 'rcv1')

for dataset in "${DATASETS[@]}"; do
    dir="${DATASET_DIR}/${dataset}"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created: $dir"
    else
        echo "Directory already exists: $dir"
    fi
done

# Create additional directories (pickles, out, log)
echo "Creating additional requisite directories..."
ADDITIONAL_DIRS=('../pickles' '../out' '../log')

for additional_dir in "${ADDITIONAL_DIRS[@]}"; do
    if [ ! -d "$additional_dir" ]; then
        mkdir -p "$additional_dir"
        echo "Created: $additional_dir"
    else
        echo "Directory already exists: $additional_dir"
    fi
done

echo "All requistite directories have been created successfully, pls download the dataset files as needed."

#
# Set up Git configuration
# update with your git info
#
echo "Setting up Git configuration..."
git config --global user.name "Peter Worth"
git config --global user.email "peterworthjr@gmail.com"


echo "Setup complete!"