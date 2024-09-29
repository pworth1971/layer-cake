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
    if [[ "$CONDA_DEFAULT_ENV" != "python312" ]]; then
        echo "ERROR: The conda environment 'python312' is not activated."
        echo "Please ensure that 'conda activate python312' is successful before proceeding."
        exit 1
    fi
}

# Initialize conda environment
echo "Initializing conda..."
conda init
conda create -n python312 python=3.12 -y

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate python312

# Check if the conda environment is activated correctly
check_conda_env

# Detect OS and install PyTorch accordingly
OS=$(detect_os)

if [[ "$OS" == "macOS" ]]; then
    echo "Detected macOS (Apple Silicon). Installing PyTorch for macOS..."
    conda install pytorch torchtext -c pytorch -y

elif [[ "$OS" == "ubuntu" ]]; then
    echo "Detected Ubuntu. Installing PyTorch with CUDA support..."
    conda install pytorch torchtext cudatoolkit=11.8 -c pytorch -c nvidia -y

else
    echo "Unsupported OS. This script only supports macOS and Ubuntu."
    exit 1
fi

# Core package dependencies
echo "Installing core package dependencies..."
pip install scikit-learn transformers simpletransformers rdflib gensim fasttext matplotlib tabulate scipy datetime numpy pandas psutil GPUtil plotly nltk seaborn fairscale

# Install Hugging Face CLI for LLaMa models
echo "Installing Hugging Face Hub and logging in for LLaMa models..."
pip install huggingface_hub

# Prompt user to log in to Hugging Face
huggingface-cli login
echo "Please enter your Hugging Face token for authentication to use LLaMa models."

# Handle optional Ubuntu-specific setup
if [[ "$OS" == "ubuntu" ]]; then
    echo "Updating Ubuntu packages and installing zip..."
    sudo apt update
    sudo apt install zip -y
fi

echo "Setup complete!"
