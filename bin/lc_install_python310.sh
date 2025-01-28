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
    pip install torch==2.4.1 torchvision==0.16.1 triton==3.0.0 --index-url https://download.pytorch.org/whl/cpu
elif [[ "$OS" == "ubuntu" ]]; then
    echo "Detected Ubuntu. Installing PyTorch with CUDA support..."
    pip install torch==2.4.1 torchvision==0.16.1 triton==3.0.0 --index-url https://download.pytorch.org/whl/cu118
else
    echo "Unsupported OS. This script only supports macOS and Ubuntu."
    exit 1
fi

# Core package dependencies
echo "Installing core package dependencies..."
pip install transformers==4.46.3 safetensors==0.4.5 scikit-learn simpletransformers rdflib gensim fasttext matplotlib tabulate scipy datetime numpy pandas psutil GPUtil plotly nltk seaborn fairscale huggingface_hub accelerate bokeh

# Handle optional Ubuntu-specific setup
if [[ "$OS" == "ubuntu" ]]; then
    echo "Updating Ubuntu packages and installing zip..."
    sudo apt update
    sudo apt install zip -y
fi

echo "Setup complete!"
