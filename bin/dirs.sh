#!/bin/bash

# Ensure the script is being run with Bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script must be run in a Bash shell."
    echo "Please restart this script using Bash: 'bash setup.sh'."
    exit 1
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

echo "Setup complete!"