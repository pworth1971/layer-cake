#!/bin/bash

# Base components
PY="python ../src/ml_class_baselines.py"
LOG="--log-file ../log/ml_baselines.test"
EMB="--embedding-dir ../.vector_cache"
GLOVE_PATH="--glove-path ../.vector_cache" 
WORD2VEC_PATH="--word2vec-path ../.vector_cache/GoogleNews-vectors-negative300.bin"
FASTTEXT_PATH="--fasttext-path ../.vector_cache/crawl-300d-2M.vec"
BERT_PATH="--bert-path ../.vector_cache"
OPTIMC="--optimc"


# Arrays of datasets and corresponding pickle paths
declare -a datasets=("20newsgroups" "reuters21578")
declare -a pickle_paths=("../pickles/20newsgroups.pickle" "../pickles/reuters21578.pickle")

# Function to run commands
function run_command() {
    local dataset=$1
    local pickle_path=$2
    local learner=$3
    local mode=$4
    local mode_path=$5

    local dataset_flag="--dataset ${dataset}"
    local pickle_flag="--pickle-dir ${pickle_path}"
    local cmd="$PY $LOG $dataset_flag $pickle_flag $EMB --learner $learner --mode $mode $mode_path $OPTIMC"

    echo
    echo $cmd
    eval $cmd
}


# Loop through datasets and run commands
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    pickle_path=${pickle_paths[$i]}
    
    # Define different mode commands for each dataset
    run_command $dataset $pickle_path "svm" "tfidf" ""
    run_command $dataset $pickle_path "svm" "glove" "$GLOVE_PATH"  
    run_command $dataset $pickle_path "svm" "glove-sup" "$GLOVE_PATH" 
    run_command $dataset $pickle_path "lr" "tfidf" ""
    run_command $dataset $pickle_path "lr" "glove" "$GLOVE_PATH"
    run_command $dataset $pickle_path "lr" "glove-sup" "$GLOVE_PATH"
    #run_command $dataset $pickle_path "svm" "word2vec" $WORD2VEC_PATH
    #run_command $dataset $pickle_path "svm" "word2vec-sup" $WORD2VEC_PATH
    #run_command $dataset $pickle_path "svm" "fasttext" $FASTTEXT_PATH
    #run_command $dataset $pickle_path "svm" "fasttext-sup" $FASTTEXT_PATH
    #run_command $dataset $pickle_path "svm" "bert" $BERT_PATH
    #run_command $dataset $pickle_path "svm" "bert-sup" $BERT_PATH
done
