#!/bin/bash

#
# Base components
#
PY="python ../src/ml_class_baselines.py"

LOG="--log-file ../log/syslog_testing.test"

EMB="--embedding-dir ../.vector_cache"

GLOVE_PATH="--glove-path ../.vector_cache" 
WORD2VEC_PATH="--word2vec-path ../.vector_cache/GoogleNews-vectors-negative300.bin"
FASTTEXT_PATH="--fasttext-path ../.vector_cache/crawl-300d-2M.vec"
BERT_PATH="--bert-path ../.vector_cache"
LLAMA_PATH="--llama-path ../.vector_cache"

OPTIMC="--optimc"

#
# Full Arrays of datasets and corresponding pickle paths
#
#declare -a datasets=("reuters21578" "20newsgroups" "ohsumed" "rcv1")
#declare -a pickle_paths=("../pickles/reuters21578.pickle" "../pickles/20newsgroups.pickle" "../pickels" "../pickels")
#declare -a modes=("tfidf" "glove" "glove-sup" "word2vec" "word2vec-sup" "fasttext" "fasttext-sup" "bert" "bert-sup" "llama" "llama-sup")
#declare -a models=("svm" "lr" "nb")

#ng_dataset="--dataset 20newsgroups --pickle-dir ../pickles"                # 20_newsgroups (single label, 20 classes)
#ohm_dataset="--dataset ohsumed --pickle-dir ../pickles"                    # ohsumed (multi-label, 23 classes)
#reut_dataset="--dataset reuters21578 --pickle-dir ../pickles"              # reuters21578 (multi-label, 115 classes)
#rcv_dataset="--dataset rcv1 --pickle-dir ../pickles"                       # RCV1-v2 (multi-label, 101 classes)

#declare -a datasets=("reuters21578" "20newsgroups" "ohsumed" "rcv1")
#declare -a pickle_paths=("../pickles" "../pickles" "../pickles" "../pickles")
declare -a datasets=("20newsgroups")
declare -a pickle_paths=("../pickles")
declare -a models=("svm" "lr" "nb")
declare -a modes=("tfidf" "glove" "glove-sup" "word2vec" "word2vec-sup" "fasttext" "fasttext-sup" "bert" "bert-sup" "llama" "llama-sup")


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

    # Execute the base command
    echo
    echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo "Running command: $cmd"
    eval $cmd

    # Execute the command with the --count argument
    local cmd_count="$cmd --count"
    echo
    echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo "Running command with count: $cmd_count"
    eval $cmd_count
}

# Loop through datasets and run commands
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    pickle_path=${pickle_paths[$i]}

    for learner in "${models[@]}"; do
        for mode in "${modes[@]}"; do
            mode_path=""
            if [[ "$mode" == "glove" || "$mode" == "glove-sup" ]]; then
                mode_path="$GLOVE_PATH"
            elif [[ "$mode" == "word2vec" || "$mode" == "word2vec-sup" ]]; then
                mode_path="$WORD2VEC_PATH"
            elif [[ "$mode" == "fasttext" || "$mode" == "fasttext-sup" ]]; then
                mode_path="$FASTTEXT_PATH"
            elif [[ "$mode" == "bert" || "$mode" == "bert-sup" ]]; then
                mode_path="$BERT_PATH"
            elif [[ "$mode" == "llama" || "$mode" == "llama-sup" ]]; then
                mode_path="$LLAMA_PATH"
            fi
            run_command $dataset $pickle_path $learner $mode "$mode_path"
        done
    done
done
