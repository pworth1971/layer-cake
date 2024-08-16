#!/bin/bash

#
# Base components
#
PY="python ../src/ml_class_baselines.py"
LOG="--log-file ../log/ml_ohsumed.test"
EMB="--embedding-dir ../.vector_cache"
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
declare -a datasets=("ohsumed")
#declare -a pickle_paths=("../pickles" "../pickles" "../pickles" "../pickles")
declare -a pickle_paths=("../pickles")
#declare -a models=("svm" "lr" "nb")
declare -a models=("nb")
#declare -a modes=("tfidf" "count")
declare -a modes=("tfidf")
#declare -a embeddings=("glove" "word2vec" "fasttext" "bert" "llama")


# Embedding config params
GLOVE="--pretrained glove --glove-path ../.vector_cache" 
GLOVE_SUP="--pretrained glove --glove-path ../.vector_cache --supervised" 

WORD2VEC="--pretrained word2vec --word2vec-path ../.vector_cache/GoogleNews-vectors-negative300.bin"
WORD2VEC_SUP="--pretrained word2vec --word2vec-path ../.vector_cache/GoogleNews-vectors-negative300.bin --supervised"

FASTTEXT="--pretrained fasttext --fasttext-path ../.vector_cache/crawl-300d-2M.vec"
FASTTEXT_SUP="--pretrained fasttext --fasttext-path ../.vector_cache/crawl-300d-2M.vec --supervised"

BERT="--pretrained bert --bert-path ../.vector_cache"
BERT_SUP="--pretrained bert --bert-path ../.vector_cache --supervised"

LLAMA="--pretrained llama --llama-path ../.vector_cache"
LLAMA_SUP="--pretrained llama --llama-path ../.vector_cache --supervised"


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
    echo "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo "Running command: $cmd"
    eval $cmd
}

# Loop through datasets and run commands
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    pickle_path=${pickle_paths[$i]}

    for learner in "${models[@]}"; do
        for mode in "${modes[@]}"; do

            # run without pretrainmed embeddings
            #run_command $dataset $pickle_path $learner $mode
            
            # Run the command for each embedding type
            run_command $dataset $pickle_path $learner $mode "$GLOVE"
            run_command $dataset $pickle_path $learner $mode "$GLOVE_SUP"

            run_command $dataset $pickle_path $learner $mode "$WORD2VEC"
            run_command $dataset $pickle_path $learner $mode "$WORD2VEC_SUP"
            
            run_command $dataset $pickle_path $learner $mode "$FASTTEXT"
            run_command $dataset $pickle_path $learner $mode "$FASTTEXT_SUP"
            
            run_command $dataset $pickle_path $learner $mode "$BERT"
            run_command $dataset $pickle_path $learner $mode "$BERT_SUP"

            run_command $dataset $pickle_path $learner $mode "$LLAMA"
            run_command $dataset $pickle_path $learner $mode "$LLAMA_SUP"
        done
    done
done
