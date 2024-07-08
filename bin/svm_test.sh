#!/bin/bash

# Base components
PY="python ../src/svm_baselines.py"
LOG="--log-file ../log/svm_test.test"
DATASET="--dataset 20newsgroups"
PICK_PATH="--pickle-dir ../pickles/20newsgroups.pickle"
EMB="--embedding-dir ../.vector_cache"

GLOVE_PATH="--glove-path ../.vector_cache" 
WORD2VEC_PATH="--word2vec-path ../.vector_cache/GoogleNews-vectors-negative300.bin"
FASTTEXT_PATH="--fasttext-path ../.vector_cache/crawl-300d-2M.vec"
BERT_PATH="--bert-path ../.vector_cache"


# Define commands using function calls to ensure variable expansion and direct output
function pg_svm_tfidf() {
    echo
    local cmd="$PY $LOG $DATASET $PICK_PATH $EMB --learner svm --mode tfidf --optimc"
    echo $cmd
    eval $cmd
}

function pg_svm_glove() {
    echo
    local cmd="$PY $LOG $DATASET $PICK_PATH $EMB --learner svm --mode glove $GLOVE_PATH --optimc"
    echo $cmd
    eval $cmd
}

function pg_svm_glove_sup() {
    echo
    local cmd="$PY $LOG $DATASET $PICK_PATH $EMB --learner svm --mode glove-sup $GLOVE_PATH --optimc"
    echo $cmd
    eval $cmd
}

function pg_svm_word2vec() {
    echo
    echo
    echo
    local cmd="$PY $LOG $DATASET $PICK_PATH $EMB --learner svm --mode word2vec $WORD2VEC_PATH --optimc"
    echo $cmd
    eval $cmd
}

function pg_svm_word2vec_sup() {
    echo
    echo
    echo
    local cmd="$PY $LOG $DATASET $PICK_PATH $EMB --learner svm --mode word2vec-sup $WORD2VEC_PATH --optimc"
    echo $cmd
    eval $cmd
}

function pg_svm_fasttext() {
    echo
    echo
    echo
    local cmd="$PY $LOG $DATASET $PICK_PATH $EMB --learner svm --mode fasttext $FASTTEXT_PATH --optimc"
    echo $cmd
    eval $cmd
}

function pg_svm_fasttext_sup() {
    echo
    echo
    echo
    local cmd="$PY $LOG $DATASET $PICK_PATH $EMB --learner svm --mode fasttext-sup $FASTTEXT_PATH --optimc"
    echo $cmd
    eval $cmd
}

function pg_svm_bert() {
    echo
    local cmd="$PY $LOG $DATASET $PICK_PATH $EMB --learner svm --mode bert $BERT_PATH --optimc"
    echo $cmd
    eval $cmd
}

function pg_svm_bert_sup() {
    echo
    echo
    echo
    local cmd="$PY $LOG $DATASET $PICK_PATH $EMB --learner svm --mode bert-sup $BERT_PATH --optimc"
    echo $cmd
    eval $cmd
}


# Array of function calls
commands=(
    pg_svm_tfidf
    pg_svm_glove
    pg_svm_glove_sup
    pg_svm_word2vec
    pg_svm_word2vec_sup
    pg_svm_fasttext
    pg_svm_fasttext_sup
    pg_svm_bert
    pg_svm_bert_sup
)

# Loop through commands, echo and execute each one
for cmd in "${commands[@]}"; do
    $cmd
done