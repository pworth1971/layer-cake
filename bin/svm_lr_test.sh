#!/bin/bash

# Base components
PY="python ../src/svm_baselines.py"
LOG="--log-file ../log/svm_test.out"
dataset="--dataset 20newsgroups"
EMB="--embedding-dir ../.vector_cache"

# Define commands using function calls to ensure variable expansion
function pg_svm_tfidf() {
    $PY $LOG $DATASET $EMB --learner svm --mode tfidf --optimc
}

function pg_svm_glove() {
    $PY $LOG $DATASET $EMB --learner svm --mode glove --optimc
}

function pg_svm_glove_sup() {
    $PY $LOG $DATASET $EMB --learner svm --mode glove-sup --optimc
}

function pg_svm_bert() {
    $PY $LOG $DATASET $EMB --learner svm --mode bert --optimc
}

function pg_svm_bert_sup() {
    $PY $LOG $DATASET $EMB --learner svm --mode bert-sup --optimc
}


#PG_SVM_TFIDF=$PY $LOG $dataset $EMB --learner svm --mode tfidf --optimc 
#PG_SVM_GLOVE="python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir ../.vector_cache --log-file ../log/svm_test.out --learner svm --mode glove --optimc"
#PG_SVM_GLOVE_SUP="python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir ../.vector_cache --log-file ../log/svm_test.out --learner svm --mode glove-sup --optimc"
#PG_SVM_BERT="python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir ../.vector_cache --log-file ../log/svm_test.out --learner svm --mode bert --optimc"
#PG_SVM_BERT_SUP="python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir ../.vector_cache --log-file ../log/svm_test.out --learner svm --mode bert-sup --optimc"


# Array of function calls
commands=(
    pg_svm_tfidf
    pg_svm_glove
    pg_svm_glove_sup
    pg_svm_bert
    pg_svm_bert_sup
)

# Loop through commands, echo and execute each one
for cmd in "${commands[@]}"; do
    echo "$cmd"
    eval "$cmd"
done