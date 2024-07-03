#!/bin/bash

# Define commands
PG_SVM_TFIDF="python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir '../.vector_cache' --log-file '../log/svm_test.out' --learner svm --mode tfidf"
PG_LR_TFIDF="python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir '../.vector_cache' --log-file '../log/svm_test.out' --learner lr --mode tfidf"
PG_SVM_GLOVE="python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir ../.vector_cache --log-file ../log/svm_test.out --learner svm --mode glove"
PG_LR_GLOVE="python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir ../.vector_cache --log-file ../log/svm_test.out --learner lr --mode glove"
PG_SVM_GLOVE_SUP="python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir ../.vector_cache --log-file ../log/svm_test.out --learner svm --mode glove-sup"
PG_LR_GLOVE_SUP="python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir ../.vector_cache --log-file ../log/svm_test.out --learner lr --mode glove-sup"

# Array of commands
commands=(
    "$PG_SVM_TFIDF"
    "$PG_LR_TFIDF"
    "$PG_SVM_GLOVE"
    "$PG_LR_GLOVE"
    "$PG_SVM_GLOVE_SUP"
    "$PG_LR_GLOVE_SUP"
)

# Loop through commands, echo and execute each one
for cmd in "${commands[@]}"; do
    echo "$cmd"
    eval "$cmd"
done
