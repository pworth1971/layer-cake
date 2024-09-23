#!/bin/bash

#
# Base components
#
PY="python ../src/ml_classification_test_v2.0.py"
LOG="--logfile ../log/ml_mps_repmodel.test"
EMB="--embedding-dir ../.vector_cache"
OPTIMC="--optimc"
CONF_MATRIX="--cm"  
DATASET_EMB_COMP="--dataset-emb-comp"

#
# Full Arrays of datasets and corresponding pickle paths
#
#declare -a datasets=("bbc-news" "20newsgroups" "reuters21578" "ohsumed" "rcv1")
#declare -a pickle_paths=("../pickles" "../pickles" "../pickles" "../pickels" "../pickels")
#declare -a modes=("tfidf" "cat" "solo" "dot")
#declare -a pretrained_embeddings=("glove" "word2vec" "fasttext" "bert" "llama")
#declare -a learners=("svm" "lr" "nb")

#bb_dataset="--dataset bbc-news --pickle-dir ../pickles"                    # bbc-news (single label, 5 classes)
#20_dataset="--dataset 20newsgroups --pickle-dir ../pickles"                # 20_newsgroups (single label, 20 classes)
#ohm_dataset="--dataset ohsumed --pickle-dir ../pickles"                    # ohsumed (multi-label, 23 classes)
#reut_dataset="--dataset reuters21578 --pickle-dir ../pickles"              # reuters21578 (multi-label, 115 classes)
#rcv_dataset="--dataset rcv1 --pickle-dir ../pickles"                       # RCV1-v2 (multi-label, 101 classes)

declare -a datasets=("bbc-news" "reuters21578" "20newsgroups")
#declare -a datasets=("rcv1")
declare -a pickle_paths=("../pickles" "../pickles" "../pickles")
#declare -a learners=("svm" "lr" "nb")
declare -a learners=("svm")
declare -a vtypes=("tfidf")
#declare -a mixes=("solo" "vmode" "cat" "dot" "lsa")
declare -a mixes=("solo" "dot" "lsa" "vmode")
#declare -a embeddings=("glove" "word2vec", "fasttext" "bert" "roberta" "llama")
declare -a embeddings=("glove" "bert" "roberta" "llama")
declare -a emb_comp_options=("weighted" "avg" "summary")

# Embedding config params

GLOVE="--pretrained glove --glove-path ../.vector_cache/GloVe" 
#GLOVE_SUP="--pretrained glove --glove-path ../.vector_cache/GloVe --supervised" 

WORD2VEC="--pretrained word2vec --word2vec-path ../.vector_cache/Word2Vec"
#WORD2VEC_SUP="--pretrained word2vec --word2vec-path ../.vector_cache/Word2Vec --supervised"

FASTTEXT="--pretrained fasttext --fasttext-path ../.vector_cache/fastText"
#FASTTEXT_SUP="--pretrained fasttext --fasttext-path ../.vector_cache/fastText --supervised"

BERT="--pretrained bert --bert-path ../.vector_cache/BERT"
#BERT_SUP="--pretrained bert --bert-path ../.vector_cache/BERT --supervised"

ROBERTA="--pretrained roberta --roberta-path ../.vector_cache/RoBERTa"
#ROBERTA_SUP="--pretrained roberta --roberta-path ../.vector_cache/RoBERTa --supervised"

LLAMA="--pretrained llama --llama-path ../.vector_cache/LLaMa"
#LLAMA_SUP="--pretrained llama --llama-path ../.vector_cache/LlaMa --supervised"

# Function to run commands
function run_command() {
    local dataset=$1
    local pickle_path=$2
    local learner=$3
    local vtype=$4
    local mix=$5
    local embedding_option=$6
    local emb_comp=$7

    local dataset_flag="--dataset ${dataset}"
    local pickle_flag="--pickle-dir ${pickle_path}"
    local cmd="$PY $LOG $dataset_flag $pickle_flag --learner $learner --vtype $vtype --mix $mix $embedding_option $DATASET_EMB_COMP $emb_comp"


    #local cmd="$PY $LOG $dataset_flag $pickle_flag $EMB --learner $learner --mode $mode $embedding_option $OPTIMC"
    #local cmd="$PY $LOG $dataset_flag $pickle_flag --learner $learner --mode $mode $embedding_option $OPTIMC $CONF_MATRIX"
    local cmd="$PY $LOG $dataset_flag $pickle_flag --learner $learner --vtype $vtype --mix $mix $embedding_option $DATASET_EMB_COMP $emb_comp"
    local cmd_opt="$PY $LOG $dataset_flag $pickle_flag --learner $learner --vtype $vtype --mix $mix $embedding_option $DATASET_EMB_COMP $emb_comp $OPTIMC"

    # Execute the base command
    echo
    echo
    echo
    echo
    echo "_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________"
    echo "#########################################################################################################################################################################################################################################################################"
    echo "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo "exeuting command: $cmd"
    eval $cmd
    echo
    echo
    echo
    #echo "_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________"
    #echo "#########################################################################################################################################################################################################################################################################"
    #echo "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    #echo "Running optimized command: $cmd_opt"
    #eval $cmd_opt
}

# Loop through datasets and run commands
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    pickle_path=${pickle_paths[$i]}

    for learner in "${learners[@]}"; do
        for vtype in "${vtypes[@]}"; do
            for mix in "${mixes[@]}"; do
                for emb_comp in "${emb_comp_options[@]}"; do
                    run_command "$dataset" "$pickle_path" "$learner" "$vtype" "$mix" "$GLOVE" "$emb_comp"
                    #run_command "$dataset" "$pickle_path" "$learner" "$vtype" "$mix" "$WORD2VEC" "$emb_comp"
                    #run_command "$dataset" "$pickle_path" "$learner" "$vtype" "$mix" "$FASTTEXT" "$emb_comp"
                    run_command "$dataset" "$pickle_path" "$learner" "$vtype" "$mix" "$BERT" "$emb_comp"
                    run_command "$dataset" "$pickle_path" "$learner" "$vtype" "$mix" "$ROBERTA" "$emb_comp"
                    run_command "$dataset" "$pickle_path" "$learner" "$vtype" "$mix" "$LLAMA" "$emb_comp"
                done
            done
        done
    done
done
