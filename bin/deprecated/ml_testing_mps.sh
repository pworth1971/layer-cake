#!/bin/bash

#
# Base components
#
PY="python ../src/ml_classification_test_v3.0.py"
LOG="--logfile ../log/ml_mps_full_opt_test.test"
EMB="--embedding-dir ../.vector_cache"
OPTIMC="--optimc"
CONF_MATRIX="--cm"  
DATASET_EMB_COMP="--dataset-emb-comp"
WCE="--wce"


# Datasets array
dataset_info=(
    "--dataset arxiv_protoformer"               # arxiv_protoformer (single-label, 10 classes)
    "--dataset 20newsgroups"                    # 20newsgroups (single label, 20 classes)
    "--dataset reuters21578"                    # reuters21578 (multi-label, 115 classes) 
    "--dataset imdb"                            # imdb (single-label, 2 classes)     
    "--dataset ohsumed"                         # ohsumed (multi-label, 23 classes) 
    "--dataset arxiv"                           # arxiv (multi-label, 58 classes) 
    "--dataset bbc-news"                        # bbc-news (single label, 5 classes)    
    "--dataset rcv1"                            # RCV1-v2 (multi-label, 101 classes)
 )   

declare -a datasets=("arxiv_protoformer" "arxiv" "imdb" "rcv1" "bbc-news" "20newsgroups" "reuters21578" "ohsumed")
declare -a pickle_paths=("../pickles" "../pickles" "../pickles" "../pickles")

declare -a learners=("svm" "lr" "nb")
declare -a vtypes=("tfidf")

declare -a mixes=("dot" "dot-wce" "solo" "solo-wce" "vmode" "cat-doc" "cat-wce" "cat-doc-wce" "lsa" "lsa-wce")

declare -a embeddings=("fasttext" "glove" "word2vec" "bert" "roberta" "distilbert" "xlnet" "gpt2")
declare -a emb_comp_options=("avg" "summary")


# Embedding config params
GLOVE="--pretrained glove" 
WORD2VEC="--pretrained word2vec"
FASTTEXT="--pretrained fasttext"
BERT="--pretrained bert"
ROBERTA="--pretrained roberta"
DISTILBERT="--pretrained distilbert"
XLNET="--pretrained xlnet"
GPT2="--pretrained gpt2"


# Function to run commands
function run_command() {
    local dataset=$1
    #local pickle_path=$2
    local learner=$2
    local vtype=$3
    local mix=$4
    local embedding_option=$5
    local emb_comp=$6

    local dataset_flag="--dataset ${dataset}"
    #local pickle_flag="--pickle-dir ${pickle_path}"
    
    local cmd="$PY $LOG $dataset_flag --net $learner --vtype $vtype --mix $mix $embedding_option $DATASET_EMB_COMP $emb_comp $OPTIMC"
    local cmd_opt="$PY $LOG $dataset_flag --net $learner --vtype $vtype --mix $mix $embedding_option $DATASET_EMB_COMP $emb_comp $OPTIMC"
    
    # Execute the base command
    echo
    echo
    echo
    echo
    echo "_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________"
    echo "#########################################################################################################################################################################################################################################################################"
    echo "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    echo "Running Default command: $cmd"
    eval $cmd
    echo
    echo
    echo
    #echo
    #echo
    #echo
    #echo "_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________"
    #echo "#########################################################################################################################################################################################################################################################################"
    #echo "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    #echo "Running optimized command: $cmd_opt"
    #eval $cmd_opt
}

# Loop through datasets and run commands
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    #pickle_path=${pickle_paths[$i]}

    for learner in "${learners[@]}"; do
        for vtype in "${vtypes[@]}"; do
            for mix in "${mixes[@]}"; do
                for emb_comp in "${emb_comp_options[@]}"; do
                    run_command "$dataset" "$learner" "$vtype" "$mix" "$GLOVE" "$emb_comp" "$OPTIMC"
                    run_command "$dataset" "$learner" "$vtype" "$mix" "$WORD2VEC" "$emb_comp" "$OPTIMC"
                    run_command "$dataset" "$learner" "$vtype" "$mix" "$FASTTEXT" "$emb_comp" "$OPTIMC"
                    run_command "$dataset" "$learner" "$vtype" "$mix" "$BERT" "$emb_comp" "$OPTIMC"
                    run_command "$dataset" "$learner" "$vtype" "$mix" "$ROBERTA" "$emb_comp" $OPTIMC
                    run_command "$dataset" "$learner" "$vtype" "$mix" "$DISTILBERT" "$emb_comp" $OPTIMC
                    run_command "$dataset" "$learner" "$vtype" "$mix" "$XLNET" "$emb_comp" $OPTIMC
                    run_command "$dataset" "$learner" "$vtype" "$mix" "$GPT2" "$emb_comp" $OPTIMC
                done
            done
        done
    done
done