
#!/bin/bash

#
# Datasets array
# this is simply for reference
#
dataset_info=(
    "--dataset bbc-news"                        # bbc-news (single label, 5 classes)    
    "--dataset reuters21578"                    # reuters21578 (multi-label, 115 classes) 
    "--dataset imdb"                            # imdb (single-label, 2 classes)     
    "--dataset ohsumed"                         # ohsumed (multi-label, 23 classes) 
    "--dataset arxiv_protoformer"               # arxiv_protoformer (single-label, 10 classes)
    "--dataset arxiv"                           # arxiv (multi-label, 58 classes) 
    "--dataset 20newsgroups"                    # 20newsgroups (single label, 20 classes)
    "--dataset rcv1"                            # RCV1-v2 (multi-label, 101 classes)
)   
#
#


# ----------------------------------------------------------------------------------------------------------------------------
#
# Shell Script Arguments
#
PY="python ../src/ml_classification_test_v5.0.py"

LOG="--logfile ../log/hyperbolic_test.test"

EMB="--embedding-dir ../.vector_cache"

#OPTIMC="--optimc"              # optimize model
OPTIMC=""                      # default params

CONF_MATRIX="--cm"  

DATASET_EMB_COMP="--dataset-emb-comp"

#declare -a datasets=("arxiv" "rcv1" "reuters21578" "20newsgroups" "imdb" "ohsumed" "bbc-news" "arxiv_protoformer")
declare -a datasets=("reuters21578" "20newsgroups")

#declare -a pickle_paths=("../pickles" "../pickles" "../pickles" "../pickles")

#declare -a learners=("svm" "lr" "nb")
declare -a learners=("svm")

#declare -a vtypes=("tfidf" "count")
declare -a vtypes=("tfidf")

#declare -a mixes=("vmode" "solo" "lsa" "dot" "cat-doc" "solo-wce" "dot-wce" "cat-wce" "lsa-wce" "cat-doc-wce")
declare -a mixes=("solo")
declare -a emb_comp_options=("avg" "summary" "weighted")


# Embedding config params
GLOVE="--pretrained glove" 
WORD2VEC="--pretrained word2vec"
FASTTEXT="--pretrained fasttext"
HYPERBOLIC="--pretrained hyperbolic"
BERT="--pretrained bert"
ROBERTA="--pretrained roberta"
DISTILBERT="--pretrained distilbert"
XLNET="--pretrained xlnet"
GPT2="--pretrained gpt2"
LLAMA="--pretrained llama"
DEEPSEEK="--pretrained deepseek"
# ----------------------------------------------------------------------------------------------------------------------------


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
#                    run_command "$dataset" "$learner" "$vtype" "$mix" "$FASTTEXT" "$emb_comp" "$OPTIMC"
#                    run_command "$dataset" "$learner" "$vtype" "$mix" "$GLOVE" "$emb_comp" "$OPTIMC"
#                    run_command "$dataset" "$learner" "$vtype" "$mix" "$WORD2VEC" "$emb_comp" "$OPTIMC"
                    run_command "$dataset" "$learner" "$vtype" "$mix" "$HYPERBOLIC" "$emb_comp" "$OPTIMC"

#                    run_command "$dataset" "$learner" "$vtype" "$mix" "$BERT" "$emb_comp" "$OPTIMC"
#                    run_command "$dataset" "$learner" "$vtype" "$mix" "$ROBERTA" "$emb_comp" $OPTIMC
#                    run_command "$dataset" "$learner" "$vtype" "$mix" "$DISTILBERT" "$emb_comp" $OPTIMC
#                    run_command "$dataset" "$learner" "$vtype" "$mix" "$XLNET" "$emb_comp" $OPTIMC
#                    run_command "$dataset" "$learner" "$vtype" "$mix" "$GPT2" "$emb_comp" $OPTIMC
#                    run_command "$dataset" "$learner" "$vtype" "$mix" "$LLAMA" "$emb_comp" $OPTIMC
#                    run_command "$dataset" "$learner" "$vtype" "$mix" "$DEEPSEEK" "$emb_comp" $OPTIMC
                done
            done
        done
    done
done
