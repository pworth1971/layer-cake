#!/usr/bin/env bash

# Static configurable variables
PROGRAM_NAME="python ../src/trans_layer_cake_v12.0.py"

# Network types
network_types=(
    "linear"
    "cnn"
    "attn"
    "lstm"
    "hf.sc"
)

MODEL='--net cnn'

#
# TEST/STG settings
#
#EPOCHS=37              # TEST
#PATIENCE=3             # TEST
#LOG_FILE="--log-file ../log/lc_nn_trans_test.test"

#
# DEV settings
#
EPOCHS=14               # DEV
PATIENCE=2              # DEV
LOG_FILE="--log-file ../log/lc_nn_trans_test.dev"

SEED=47


# Datasets array
datasets=(
    "--dataset imdb"                            # imdb (single-label, 2 classes)    
    "--dataset arxiv"                           # arxiv (multi-label, 58 classes)
    "--dataset arxiv_protoformer"               # arxiv_protoformer (single-label, 10 classes)
    "--dataset ohsumed"                         # ohsumed (multi-label, 23 classes)    
    "--dataset 20newsgroups"                    # 20newsgroups (single label, 20 classes)
#    "--dataset rcv1"                            # RCV1-v2 (multi-label, 101 classes)
    "--dataset bbc-news"                        # bbc-news (single label, 5 classes)    
    "--dataset reuters21578"                    # reuters21578 (multi-label, 115 classes)
)   

# -------------------------------------------------------------------------------
#
# Pretrained embeddings array (indexed to preserve order)
# NB: issues with Albert so leaving out. LlaMa has not been tested
#
embedding_names=(
    "BERT"
    "ROBERTA"
    "DISTILBERT"
    #"XLNET"
    #"GPT2"
)

embedding_args=(    
    "--pretrained bert"
    "--pretrained roberta"
    "--pretrained distilbert"
    #"--pretrained xlnet"
    #"--pretrained gpt2" 
)

# ------------------------------------------------------------------------------

# Iterate through datasets and embeddings
for dataset in "${datasets[@]}"; do
    
    for i in "${!embedding_names[@]}"; do
    
        embed_name="${embedding_names[$i]}"
        embed_arg="${embedding_args[$i]}"
        
        echo
        echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE $MODEL"
        $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE $MODEL
        echo

        #echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat $MODEL"
        #echo
        #$PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat $MODEL
        #echo

        #echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add $MODEL"
        #echo
        #$PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add $MODEL
        #echo

        #echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot $MODEL"
        #echo
        #$PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot $MODEL
        #echo

        #echo
        #echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED --log_file $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --tunable $MODEL"
        #$PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --tunable $MODEL
        #echo

        #echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --tunable $MODEL"
        #echo
        #$PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised tunable $MODEL
        #echo

    done
done