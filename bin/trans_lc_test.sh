#!/usr/bin/env bash

# Static configurable variables
PROGRAM_NAME="python ../src/trans_layer_cake_v11.2.py"

#
# TEST/STG settings
#
EPOCHS=37              # TEST
PATIENCE=3             # TEST
LOG_FILE="--log-file ../log/lc_nn_trans_test.test"

#
# DEV settings
#
#EPOCHS=14               # DEV
#PATIENCE=2              # DEV
#LOG_FILE="--log-file ../log/lc_nn_trans_test.dev"

SEED=47


# Datasets array
datasets=(
    "--dataset 20newsgroups"                    # 20newsgroups (single label, 20 classes)
    "--dataset imdb"                            # imdb (single-label, 2 classes)    
    "--dataset arxiv"                           # arxiv (multi-label, 58 classes)
    "--dataset arxiv_protoformer"               # arxiv_protoformer (single-label, 10 classes)
    "--dataset reuters21578"                    # reuters21578 (multi-label, 115 classes)
    "--dataset bbc-news"                        # bbc-news (single label, 5 classes)    
    "--dataset ohsumed"                         # ohsumed (multi-label, 23 classes)    
    "--dataset rcv1"                            # RCV1-v2 (multi-label, 101 classes)
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
    "XLNET"
    "GPT2"
)

embedding_args=(
    "--pretrained bert"
    "--pretrained roberta"
    "--pretrained distilbert"
    "--pretrained xlnet"
    "--pretrained gpt2"
)
# ------------------------------------------------------------------------------

# Iterate through datasets and embeddings
for dataset in "${datasets[@]}"; do
    
    for i in "${!embedding_names[@]}"; do
    
        embed_name="${embedding_names[$i]}"
        embed_arg="${embedding_args[$i]}"
        
        echo
        echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
        $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
        echo

        echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat"
        echo
        $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat
        echo

        echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add"
        echo
        $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add
        echo

        echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot"
        echo
        $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot
        echo


        #echo
        #echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED --log_file $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --tunable"
        #$PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --tunable
        #echo


        #echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --tunable"
        #echo
        #$PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised tunable
        #echo

    done
done
