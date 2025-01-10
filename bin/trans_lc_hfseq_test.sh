#!/usr/bin/env bash

#
# Static configurable variables
#
PROGRAM_NAME="python ../src/trans_layer_cake_v13.5.py   --net hf.sc"

#
# DEV settings
#
#EPOCHS=16               # DEV
#PATIENCE=2              # DEV
#LOG_FILE="--log-file ../log/lc_nn_hfseq_trans_test.dev"


#
# TEST/STG settings
#
EPOCHS=37              # TEST
PATIENCE=3             # TEST
LOG_FILE="--log-file ../log/lc_nn_hfseq_trans_test.test"


SEED=77


# -------------------------------------------------------------------------------
#
# Pretrained embeddings array (indexed to preserve order)
# NB: issues with Albert so leaving out. LlaMa has not been tested
#
embedding_names=(
    "BERT"
    "XLNET"
    "DISTILBERT"
    "ROBERTA"
    "GPT2"
)

embedding_args=(    
    "--pretrained bert"
    "--pretrained xlnet"
    "--pretrained distilbert"
    "--pretrained roberta"
    "--pretrained gpt2" 
)

#
# Datasets array
#
datasets=(
    "--dataset 20newsgroups"                    # 20newsgroups (single label, 20 classes)
    "--dataset reuters21578"                    # reuters21578 (multi-label, 115 classes) 
    "--dataset imdb"                            # imdb (single-label, 2 classes)     
    "--dataset ohsumed"                         # ohsumed (multi-label, 23 classes) 
    "--dataset arxiv_protoformer"               # arxiv_protoformer (single-label, 10 classes)
    "--dataset arxiv"                           # arxiv (multi-label, 58 classes) 
    "--dataset bbc-news"                        # bbc-news (single label, 5 classes)   
    "--dataset rcv1"                            # RCV1-v2 (multi-label, 101 classes)
 )   


for dataset in "${datasets[@]}"; do

    for i in "${!embedding_names[@]}"; do

        embed_name="${embedding_names[$i]}"
        embed_arg="${embedding_args[$i]}"

        #
        # TUNABLE model, unsupervised    
        #
        echo
        echo "Running: $PROGRAM_NAME $dataset --tunable $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
        $PROGRAM_NAME $dataset --tunable $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
        echo


        #
        # TUNABLE model, supervised (cat), tunable tce layer        
        #
        echo
        echo "Running: $PROGRAM_NAME $dataset --tunable $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
        $PROGRAM_NAME $dataset --tunable $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
        echo

        #
        # TUNABLE model, supervised (add), tunable tce layer        
        #
        echo
        echo "Running: $PROGRAM_NAME $dataset --tunable $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
        $PROGRAM_NAME $dataset --tunable $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
        echo


    done
done
