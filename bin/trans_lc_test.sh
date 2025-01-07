#!/usr/bin/env bash

# Static configurable variables
PROGRAM_NAME="python ../src/trans_layer_cake_v13.3.py"

# Network types
network_types=(
    "linear"
    "cnn"
    "attn"
    "lstm"
    "hf.sc"
)

#MODEL='--net hf.sc'
#MODEL='--net linear'
MODEL='--net cnn'
#MODEL='--net attn'
#MODEL='--net lstm'

#
# TEST/STG settings
#
#EPOCHS=37              # TEST
#PATIENCE=3             # TEST
#LOG_FILE="--log-file ../log/lc_nn_trans_test.test"

#
# DEV settings
#
EPOCHS=16               # DEV
PATIENCE=2              # DEV
LOG_FILE="--log-file ../log/lc_nn_trans_test.dev"

SEED=49

# Datasets array
datasets=(
    "--dataset 20newsgroups"                    # 20newsgroups (single label, 20 classes)
 #   "--dataset reuters21578"                    # reuters21578 (multi-label, 115 classes) 
 #   "--dataset imdb"                            # imdb (single-label, 2 classes)     
 #   "--dataset ohsumed"                         # ohsumed (multi-label, 23 classes) 
 #   "--dataset arxiv_protoformer"               # arxiv_protoformer (single-label, 10 classes)
    "--dataset arxiv"                           # arxiv (multi-label, 58 classes) 
 #   "--dataset bbc-news"                        # bbc-news (single label, 5 classes)    
 #   "--dataset rcv1"                            # RCV1-v2 (multi-label, 101 classes)
 )   

# -------------------------------------------------------------------------------
#
# Pretrained embeddings array (indexed to preserve order)
# NB: issues with Albert so leaving out. LlaMa has not been tested
#
embedding_names=(
#    "XLNET"
#    "DISTILBERT"
#    "ROBERTA"
#    "BERT"
    "GPT2"
)

embedding_args=(    
#    "--pretrained xlnet"
#    "--pretrained distilbert"
#    "--pretrained roberta"
#    "--pretrained bert"
    "--pretrained gpt2" 
)

# ------------------------------------------------------------------------------


TUNING_ARGS='--tunable-tces    -tunable pretrained'



# Iterate through datasets and embeddings
for dataset in "${datasets[@]}"; do
    
    for i in "${!embedding_names[@]}"; do
    
        embed_name="${embedding_names[$i]}"
        embed_arg="${embedding_args[$i]}"

        #
        # STATIC model, unsupervised        
        #
        echo
        echo "Running: $PROGRAM_NAME $MODEL $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
        $PROGRAM_NAME $MODEL    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
        echo

        #
        # TUNABLE model, unsupervised        
        #
        echo
        echo "Running: $PROGRAM_NAME    $MODEL --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
        $PROGRAM_NAME   $MODEL --tunable    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE $MODEL
        echo


        #
        # STATIC model, supervised (cat, add, dot)        
        #
        #echo "Running: $PROGRAM_NAME    $MODEL $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat"
        #echo
        #$PROGRAM_NAME   $MODEL $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat
        #echo

        #echo "Running: $PROGRAM_NAME  $MODEL   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add"
        #echo
        #$PROGRAM_NAME   $MODEL $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add
        #echo

        #echo "Running: $PROGRAM_NAME    $MODEL $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot"
        #echo
        #$PROGRAM_NAME   $MODEL $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot
        #echo


        #
        # STATIC model, supervised (cat, add, dot), tunable tce layer        
        #
        #echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces $MODEL"
        #echo
        #$PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces   $MODEL
        #echo

        #echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces   $MODEL"
        #echo
        #$PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces  $MODEL
        #echo

        #echo "Running: $PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces    $MODEL"
        #echo
        #$PROGRAM_NAME $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces   $MODEL
        #echo


        #
        # TUNABLE model, supervised (cat, add, dot), tunable tce layer        
        #
        echo "Running: $PROGRAM_NAME $MODEL  --tunable  $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
        echo
        $PROGRAM_NAME $MODEL  --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
        echo

        echo "Running: $PROGRAM_NAME $MODEL  --tunable  $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
        echo
        $PROGRAM_NAME $MODEL  --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
        echo

        echo "Running: $PROGRAM_NAME $MODEL  --tunable  $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
        echo
        $PROGRAM_NAME $MODEL  --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
        echo
        

    done
done
