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
    "--dataset reuters21578"                    # reuters21578 (multi-label, 115 classes) 
 #   "--dataset imdb"                            # imdb (single-label, 2 classes)     
 #   "--dataset ohsumed"                         # ohsumed (multi-label, 23 classes) 
    "--dataset arxiv_protoformer"               # arxiv_protoformer (single-label, 10 classes)
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
    "DISTILBERT"
    "ROBERTA"
    "BERT"
#    "GPT2"
)

embedding_args=(    
#    "--pretrained xlnet"
    "--pretrained distilbert"
    "--pretrained roberta"
    "--pretrained bert"
#    "--pretrained gpt2" 
)






# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

#
# CNN Model Testing
#

dataset="--dataset reuters21578"

MODEL="--net cnn"
channels="--channels 256"

#MODEL="--net lstm"
#lstm_hidden="--hidden 256"

#MODEL="--net attn"
#attn_hidden="--hidden 1024"


for i in "${!embedding_names[@]}"; do
    
    embed_name="${embedding_names[$i]}"
    embed_arg="${embedding_args[$i]}"

    #
    # STATIC model, unsupervised        
    #
    #echo
    #echo "Running: $PROGRAM_NAME $MODEL $channels $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    #$PROGRAM_NAME $MODEL    $channels    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
    #echo

    #
    # TUNABLE model, unsupervised        
    #
    echo
    echo "Running: $PROGRAM_NAME    $MODEL  $channels --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    $PROGRAM_NAME   $MODEL $channels    --tunable    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE $MODEL
    echo


    #
    # TUNABLE model, supervised (cat, add, dot), tunable tce layer        
    #
    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    echo

done




# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

#
# CNN Model Testing
#

dataset="--dataset 20newsgroups"

MODEL="--net cnn"
channels="--channels 128"

#MODEL="--net lstm"
#lstm_hidden="--hidden 256"

#MODEL="--net attn"
#attn_hidden="--hidden 1024"

for i in "${!embedding_names[@]}"; do
    
    embed_name="${embedding_names[$i]}"
    embed_arg="${embedding_args[$i]}"

    #
    # STATIC model, unsupervised        
    #
    #echo
    #echo "Running: $PROGRAM_NAME $MODEL $channels $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    #$PROGRAM_NAME $MODEL    $channels    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
    #echo

    #
    # TUNABLE model, unsupervised        
    #
    echo
    echo "Running: $PROGRAM_NAME    $MODEL  $channels --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    $PROGRAM_NAME   $MODEL $channels    --tunable    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE $MODEL
    echo


    #
    # TUNABLE model, supervised (cat, add, dot), tunable tce layer        
    #
    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    echo

done




# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

dataset="--dataset bbc-news"

MODEL="--net cnn"
channels="--channels 128"

#MODEL="--net lstm"
#lstm_hidden="--hidden 256"

#MODEL="--net attn"
#attn_hidden="--hidden 256"

for i in "${!embedding_names[@]}"; do
    
    embed_name="${embedding_names[$i]}"
    embed_arg="${embedding_args[$i]}"

    #
    # STATIC model, unsupervised        
    #
    #echo
    #echo "Running: $PROGRAM_NAME $MODEL $channels $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    #$PROGRAM_NAME $MODEL    $channels    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
    #echo

    #
    # TUNABLE model, unsupervised        
    #
    echo
    echo "Running: $PROGRAM_NAME    $MODEL  $channels --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    $PROGRAM_NAME   $MODEL $channels    --tunable    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE $MODEL
    echo


    #
    # TUNABLE model, supervised (cat, add, dot), tunable tce layer        
    #
    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    echo

done





# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

dataset="--dataset ohsumed"

MODEL="--net cnn"
channels="--channels 256"

#MODEL="--net lstm"
#lstm_hidden="--hidden 1024"

#MODEL="--net attn"
#attn_hidden="--hidden 1024"

for i in "${!embedding_names[@]}"; do
    
    embed_name="${embedding_names[$i]}"
    embed_arg="${embedding_args[$i]}"

    #
    # STATIC model, unsupervised        
    #
    #echo
    #echo "Running: $PROGRAM_NAME $MODEL $channels $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    #$PROGRAM_NAME $MODEL    $channels    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
    #echo

    #
    # TUNABLE model, unsupervised        
    #
    echo
    echo "Running: $PROGRAM_NAME    $MODEL  $channels --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    $PROGRAM_NAME   $MODEL $channels    --tunable    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE $MODEL
    echo


    #
    # TUNABLE model, supervised (cat, add, dot), tunable tce layer        
    #
    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    echo

done





# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

#
# CNN Model Testing
#

dataset="--dataset arxiv_protoformer"

MODEL="--net cnn"
channels="--channels 256"

#MODEL="--net lstm"
#lstm_hidden="--hidden 256"

#MODEL="--net attn"
#attn_hidden="--hidden 1024"


for i in "${!embedding_names[@]}"; do
    
    embed_name="${embedding_names[$i]}"
    embed_arg="${embedding_args[$i]}"

    #
    # STATIC model, unsupervised        
    #
    #echo
    #echo "Running: $PROGRAM_NAME $MODEL $channels $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    #$PROGRAM_NAME $MODEL    $channels    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
    #echo

    #
    # TUNABLE model, unsupervised        
    #
    echo
    echo "Running: $PROGRAM_NAME    $MODEL  $channels --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    $PROGRAM_NAME   $MODEL $channels    --tunable    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE $MODEL
    echo


    #
    # TUNABLE model, supervised (cat, add, dot), tunable tce layer        
    #
    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    echo

done





# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

#
# CNN Model Testing
#

dataset="--dataset arxiv"

MODEL="--net cnn"
channels="--channels 256"

#MODEL="--net lstm"
#lstm_hidden="--hidden 256"

#MODEL="--net attn"
#attn_hidden="--hidden 1024"


for i in "${!embedding_names[@]}"; do
    
    embed_name="${embedding_names[$i]}"
    embed_arg="${embedding_args[$i]}"

    #
    # STATIC model, unsupervised        
    #
    #echo
    #echo "Running: $PROGRAM_NAME $MODEL $channels $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    #$PROGRAM_NAME $MODEL    $channels    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
    #echo

    #
    # TUNABLE model, unsupervised        
    #
    echo
    echo "Running: $PROGRAM_NAME    $MODEL  $channels --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    $PROGRAM_NAME   $MODEL $channels    --tunable    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE $MODEL
    echo


    #
    # TUNABLE model, supervised (cat, add, dot), tunable tce layer        
    #
    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    echo

done





# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

#
# CNN Model Testing
#

dataset="--dataset imdb"

MODEL="--net cnn"
channels="--channels 128"

#MODEL="--net lstm"
#lstm_hidden="--hidden 256"

#MODEL="--net attn"
#attn_hidden="--hidden 256"


for i in "${!embedding_names[@]}"; do
    
    embed_name="${embedding_names[$i]}"
    embed_arg="${embedding_args[$i]}"

    #
    # STATIC model, unsupervised        
    #
    #echo
    #echo "Running: $PROGRAM_NAME $MODEL $channels $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    #$PROGRAM_NAME $MODEL    $channels    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
    #echo

    #
    # TUNABLE model, unsupervised        
    #
    echo
    echo "Running: $PROGRAM_NAME    $MODEL  $channels --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    $PROGRAM_NAME   $MODEL $channels    --tunable    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE $MODEL
    echo


    #
    # TUNABLE model, supervised (cat, add, dot), tunable tce layer        
    #
    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    echo

done




# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------






# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

#
# CNN Model Testing
#

dataset="--dataset rcv1"

MODEL="--net cnn"
channels="--channels 512"

#MODEL="--net lstm"
#lstm_hidden="--hidden 1024"

#MODEL="--net attn"
#attn_hidden="--hidden 1024"


for i in "${!embedding_names[@]}"; do
    
    embed_name="${embedding_names[$i]}"
    embed_arg="${embedding_args[$i]}"

    #
    # STATIC model, unsupervised        
    #
    #echo
    #echo "Running: $PROGRAM_NAME $MODEL $channels $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    #$PROGRAM_NAME $MODEL    $channels    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
    #echo

    #
    # TUNABLE model, unsupervised        
    #
    echo
    echo "Running: $PROGRAM_NAME    $MODEL  $channels --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE"
    $PROGRAM_NAME   $MODEL $channels    --tunable    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE $MODEL
    echo


    #
    # TUNABLE model, supervised (cat, add, dot), tunable tce layer        
    #
    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    echo
    $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    echo

done




# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------
