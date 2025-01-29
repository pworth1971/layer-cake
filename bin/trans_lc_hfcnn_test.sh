#!/usr/bin/env bash

# Static configurable variables
PROGRAM_NAME="python ../src/trans_layer_cake_v13.5.py"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

MODEL='--net hf.cnn'

#
# TEST/STG settings
#
EPOCHS=37              # TEST
PATIENCE=3             # TEST
#LOG_FILE="--log-file ../log/lc_nn_hf.cnn_test.test"
LOG_FILE="--log-file ../log/lc_nn_trans_test.test.modified"

#
# DEV settings
#
#EPOCHS=19               # DEV
#PATIENCE=2              # DEV
#LOG_FILE="--log-file ../log/lc_nn_hf.cnn_test.dev"

SEED=77

# Datasets array
datasets=(
    "--dataset rcv1"                            # RCV1-v2 (multi-label, 101 classes)
    "--dataset bbc-news"                        # bbc-news (single label, 5 classes)    
    "--dataset reuters21578"                    # reuters21578 (multi-label, 115 classes) 
    "--dataset 20newsgroups"                    # 20newsgroups (single label, 20 classes)
    "--dataset ohsumed"                         # ohsumed (multi-label, 23 classes) 
    "--dataset arxiv_protoformer"               # arxiv_protoformer (single-label, 10 classes)
    "--dataset arxiv"                           # arxiv (multi-label, 58 classes) 
    "--dataset imdb"                            # imdb (single-label, 2 classes)     
 )   

# -------------------------------------------------------------------------------
#
# Pretrained embeddings array (indexed to preserve order)
#
embedding_names=(
    "DEEPSEEK"
    "LLAMA"
    "BERT"
    "DISTILBERT"
    "ROBERTA"
    "XLNET"
    "GPT2"
)

embedding_args=(    
    "--pretrained deepseek"
    "--pretrained llama"
    "--pretrained bert"
    "--pretrained distilbert"
    "--pretrained roberta"
    "--pretrained xlnet"
    "--pretrained gpt2" 
)
#
# --------------------------------------------------------------------------------




# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

dataset="--dataset bbc-news"

channels="--channels 128"


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
    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    #echo

done


# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------



dataset="--dataset reuters21578"

channels="--channels 256"


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
    $PROGRAM_NAME   $MODEL $channels    --tunable    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
    echo


    #
    # TUNABLE model, supervised (cat, add, dot), tunable tce layer        
    #
    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    #echo

done



# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

dataset="--dataset 20newsgroups"

channels="--channels 128"

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
    $PROGRAM_NAME   $MODEL $channels    --tunable    $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE
    echo


    #
    # TUNABLE model, supervised (cat, add, dot), tunable tce layer        
    #
    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    #echo

done




# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

dataset="--dataset ohsumed"

channels="--channels 512"


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
    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    #echo

done



# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

dataset="--dataset arxiv_protoformer"

channels="--channels 256"


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
    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    #echo

done



# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

dataset="--dataset arxiv"

channels="--channels 512"


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
    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    #echo

done



# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------

dataset="--dataset imdb"

channels="--channels 64"


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
    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    #echo

done


# -----------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------



dataset="--dataset rcv1"

channels="--channels 512"

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
    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable   $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode cat --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode add --tunable-tces
    #echo

    #echo "Running: $PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces"
    #echo
    #$PROGRAM_NAME $MODEL $channels   --tunable $dataset $embed_arg --seed $SEED $LOG_FILE --epochs $EPOCHS --patience $PATIENCE --supervised --sup-mode dot --tunable-tces
    #echo

done


