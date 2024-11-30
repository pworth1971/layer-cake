#!/bin/bash

# -----------------------------------------------------------------------------------------------------------------------------------------
# CONFIG INFO
# NB: must be run from /bin directory

PY="python ../src/trans_layer_cake_v3.py"                            # source file
LOG="--log-file ../log/trans_lc_test.test"                           # output log file for metrics

EP="12"                # number of epochs
NUM_RUNS=1

#
# embedding config command arguments
# regular array to enforce ordering
#
declare -a embeddings=(  
    "BERT --pretrained bert"
    "ROBERTA --pretrained roberta"
    "DISTILBERT --pretrained distilbert"
    "XLNET --pretrained xlnet"
    "GPT2 --pretrained gpt2"
    "LLAMA --pretrained llama"
)

#
# dataset config (list of datasets)
#
datasets=(
    "--dataset 20newsgroups"        # 20newsgroups (single label, 20 classes)
    "--dataset reuters21578"        # reuters21578 (multi-label, 115 classes)
    "--dataset bbc-news"            # bbc-news (single label, 5 classes)
    "--dataset ohsumed"             # ohsumed (multi-label, 23 classes)
    "--dataset rcv1"                # RCV1-v2 (multi-label, 101 classes)
)

# -----------------------------------------------------------------------------------------------------------------------------------------


for dataset in "${datasets[@]}"; do
    echo
    echo "------------------------------------------------------------------------------------------------"
    echo "---------- Processing dataset: $dataset ----------"

    for ((run=1; run<=NUM_RUNS; run++)); do
    
        echo "RUN: $run"

        # Base configurations for training without embeddings
        echo "$PY $LOG $dataset --nepochs $EP"
        echo
        $PY $LOG $dataset --nepochs $EP
        echo

        echo "$PY $LOG $dataset --supervised --nepochs $EP"
        echo
        $PY $LOG $dataset --supervised  --nepochs $EP 

        # Loop over ordered embeddings
        for embedding in "${embeddings[@]}"; do

            # Split the embedding name and command
            embed_name=$(echo $embedding | awk '{print $1}')
            embed_cmd=$(echo $embedding | cut -d' ' -f2-)
            
            echo "running $embed_name language model..."
            echo
            
            #
            # Command with embedding configurations
            #
            echo "$PY $LOG $dataset $embed_cmd --nepochs $EP"
            $PY $LOG $dataset $embed_cmd --nepochs $EP
            
            echo "$PY $LOG $dataset $embed_cmd --supervised --nepochs $EP"
            $PY $LOG $dataset $ATTN $embed_cmd --supervised --nepochs $EP
            
        done

    done

done
