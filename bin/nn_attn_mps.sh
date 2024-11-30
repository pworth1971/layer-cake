#!/bin/bash

# -----------------------------------------------------------------------------------------------------------------------------------------
# CONFIG INFO
# NB: must be run from /bin directory


PY="python ../src/layer_cake.py"                                        # source file
LOG="--log-file ../log/nn_attn_mps.test"                                # output log file for metrics

CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"

# supported networks, drop probability included
#CNN="--net cnn --dropprob .2"
#LSTM="--net lstm --dropprob .2"
#ATTN="--net attn --dropprob .2"

EP="75"                # number of epochs
NUM_RUNS=1


# Embedding config command arguments
embeddings=(
    "--pretrained glove --glove-path ../.vector_cache/GloVe"
    "--pretrained word2vec --word2vec-path ../.vector_cache/Word2Vec"
    "--pretrained bert --bert-path ../.vector_cache/BERT"
    "--pretrained roberta --roberta-path ../.vector_cache/RoBERTa"
    "--pretrained distilbert --distilbert-path ../.vector_cache/DistilBERT"
)

# Dataset config (list of datasets)
datasets=(
    "--dataset 20newsgroups"
    "--dataset reuters21578"
    "--dataset bbc-news"
    "--dataset ohsumed"
)

# -----------------------------------------------------------------------------------------------------------------------------------------

set -x  # Enable debugging

for dataset in "${datasets[@]}"; do
    echo
    echo "------------------------------------------------------------------------------------------------"
    echo "---------- Processing dataset: $dataset ----------"

    for ((run=1; run<=NUM_RUNS; run++)); do
        echo "RUN: $run"

        # Base configurations for training without embeddings
        #echo "$PY $LOG $dataset $ATTN --learnable 200 --hidden 256 --seed $run --nepochs $EP"
        #echo
        #$PY $LOG $dataset   $ATTN   --learnable 200 --hidden 256    --seed $run --nepochs $EP
        #echo

        #echo "$PY $LOG $dataset $ATTN --learnable 200 --hidden 256 --supervised --seed $run --nepochs $EP"
        #echo
        #$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256    --supervised   --seed $run    --nepochs $EP 

        for embedding in "${embeddings[@]}"; do
            echo
            echo "-------- running language model via: $embedding --------"
            echo

            #
            # Command with embedding configurations
            #
            echo "$PY $LOG $dataset $ATTN --hidden 256 $embedding --seed $run --nepochs $EP"
            $PY $LOG $dataset $ATTN --hidden 256 $embedding --seed $run --nepochs $EP
            echo
            
            #echo "$PY $LOG $dataset $ATTN --hidden 256 $embedding --tunable --seed $run --nepochs $EP"
            #$PY $LOG $dataset $ATTN --hidden 256 $embedding --tunable --seed $run --nepochs $EP
            #echo

            #echo "$PY $LOG $dataset $ATTN --learnable 20 --hidden 256 $embedding --tunable --seed $run --droptype learn --nepochs $EP"
            #$PY $LOG $dataset $ATTN --learnable 20 --hidden 256 $embedding --tunable --seed $run --droptype learn --nepochs $EP
            #echo

            echo "$PY $LOG $dataset $ATTN --hidden 2048 $embedding --supervised --seed $run --nepochs $EP"
            $PY $LOG $dataset $ATTN --hidden 2048 $embedding --supervised --seed $run --nepochs $EP
            echo

            #echo "$PY $LOG $dataset $ATTN --hidden 1024 $embedding --supervised --tunable --seed $run --nepochs $EP"
            #$PY $LOG $dataset $ATTN --hidden 1024 $embedding --supervised --tunable --seed $run --nepochs $EP
            #echo

            #echo "$PY $LOG $dataset $ATTN --learnable 20 --hidden 256 $embedding --supervised --tunable --seed $run --droptype learn --nepochs $EP"
            #$PY $LOG $dataset $ATTN --learnable 20 --hidden 256 $embedding --supervised --tunable --seed $run --droptype learn --nepochs $EP
            #echo
        done
    
    done

done
