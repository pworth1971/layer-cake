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

EP="35"                # number of epochs
NUM_RUNS=1

#
# embedding config command arguments
# regular array to enforce ordering
#
#"GLOVE --pretrained glove --glove-path ../.vector_cache/GloVe"
#"WORD2VEC --pretrained word2vec --word2vec-path ../.vector_cache/Word2Vec"
#"FASTTEXT --pretrained fasttext --fasttext-path ../.vector_cache/fastText"
#"LLAMA --pretrained llama --llama-path ../.vector_cache/LLaMa"
#
declare -a embeddings=(  
    "BERT --pretrained bert --bert-path ../.vector_cache/BERT"
    "ROBERTA --pretrained roberta --roberta-path ../.vector_cache/RoBERTa"
    "XLNET --pretrained xlnet --xlnet-path ../.vector_cache/XLNet"
    "GPT2 --pretrained gpt2 --gpt2-path ../.vector_cache/GPT2"
)

#
# dataset config (list of datasets)
#
datasets=(
    "--dataset      bbc-news        --pickle-dir ../pickles"                 # bbc-news (single label, 5 classes)
    "--dataset      reuters21578    --pickle-dir ../pickles"                 # reuters21578 (multi-label, 115 classes)
    "--dataset      20newsgroups    --pickle-dir ../pickles"                 # 20newsgroups (single label, 20 classes)
    "--dataset      ohsumed         --pickle-dir ../pickles"                 # ohsumed (multi-label, 23 classes)
    "--dataset      rcv1            --pickle-dir ../pickles"                 # RCV1-v2 (multi-label, 101 classes)
)

# -----------------------------------------------------------------------------------------------------------------------------------------


for dataset in "${datasets[@]}"; do
    echo
    echo "-----------------------------------------------------------------------------------------------"
    echo "---------- Processing dataset: $dataset ----------"

    for ((run=1; run<=NUM_RUNS; run++)); do
    
        echo "RUN: $run"

        # Base configurations for training without embeddings
        echo "$PY $LOG $dataset $ATTN --learnable 200 --hidden 256 --seed $run --nepochs $EP"
        echo
        $PY $LOG $dataset   $ATTN   --learnable 200 --hidden 256    --seed $run --nepochs $EP
        echo

        echo "$PY $LOG $dataset $ATTN --learnable 200 --hidden 256 --supervised --seed $run --nepochs $EP"
        echo
        $PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256    --supervised   --seed $run    --nepochs $EP 

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
            echo "$PY $LOG $dataset $ATTN --hidden 256 $embed_cmd --seed $run --nepochs $EP"
            $PY $LOG $dataset $ATTN --hidden 256 $embed_cmd --seed $run --nepochs $EP
            
            echo "$PY $LOG $dataset $ATTN --hidden 256 $embed_cmd --tunable --seed $run --nepochs $EP"
            $PY $LOG $dataset $ATTN --hidden 256 $embed_cmd --tunable --seed $run --nepochs $EP
            
            echo "$PY $LOG $dataset $ATTN --learnable 20 --hidden 256 $embed_cmd --tunable --seed $run --droptype learn --nepochs $EP"
            $PY $LOG $dataset $ATTN --learnable 20 --hidden 256 $embed_cmd --tunable --seed $run --droptype learn --nepochs $EP
            
            echo "$PY $LOG $dataset $ATTN --hidden 2048 $embed_cmd --supervised --seed $run --nepochs $EP"
            $PY $LOG $dataset $ATTN --hidden 2048 $embed_cmd --supervised --seed $run --nepochs $EP
            
            echo "$PY $LOG $dataset $ATTN --hidden 1024 $embed_cmd --supervised --tunable --seed $run --nepochs $EP"
            $PY $LOG $dataset $ATTN --hidden 1024 $embed_cmd --supervised --tunable --seed $run --nepochs $EP
            
            echo "$PY $LOG $dataset $ATTN --learnable 20 --hidden 256 $embed_cmd --supervised --tunable --seed $run --droptype learn --nepochs $EP"
            $PY $LOG $dataset $ATTN --learnable 20 --hidden 256 $embed_cmd --supervised --tunable --seed $run --droptype learn --nepochs $EP
        done

    done
done
