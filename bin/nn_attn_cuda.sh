#!/bin/bash

# -----------------------------------------------------------------------------------------------------------------------------------------
# CONFIG INFO
# NB: must be run from /bin directory

# Set the CUDA device for all processes
export CUDA_VISIBLE_DEVICES=0                   # Change to your specific GPU ID as needed


# supported networks, drop probability included
#CNN="--net cnn --dropprob .2"
#LSTM="--net lstm --dropprob .2"
#ATTN="--net attn --dropprob .2"

PY="python ../src/layer_cake.py"                                    # source file
LOG="--log-file ../log/nn_attn_cuda.test"                       # output log file for metrics

#CNN="--net cnn"
#LSTM="--net lstm"
ATTN="--net attn"

EP="200"                # number of epochs
NUM_RUNS=1

# embedding config
declare -A embeddings
embeddings=(
    ["GLOVE"]="--pretrained glove --glove-path ../.vector_cache/GloVe"
    ["WORD2VEC"]="--pretrained word2vec --word2vec-path ../.vector_cache/Word2Vec"
    ["FASTTEXT"]="--pretrained fasttext --fasttext-path ../.vector_cache/fastText"
)


# dataset config
#ng_dataset="--dataset 20newsgroups --pickle-dir ../pickles"                     # 20_newsgroups (single label, 20 classes)
#ohm_dataset="--dataset ohsumed --pickle-dir ../pickles"                         # ohsumed (multi-label, 23 classes)
#reut_dataset="--dataset reuters21578 --pickle-dir ../pickles"                   # reuters21578 (multi-label, 115 classes)
#rcv_dataset="--dataset rcv1 --pickle-dir ../pickles"                            # RCV1-v2 (multi-label, 101 classes)
 
#dataset="--dataset rcv1 --pickle-dir ../pickles"           
#dataset="--dataset bbc-news --pickle-dir ../pickles"                        
# dataset config (list of datasets)

datasets=(
    "--dataset      reuters21578    --pickle-dir ../pickles"                 # reuters21578 (multi-label, 115 classes)
    "--dataset      bbc-news        --pickle-dir ../pickles"                 # bbc-news (single label, 5 classes)
    "--dataset      ohsumed         --pickle-dir ../pickles"                 # ohsumed (multi-label, 23 classes)
    "--dataset      rcv1            --pickle-dir ../pickles"                 # RCV1-v2 (multi-label, 101 classes)
    "--dataset      20newsgroups    --pickle-dir ../pickles"                 # 20newsgroups (single label, 20 classes)
)
# -----------------------------------------------------------------------------------------------------------------------------------------


for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    for ((run=1; run<=NUM_RUNS; run++)); do
        
        echo
        echo "Processing run: $run"
        echo

        $PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256    --seed $run    --nepochs $EP
        $PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256    --supervised   --seed $run    --nepochs $EP 

        for embed_name in "${!embeddings[@]}"; do
            embed=${embeddings[$embed_name]}
            echo
            echo "----- embedding: $embed_name ... -----"

            # Base configuration
            $PY $LOG $dataset $ATTN --hidden 256 $embed --seed $run --nepochs $EP
            $PY $LOG $dataset $ATTN --hidden 256 $embed --tunable --seed $run --nepochs $EP
            $PY $LOG $dataset $ATTN --learnable 20 --hidden 256 $embed --tunable --seed $run --droptype learn --nepochs $EP
            $PY $LOG $dataset $ATTN --hidden 2048 $embed --supervised --seed $run --nepochs $EP
            $PY $LOG $dataset $ATTN --hidden 1024 $embed --supervised --tunable --seed $run --nepochs $EP
            $PY $LOG $dataset $ATTN --learnable 20 --hidden 256 $embed --supervised --tunable --seed $run --droptype learn --nepochs $EP
        done

    done
done