#!/bin/bash

# -----------------------------------------------------------------------------------------------------------------------------------------
# CONFIG INFO
# NB: must be run from /bin directory


# supported networks, drop probability included
#CNN="--net cnn --dropprob .2"
#LSTM="--net lstm --dropprob .2"
#ATTN="--net attn --dropprob .2"

PY="python ../src/layer_cake.py"                                # source file
LOG="--log-file ../log/nn_cnn_mps2.test"                        # output log file for metrics

CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"

EP="65"                # number of epochs
NUM_RUNS=1

# embedding config
declare -a embeddings=(
    ["GLOVE"]="--pretrained glove --glove-path ../.vector_cache/GloVe"
    ["WORD2VEC"]="--pretrained word2vec --word2vec-path ../.vector_cache/Word2Vec"
    #["FASTTEXT"]="--pretrained fasttext --fasttext-path ../.vector_cache/fastText"
    #["BERT"]="--pretrained bert --bert-path ../.vector_cache/BERT"
    #["ROBERTA"]="--pretrained roberta --roberta-path ../.vector_cache/RoBERTa"
    #["XLNET"]="--pretrained xlnet --xlnet-path ../.vector_cache/XLNet"
    #["GPT2"]="--pretrained gpt2 --gpt2-path ../.vector_cache/GPT2"
    #["LLAMA"]="--pretrained llama --llama-path ../.vector_cache/LLaMa"
)

#
# dataset config (list of datasets)
#
datasets=(
    "--dataset reuters21578 --pickle-dir ../pickles"    # reuters21578 (multi-label, 115 classes)
    "--dataset 20newsgroups --pickle-dir ../pickles"    # 20newsgroups (single label, 20 classes)
    "--dataset bbc-news --pickle-dir ../pickles"        # bbc-news (single label, 5 classes)
    "--dataset ohsumed --pickle-dir ../pickles"         # ohsumed (multi-label, 23 classes)
    "--dataset rcv1 --pickle-dir ../pickles"            # RCV1-v2 (multi-label, 101 classes)
)
# -----------------------------------------------------------------------------------------------------------------------------------------


for dataset in "${datasets[@]}"; do
    echo
    echo "------------------------------------------------------------------------------------------------"
    echo "---------- Processing dataset: $dataset ----------"

    for ((run=1; run<=NUM_RUNS; run++)); do
        
        echo "RUN: $run"
        
        # Base configurations for training without embeddings
        #echo "$PY $LOG $dataset $CNN --learnable 200 --hidden 256 --seed $run --nepochs $EP"
        #echo
        #$PY $LOG $dataset   $CNN   --learnable 200 --hidden 256    --seed $run --nepochs $EP
        #echo

        #echo "$PY $LOG $dataset $CNN --learnable 200 --hidden 256 --supervised --seed $run --nepochs $EP"
        #echo
        #$PY $LOG $dataset	$CNN	--learnable 200	--hidden 256    --supervised   --seed $run    --nepochs $EP 

        # Loop over ordered embeddings
        for embedding in "${embeddings[@]}"; do

            # Split the embedding name and command
            embed_name=$(echo $embedding | awk '{print $2}')
            embed_cmd=$(echo $embedding | cut -d' ' -f2-)
            
            echo
            echo "-------- running $embed_name language model via $embedding --------"
            echo
            
            #
            # Command with embedding configurations
            #
            echo "$PY $LOG $dataset $CNN --hidden 256 $embedding --seed $run --nepochs $EP"
            $PY $LOG $dataset $CNN --hidden 256 $embedding --seed $run --nepochs $EP
            
            #echo "$PY $LOG $dataset $CNN --hidden 256 $embedding --tunable --seed $run --nepochs $EP"
            #$PY $LOG $dataset $CNN --hidden 256 $embedding --tunable --seed $run --nepochs $EP
            
            #echo "$PY $LOG $dataset $CNN --learnable 20 --hidden 256 $embedding --tunable --seed $run --droptype learn --nepochs $EP"
            #$PY $LOG $dataset $CNN --learnable 20 --hidden 256 $embedding --tunable --seed $run --droptype learn --nepochs $EP
            
            echo "$PY $LOG $dataset $CNN --hidden 2048 $embedding --supervised --seed $run --nepochs $EP"
            $PY $LOG $dataset $CNN --hidden 2048 $embedding --supervised --seed $run --nepochs $EP
            
            #echo "$PY $LOG $dataset $CNN --hidden 1024 $embedding --supervised --tunable --seed $run --nepochs $EP"
            #$PY $LOG $dataset $CNN --hidden 1024 $embedding --supervised --tunable --seed $run --nepochs $EP
            
            #echo "$PY $LOG $dataset $CNN --learnable 20 --hidden 256 $embedding --supervised --tunable --seed $run --droptype learn --nepochs $EP"
            #$PY $LOG $dataset $CNN --learnable 20 --hidden 256 $embedding --supervised --tunable --seed $run --droptype learn --nepochs $EP

        done

    done
done
