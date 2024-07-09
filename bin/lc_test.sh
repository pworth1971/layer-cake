#!/bin/bash
#!conda activate python38


# NB: must be run from /bin directory


PY="python ../src/layer_cake.py"
LOG="--log-file ../log/lc_test.test"

CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"

EP="10"


#ng_dataset="--dataset 20newsgroups --pickle-dir ../pickles"                     # 20_newsgroups (single label, 20 classes)
#ohm_dataset="--dataset ohsumed --pickle-dir ../pickles"                         # ohsumed (multi-label, 23 classes)
#reut_dataset="--dataset reuters21578 --pickle-dir ../pickles"                   # reuters21578 (multi-label, 115 classes)
#datasets=(ng_dataset ohm_dataset reut_dataset)

datasets=(\
    "--dataset ohsumed --pickle-dir ../pickles" \
    "--dataset reuters21578 --pickle-dir ../pickles" \
    "--dataset 20newsgroups --pickle-dir ../pickles"
)

nets=(lstm attn cnn)

GLOVE="--pretrained glove --glove-path ../.vector_cache" 
WORD2VEC="--pretrained word2vec --word2vec-path ../.vector_cache/GoogleNews-vectors-negative300.bin"
FASTTEXT="--pretrained fasttext --fasttext-path ../.vector_cache/crawl-300d-2M.vec"
BERT="--pretrained bert --bert-path ../.vector_cache"


for dataset in "${datasets[@]}"; do
    
    for net in "${nets[@]}"; do

        echo
        echo
        echo "-----------------------------------------------------------------------------------------------------------------------------------------"
        echo
        echo "\t\t\t*********************** Starting runs for $net on $dataset ***********************"
        echo 
        echo "-----------------------------------------------------------------------------------------------------------------------------------------"

        for run in {1..10}; do

            # Run base Neural Net config
            echo
            echo
            echo "$PY $LOG $dataset --net $net --learnable 200 --hidden 256 --seed $run --nepochs $EP"
            $PY $LOG $dataset --net "$net" --learnable 200 --hidden 256 --seed $run --nepochs $EP
                       
            echo
            echo
            echo "\t\t------------------------ run $run for $net on $dataset ------------------------"
            echo 

            # Iterate over embeddings
            for emb in "$GLOVE" "$WORD2VEC" "$FASTTEXT" "$BERT"; do
                echo
                echo
                echo "$PY $LOG $dataset --net $net --hidden 256 $emb --seed $run --nepochs $EP"
                $PY $LOG $dataset --net "$net" --hidden 256 $emb --seed $run --nepochs $EP

                echo
                echo
                echo "$PY $LOG $dataset --net $net --hidden 256 $emb --tunable --seed $run --nepochs $EP"
                $PY $LOG $dataset --net "$net" --hidden 256 $emb --tunable --seed $run --nepochs $EP

                echo
                echo
                echo "$PY $LOG $dataset --net $net --learnable 20 --hidden 256 $emb --tunable --seed $run --droptype learn --nepochs $EP"
                $PY $LOG $dataset --net "$net" --learnable 20 --hidden 256 $emb --tunable --seed $run --droptype learn --nepochs $EP

                echo
                echo
                echo "$PY $LOG $dataset --net $net --hidden 2048 $emb --supervised --seed $run --nepochs $EP"
                $PY $LOG $dataset --net "$net" --hidden 2048 $emb --supervised --seed $run --nepochs $EP

                echo
                echo
                echo "$PY $LOG $dataset --net $net --hidden 1024 $emb --supervised --tunable --seed $run --nepochs $EP"
                $PY $LOG $dataset --net "$net" --hidden 1024 $emb --supervised --tunable --seed $run --nepochs $EP
            done
        done        # for run
    done            # for net
done                # for dataset
