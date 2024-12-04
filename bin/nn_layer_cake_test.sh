#!/usr/bin/env bash

PY="python ../src/main.py"
LOG="--log-file ../log/legacy.test"

# Define datasets, network types, embeddings, and number of runs
DATASETS=("20newsgroups" "ohsumed" "reuters21578" "bbc-news")

#NET_TYPES=("cnn" "lstm" "attn")
NET_TYPES=("cnn")

#EMBEDDINGS=("glove" "word2vec" "fasttext" "bert")

EMBEDDINGS=("glove" "word2vec" "fasttext")

RUNS=1

#
# Iterate over datasets, network types, embeddings, and runs
#
for dataset_name in "${DATASETS[@]}"; do
    for net_type in "${NET_TYPES[@]}"; do

        for embedding in "${EMBEDDINGS[@]}"; do
            for run in $(seq 1 $RUNS); do

                # run once per combination
                #single_run_cmd="$PY $LOG $dataset $net --learnable 200 --channels 256 --seed $run"
                #echo $single_run_cmd
                #eval $single_run_cmd

                dataset="--dataset ${dataset_name}"
                net="--net ${net_type}"
                pretrained="--pretrained ${embedding}"

                # Debugging output
                echo
                echo "-----------------------------------"
                echo "Dataset: ${dataset_name}, Net: ${net_type}, Pretrained: ${embedding}, Run: ${run}"
                echo "-----------------------------------"
                echo

                # Generate the five commands dynamically
                # Echo and execute each command
                cmd1="$PY $LOG $dataset $net --channels 128 $pretrained --seed $run"
                echo $cmd1
                eval $cmd1

                #cmd2="$PY $LOG $dataset $net --channels 128 $pretrained --tunable --seed $run"
                #echo $cmd2
                #eval $cmd2

                #cmd3="$PY $LOG $dataset $net --learnable 28 --channels 128 $pretrained --tunable --seed $run --droptype learn"
                #echo $cmd3
                #eval $cmd3

                cmd4="$PY $LOG $dataset $net --channels 128 $pretrained --supervised --seed $run"
                echo $cmd4
                eval $cmd4

                #cmd5="$PY $LOG $dataset $net --channels 128 $pretrained --supervised --tunable --seed $run"
                #echo $cmd5
                #eval $cmd5

            done
        done
    done
done
