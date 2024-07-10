#!/bin/bash

#
# NB: presumes 'conda activate python38' is in place with necessary python dependencies
#

#
# Get script name using basename with the $0 variable, 
# which contains the script's call path
#
script_name=$(basename "$0")


#
# command line params
# NB: must be run from /bin directory
#
PY="python ../src/layer_cake.py"
LOG="--log-file ../log/lc_test2.test"
CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"
EP="10"


#
# supported neural network architecture
#
nets=(lstm attn cnn)

#
#supported datasets
#
#datasets=(ng_dataset ohm_dataset reut_dataset)

datasets=(\
    "--dataset reuters21578 --pickle-dir ../pickles" \
    "--dataset ohsumed --pickle-dir ../pickles" \
    "--dataset 20newsgroups --pickle-dir ../pickles"
)
#ng_dataset="--dataset 20newsgroups --pickle-dir ../pickles"                     # 20_newsgroups (single label, 20 classes)
#ohm_dataset="--dataset ohsumed --pickle-dir ../pickles"                         # ohsumed (multi-label, 23 classes)
#reut_dataset="--dataset reuters21578 --pickle-dir ../pickles"                   # reuters21578 (multi-label, 115 classes)


# Embedding config params
GLOVE="--pretrained glove --glove-path ../.vector_cache" 
WORD2VEC="--pretrained word2vec --word2vec-path ../.vector_cache/GoogleNews-vectors-negative300.bin"
FASTTEXT="--pretrained fasttext --fasttext-path ../.vector_cache/crawl-300d-2M.vec"
BERT="--pretrained bert --bert-path ../.vector_cache"

# Main log directory
main_log_dir="../log"


# Print script name
echo "Running script: $script_name"
echo "Logs will be stored in directory: $main_log_dir"



# Function to run experiments for a single network
run_network() {

    local net=$1
    local net_log_file="$main_log_dir/${net}_output_$(date +%Y%m%d%H%M%S).log"

    echo "Starting runs for $net, logging to $net_log_file"

    for dataset in "${datasets[@]}"; do
        for run in {1..2}; do
            echo "Run $run for $net on $dataset" >> "$net_log_file"
            
            # Base Neural Net config
            $PY $LOG $dataset --net $net --learnable 200 --hidden 256 --seed $run --nepochs $EP                                             >> "$net_log_file" 2>&1
            
            # Iterate over embeddings
            for emb in "$GLOVE" "$WORD2VEC" "$FASTTEXT" "$BERT"; do
                $PY $LOG $dataset --net $net --hidden 256 $emb --seed $run --nepochs $EP                                                    >> "$net_log_file" 2>&1
                $PY $LOG $dataset --net $net --hidden 256 $emb --tunable --seed $run --nepochs $EP                                          >> "$net_log_file" 2>&1
                $PY $LOG $dataset --net $net --learnable 20 --hidden 256 $emb --tunable --seed $run --droptype learn --nepochs $EP          >> "$net_log_file" 2>&1
                $PY $LOG $dataset --net $net --hidden 2048 $emb --supervised --seed $run --nepochs $EP                                      >> "$net_log_file" 2>&1
                $PY $LOG $dataset --net $net --hidden 1024 $emb --supervised --tunable --seed $run --nepochs $EP                            >> "$net_log_file" 2>&1
            done
        done
    done
}


# Start each network in its own background process
for net in "${nets[@]}"; do
    run_network "$net" &
done

# Wait for all background jobs to finish
wait

# Consolidate logs (optional)
echo "Consolidating logs..."
consolidated_log="$main_log_dir/consolidated_output_$(date +%Y%m%d%H%M%S).log"
cat "$main_log_dir"/*.log > "$consolidated_log"
echo "All logs consolidated into $consolidated_log"


