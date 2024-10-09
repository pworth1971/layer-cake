#!/bin/bash


# -----------------------------------------------------------------------------------------------------------------------------------------
# CONFIG INFO
# NB: must be run from /bin directory

# Set the CUDA device for all processes
export CUDA_VISIBLE_DEVICES=1                   # Change to your specific GPU ID as needed

# supported networks, drop probability included
#CNN="--net cnn --dropprob .2"
#LSTM="--net lstm --dropprob .2"
#ATTN="--net attn --dropprob .2"

CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"

EP="200"                # number of epochs

# embedding config
GLOVE="--pretrained glove --glove-path ../.vector_cache/GloVe" 
WORD2VEC="--pretrained word2vec --word2vec-path ../.vector_cache/Word2Vec"
FASTTEXT="--pretrained fasttext --fasttext-path ../.vector_cache/fastText"
BERT="--pretrained bert --bert-path ../.vector_cache/BERT"
ROBERTA="--pretrained roberta --roberta-path ../.vector_cache/RoBERTa"
#LLAMA="--pretrained llama --llama-path ../.vector_cache/LLaMa"
XLNET="--pretrained xlnet --xlnet-path ../.vector_cache/XLNet"
GPT2="--pretrained gpt2 --gp2-path ../.vector_cache/GPT2"


PY="python ../src/layer_cake.py"                                # source file
LOG="--log-file ../log/nn_lstm_rcv1.test"                       # output log file for metrics

# dataset config
#ng_dataset="--dataset 20newsgroups --pickle-dir ../pickles"                     # 20_newsgroups (single label, 20 classes)
#ohm_dataset="--dataset ohsumed --pickle-dir ../pickles"                         # ohsumed (multi-label, 23 classes)
#reut_dataset="--dataset reuters21578 --pickle-dir ../pickles"                   # reuters21578 (multi-label, 115 classes)
#rcv_dataset="--dataset rcv1 --pickle-dir ../pickles"                            # RCV1-v2 (multi-label, 101 classes)
 
dataset="--dataset rcv1 --pickle-dir ../pickles"                        
# -----------------------------------------------------------------------------------------------------------------------------------------


for run in {1..2}                   
do

$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256    --seed $run    --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256    --supervised   --seed $run    --nepochs $EP 

## GloVe
$PY $LOG $dataset	$LSTM	--hidden 256	$GLOVE    --seed $run --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	$GLOVE   --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	$GLOVE   --tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 2048	$GLOVE   --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 1024	$GLOVE   --supervised	--tunable --seed $run   --nepochs $EP

## Word2Vec
$PY $LOG $dataset	$LSTM	--hidden 256	$WORD2VEC    --seed $run    --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	$WORD2VEC   --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	$WORD2VEC   --tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 2048	$WORD2VEC   --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 1024	$WORD2VEC   --supervised	--tunable --seed $run   --nepochs $EP

## fastText
$PY $LOG $dataset	$LSTM	--hidden 256	$FASTTEXT    --seed $run --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	$FASTTEXT    --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	$FASTTEXT   --tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 2048	$FASTTEXT    --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 1024	$FASTTEXT    --supervised	--tunable --seed $run   --nepochs $EP

## BERT
$PY $LOG $dataset	$LSTM	--channels 128	$BERT  --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--channels 128	$BERT  --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 20	--channels 128	$BERT --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$LSTM	--channels 128	$BERT  --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$LSTM	--channels 128	$BERT  --supervised	--tunable --seed $run   --nepochs $EP

## LLAMA
$PY $LOG $dataset	$LSTM	--channels 128	$LLAMA  --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--channels 128	$LLAMA  --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 20	--channels 128	$LLAMA --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$LSTM	--channels 128	$LLAMA  --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$LSTM	--channels 128	$LLAMA  --supervised	--tunable --seed $run   --nepochs $EP

done

