#!/bin/bash

# 
# NB: must be run from /bin directory
#

PY="python ../src/layer_cake.py"
LOG="--log-file ../log/lc_systest_reuters2.test"

CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"
EP="200"

GLOVE="--pretrained glove --glove-path ../.vector_cache" 
WORD2VEC="--pretrained word2vec --word2vec-path ../.vector_cache/GoogleNews-vectors-negative300.bin"
FASTTEXT="--pretrained fasttext --fasttext-path ../.vector_cache/crawl-300d-2M.vec"
BERT="--pretrained bert --bert-path ../.vector_cache"
LLAMA="--pretrained llama --llama-path ../.vector_cache"

for run in {1..10}                   
do

# -----------------------------------------------------------------------------------------------------------------------------------------
#

#
# -----------------------------------------------------------------------------------------------------------------------------------------
#ng_dataset="--dataset 20newsgroups --pickle-dir ../pickles"                     # 20_newsgroups (single label, 20 classes)
#ohm_dataset="--dataset ohsumed --pickle-dir ../pickles"                         # ohsumed (multi-label, 23 classes)
#reut_dataset="--dataset reuters21578 --pickle-dir ../pickles"                   # reuters21578 (multi-label, 115 classes)
#rcv_dataset="--dataset rcv1 --pickle-dir ../pickles"                            # RCV1-v2 (multi-label, 101 classes)

#dataset="--dataset 20newsgroups --pickle-dir ../pickles"                     
dataset="--dataset reuters21578 --pickle-dir ../pickles"

#######################################################################
# LSTM
#######################################################################
$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256 --seed $run    --nepochs $EP

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


#######################################################################
# ATTN
#######################################################################
$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run    --nepochs $EP

## GloVe
$PY $LOG $dataset	$ATTN	--hidden 256	$GLOVE    --seed $run --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	$GLOVE   --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	$GLOVE   --tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	$GLOVE   --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	$GLOVE   --supervised	--tunable --seed $run   --nepochs $EP

## Word2Vec
$PY $LOG $dataset	$ATTN	--hidden 256	$WORD2VEC --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	$WORD2VEC	--tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	$WORD2VEC	--tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	$WORD2VEC	--supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	$WORD2VEC	--supervised	--tunable --seed $run   --nepochs $EP

## fastText
$PY $LOG $dataset	$ATTN	--hidden 256	$FASTTEXT    --seed $run --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	$FASTTEXT    --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	$FASTTEXT   --tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	$FASTTEXT    --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	$FASTTEXT    --supervised	--tunable --seed $run   --nepochs $EP

## BERT
$PY $LOG $dataset	$ATTN	--channels 128	$BERT  --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--channels 128	$BERT  --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--learnable 20	--channels 128	$BERT --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$ATTN	--channels 128	$BERT  --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$ATTN	--channels 128	$BERT  --supervised	--tunable --seed $run   --nepochs $EP

## LLAMA
$PY $LOG $dataset	$ATTN	--channels 128	$LLAMA  --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--channels 128	$LLAMA  --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--learnable 20	--channels 128	$LLAMA --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$ATTN	--channels 128	$LLAMA  --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$ATTN	--channels 128	$LLAMA  --supervised	--tunable --seed $run   --nepochs $EP


#######################################################################
# CNN
#######################################################################
$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run  --nepochs $EP

## GloVe
$PY $LOG $dataset	$CNN	--channels 128	$GLOVE      --seed $run
$PY $LOG $dataset	$CNN	--channels 128	$GLOVE	    --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	    $GLOVE    --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$GLOVE	    --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$GLOVE	    --supervised	--tunable --seed $run   --nepochs $EP

## Word2Vec
$PY $LOG $dataset	$CNN	--channels 128	$WORD2VEC   --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$WORD2VEC     --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	$WORD2VEC  --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$WORD2VEC     --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$WORD2VEC --supervised	--tunable --seed $run   --nepochs $EP

## fastText
$PY $LOG $dataset	$CNN	--channels 128	$FASTTEXT    --seed $run
$PY $LOG $dataset	$CNN	--channels 128	$FASTTEXT      --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	    $FASTTEXT     --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$FASTTEXT      --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$FASTTEXT     --supervised	--tunable --seed $run   --nepochs $EP

## BERT
$PY $LOG $dataset	$CNN	--channels 128	$BERT   --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$BERT  --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	$BERT --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$BERT --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$BERT  --supervised	--tunable --seed $run   --nepochs $EP

## LLAMA
$PY $LOG $dataset	$CNN	--channels 128	$LLAMA   --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$LLAMA  --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	$LLAMA --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$LLAMA --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	$LLAMA  --supervised	--tunable --seed $run   --nepochs $EP

# -----------------------------------------------------------------------------------------------------------------------------------------




done
