#!/bin/bash

# 
# must be run from /bin directory
#
PY="python ../src/layer_cake.py"
FT="python ../src/fasttext.py"
LOG="--log-file ../log/20newsgroups.test"
CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"
EP="10"

GLOVE="--pretrained glove --glove-path ../.vector_cache" 
WORD2VEC="--pretrained word2vec --word2vec-path ../.vector_cache/GoogleNews-vectors-negative300.bin"
FASTTEXT="--pretrained fasttext --fasttext-path ../.vector_cache/crawl-300d-2M.vec"
BERT="--pretrained bert --bert-path ../.vector_cache"


for run in {1..10}                   
do

# -----------------------------------------------------------------------------------------------------------------------------------------
#
# 20_newsgroups (single label, 20 classes)
#
# -----------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset 20newsgroups --pickle-dir ../pickles"

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


#######################################################################
# LSTM
#######################################################################
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256 --seed $run    --nepochs $EP

## GloVe
#$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --glove-path $EMB    --seed $run --nepochs $EP
#$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run   --nepochs $EP
#$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run --droptype learn  --nepochs $EP
#$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--glove-path $EMB   --supervised --seed $run    --nepochs $EP
#$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--glove-path $EMB   --supervised	--tunable --seed $run   --nepochs $EP

## Word2Vec
#$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin    --seed $run    --nepochs $EP
#$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --tunable --seed $run   --nepochs $EP
#$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --tunable --seed $run --droptype learn  --nepochs $EP
#$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --supervised --seed $run    --nepochs $EP
#$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --supervised	--tunable --seed $run   --nepochs $EP

## fastText
#$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --seed $run --nepochs $EP
#$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --tunable --seed $run   --nepochs $EP
#$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec   --tunable --seed $run --droptype learn  --nepochs $EP
#$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised --seed $run    --nepochs $EP
#$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised	--tunable --seed $run   --nepochs $EP

## BERT
#$PY $LOG $dataset	$LSTM	--channels 128	--pretrained bert --bert-path $EMB  --seed $run   --nepochs $EP
#$PY $LOG $dataset	$LSTM	--channels 128	--pretrained bert --bert-path $EMB  --tunable --seed $run   --nepochs $EP
#$PY $LOG $dataset	$LSTM	--learnable 20	--channels 128	--pretrained bert   --bert-path $EMB --tunable --seed $run --droptype learn    --nepochs $EP
#$PY $LOG $dataset	$LSTM	--channels 128	--pretrained bert --bert-path $EMB  --supervised --seed $run    --nepochs $EP
#$PY $LOG $dataset	$LSTM	--channels 128	--pretrained bert --bert-path $EMB  --supervised	--tunable --seed $run   --nepochs $EP


#######################################################################
# ATTN
#######################################################################
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run    --nepochs $EP

## GloVe
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove --glove-path $EMB    --seed $run --nepochs $EP
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run   --nepochs $EP
#$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run --droptype learn  --nepochs $EP
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--glove-path $EMB   --supervised --seed $run    --nepochs $EP
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--glove-path $EMB   --supervised	--tunable --seed $run   --nepochs $EP

## Word2Vec
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run   --nepochs $EP
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--tunable --seed $run   --nepochs $EP
#$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--tunable --seed $run --droptype learn  --nepochs $EP
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--supervised --seed $run    --nepochs $EP
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--supervised	--tunable --seed $run   --nepochs $EP

## fastText
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --seed $run --nepochs $EP
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --tunable --seed $run   --nepochs $EP
#$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec   --tunable --seed $run --droptype learn  --nepochs $EP
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised --seed $run    --nepochs $EP
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised	--tunable --seed $run   --nepochs $EP

## BERT
#$PY $LOG $dataset	$ATTN	--channels 128	--pretrained bert --bert-path $EMB  --seed $run   --nepochs $EP
#$PY $LOG $dataset	$ATTN	--channels 128	--pretrained bert --bert-path $EMB  --tunable --seed $run   --nepochs $EP
#$PY $LOG $dataset	$ATTN	--learnable 20	--channels 128	--pretrained bert   --bert-path $EMB --tunable --seed $run --droptype learn    --nepochs $EP
#$PY $LOG $dataset	$ATTN	--channels 128	--pretrained bert --bert-path $EMB  --supervised --seed $run    --nepochs $EP
#$PY $LOG $dataset	$ATTN	--channels 128	--pretrained bert --bert-path $EMB  --supervised	--tunable --seed $run   --nepochs $EP


# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------



done
