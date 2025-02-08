#!/usr/bin/env bash

PY="python ../src/wrd_layer_cake_v3.0.py --nepochs 77 --patience 5 --seed 77"
LOG="--log-file ../log/lc_nn_wrd_test.v3.0.test"

CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"

#
# Set the CUDA device for all processes
#
export CUDA_VISIBLE_DEVICES=0                                               # GPU ID for code execution





# ---------------------------------------------------------------------------------------------------------------------------------------
#
# imdb dataset, single-label, 2 classes
#
dataset="--dataset imdb"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 56	--channels 64

$PY $LOG $dataset	$CNN	--channels 64	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 64	--pretrained glove	--tunable
#$PY $LOG $dataset	$CNN	--learnable 56	--channels 64	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 64	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 64	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 64	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 64	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$CNN	--learnable 56	--channels 64	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 64	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 64	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 64	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 64	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$CNN	--learnable 56	--channels 64	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 64	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 64	--pretrained fasttext	--supervised	--tunable


##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 56	--hidden 64

$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained glove	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 56	--hidden 64	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 56	--hidden 64	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 56	--hidden 64	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 64	--pretrained fasttext	--supervised	--tunable

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 56	--hidden 64

$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained glove	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 56	--hidden 64	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 56	--hidden 64	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 56	--hidden 64	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 64	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------
#
# dataset bbc-news, 5 classes, single-label
#
dataset="--dataset bbc-news"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 56	--channels 128

$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--tunable
#$PY $LOG $dataset	$CNN	--learnable 56	--channels 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$CNN	--learnable 56	--channels 128	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$CNN	--learnable 56	--channels 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised	--tunable


##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 56	--hidden 128

$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained glove	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 56	--hidden 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 56	--hidden 128	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 56	--hidden 128	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained fasttext	--supervised	--tunable


##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 56	--hidden 128

$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained glove	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 56	--hidden 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 56	--hidden 128	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 56	--hidden 128	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained fasttext	--supervised	--tunable

# ---------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------
#
# reuters21578 dataset, 115 classes, multi-label
#
dataset="--dataset reuters21578"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 256

$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200 --hidden 256

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--supervised	--tunable


##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------
#
# 20newsgroups dataset, 20 classes, single-label
#
dataset="--dataset 20newsgroups"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 100	--channels 128

$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--tunable
#$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 100	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$CNN	--learnable 100	--channels 128	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$CNN	--learnable 100	--channels 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 100	--hidden 128

$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained glove	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 100	--hidden 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 100	--hidden 128	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 100	--hidden 128	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 128	--pretrained fasttext	--supervised	--tunable

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 100	--hidden 128

$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained glove	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 100	--hidden 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 100	--hidden 128	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 100	--hidden 128	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 128	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------




# ---------------------------------------------------------------------------------------------------------------------------------------
#
# ohsumed data, 23 classes, multi-label
#
dataset="--dataset ohsumed"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512

$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200 --hidden 512

$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 512	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove	--supervised

$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 512	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec	--supervised

$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 512	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttext	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttext	--supervised

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	--hideen 512

$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 512	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 512	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 512	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------
#
# arxiv_protoformer data, 10 classes, single-label
#
dataset="--dataset arxiv_protoformer"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 115	--channels 256

$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--tunable
#$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 115    --hidden 256

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttest	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--supervised	--tunable

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 256

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------
#
# arxiv data, 58 classes, multi-label
#
dataset="--dataset arxiv"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512

$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200 --channels 512	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200 --hidden 512

$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 512	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 512	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttest	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 512	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttext	--supervised	--tunable

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 512

$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 512	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 512	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 512	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------




# ---------------------------------------------------------------------------------------------------------------------------------------
#
# rcv1 dataset, 101 classes, multi-label
#
dataset="--dataset rcv1"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512

$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200 --hidden 512

$PY $LOG $dataset	$LSTM	--hidden 512 	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 512	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove	--supervised

$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 512	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained word2vec	--supervised

$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 512	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttext	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained fasttext	--supervised

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 512

$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 512	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 512	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512 	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext	--tunable
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 512	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------
