#!/usr/bin/env bash

PY="python ../src/wrd_layer_cake_v2.3.py --nepochs 111 --patience 7 --seed 77"
LOG="--log-file ../log/lc_nn_cuda_test.legacy.test"

CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"

#
# Set the CUDA device for all processes
#
export CUDA_VISIBLE_DEVICES=0                                               # GPU ID for code execution




# ---------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset arxiv_protoformer"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 256

$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--tunable
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--tunable
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttest	--tunable
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--supervised	--tunable

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec	--tunable
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained fasttext	--tunable
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset arxiv"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 256

$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--tunable
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--tunable
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttest	--tunable
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--supervised	--tunable

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec	--tunable
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained fasttext	--tunable
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset imdb"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 256

$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--tunable
$PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--tunable
$PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--tunable
$PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised	--tunable


##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable
$PY $LOG $dataset	$LSTM	--learnable 5	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$LSTM	--learnable 5	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--tunable
$PY $LOG $dataset	$LSTM	--learnable 5	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained fasttext	--supervised	--tunable

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--tunable
$PY $LOG $dataset	$ATTN	--learnable 5	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$ATTN	--learnable 5	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--tunable
$PY $LOG $dataset	$ATTN	--learnable 5	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset reuters21578"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 256

$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--tunable
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--tunable
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--tunable
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--supervised	--tunable


##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec	--tunable
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained fasttext	--tunable
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset bbc-news"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 256

$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--tunable
$PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--tunable
$PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--tunable
$PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable
$PY $LOG $dataset	$LSTM	--learnable 5	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$LSTM	--learnable 5	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--tunable
$PY $LOG $dataset	$LSTM	--learnable 5	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained fasttext	--supervised	--tunable

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--tunable
$PY $LOG $dataset	$ATTN	--learnable 5	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$ATTN	--learnable 5	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--tunable
$PY $LOG $dataset	$ATTN	--learnable 5	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset 20newsgroups"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 256

$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--tunable
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--tunable
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--tunable
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained word2vec	--supervised
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext	--tunable
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained fasttext	--supervised
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained fasttext	--supervised	--tunable

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--tunable
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--tunable
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--tunable
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset ohsumed"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512

$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--tunable
$PY $LOG $dataset	$CNN	--learnable 23	--channels 512	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--tunable
$PY $LOG $dataset	$CNN	--learnable 23	--channels 512	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--tunable
$PY $LOG $dataset	$CNN	--learnable 23	--channels 512	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200

$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--tunable
$PY $LOG $dataset	$LSTM	--learnable 23	--hidden 1024	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised

$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained word2vec	--tunable
$PY $LOG $dataset	$LSTM	--learnable 23	--hidden 1024	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained word2vec	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained word2vec	--supervised

$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained fasttext	--tunable
$PY $LOG $dataset	$LSTM	--learnable 23	--hidden 1024	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained fasttext	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained fasttext	--supervised

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable
$PY $LOG $dataset	$ATTN	--learnable 23	--hidden 1024	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec	--tunable
$PY $LOG $dataset	$ATTN	--learnable 23	--hidden 1024	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained fasttext	--tunable
$PY $LOG $dataset	$ATTN	--learnable 23	--hidden 1024	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset rcv1"

##
# CNN runs
##
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 512

$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--tunable
$PY $LOG $dataset	$CNN	--learnable 101	--channels 512	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--tunable
$PY $LOG $dataset	$CNN	--learnable 101	--channels 512	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--tunable
$PY $LOG $dataset	$CNN	--learnable 101	--channels 512	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--supervised
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--supervised	--tunable

##
# LSTM runs
##
#$PY $LOG $dataset	$LSTM	--learnable 200

$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--tunable
$PY $LOG $dataset	$LSTM	--learnable 101	--hidden 1024	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised

$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained word2vec
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained word2vec	--tunable
$PY $LOG $dataset	$LSTM	--learnable 101	--hidden 1024	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained word2vec	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained word2vec	--supervised

$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained fasttext
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained fasttext	--tunable
$PY $LOG $dataset	$LSTM	--learnable 101	--hidden 1024	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained fasttext	--supervised	--tunable
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained fasttext	--supervised

##
# ATTN runs
##
#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256

$PY $LOG $dataset	$ATTN	--hidden 2048	--pretrained glove
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable
$PY $LOG $dataset	$ATTN	--learnable 101	--hidden 1024	--pretrained glove	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--pretrained glove	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 2048	--pretrained word2vec
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec	--tunable
$PY $LOG $dataset	$ATTN	--learnable 101	--hidden 1024	--pretrained word2vec	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--pretrained word2vec	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained word2vec	--supervised	--tunable

$PY $LOG $dataset	$ATTN	--hidden 2048	--pretrained fasttext
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained fasttext	--tunable
$PY $LOG $dataset	$ATTN	--learnable 101	--hidden 1024	--pretrained fasttext	--tunable --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--pretrained fasttext	--supervised
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained fasttext	--supervised	--tunable
# ---------------------------------------------------------------------------------------------------------------------------------------
