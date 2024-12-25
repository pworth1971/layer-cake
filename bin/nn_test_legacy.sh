#!/usr/bin/env bash

PY="python ../src/wrd_layer_cake.py"
LOG="--log-file ../log/lc_nn_cuda_test.legacy.test"


CNN="--net cnn"
#LSTM="--net lstm"
#ATTN="--net attn"

#
# Set the CUDA device for all processes
#
export CUDA_VISIBLE_DEVICES=0                                               # GPU ID for code execution



for run in {1..1}                   # 0 is for plots, 1 is already performed in hyper parameter search
do


    dataset="--dataset arxiv"
    #$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run

    $PY $LOG $dataset	$CNN	--channels 256	--pretrained glove --seed $run
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained glove	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec --seed $run
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained word2vec	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext --seed $run
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained fasttext	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised	--tunable --seed $run

    #$PY $LOG $dataset	$LSTM	--learnable 200 --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run

    #$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run



    dataset="--dataset bbc-news"
    #$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run

    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained glove	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained word2vec	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained glove	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised	--tunable --seed $run


    #$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256 --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$LSTM	--learnable 5	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable --seed $run

    #$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$ATTN	--learnable 5	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run



    dataset="--dataset reuters21578"
    #$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run

    $PY $LOG $dataset	$CNN	--channels 256	--pretrained glove --seed $run
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained glove	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec --seed $run
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained word2vec	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext --seed $run
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained fasttext	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--supervised	--tunable --seed $run

    #$PY $LOG $dataset	$LSTM	--learnable 200 --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run

    #$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run



    dataset="--dataset 20newsgroups"
    #$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run

    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 20	--channels 128	--pretrained glove	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 20	--channels 128	--pretrained word2vec	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 20	--channels 128	--pretrained glove	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised	--tunable --seed $run

    #$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256 --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable --seed $run

    #$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run


    dataset="--dataset imdb"
    #$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run

    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained glove	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained word2vec	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 5	--channels 128	--pretrained glove	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext	--supervised	--tunable --seed $run


    #$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256 --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$LSTM	--learnable 5	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable --seed $run

    #$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$ATTN	--learnable 5	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run



    dataset="--dataset ohsumed"
    #$PY $LOG $dataset	$CNN	--learnable 200	--channels 512 --seed $run

    $PY $LOG $dataset	$CNN	--channels 512	--pretrained glove --seed $run
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 23	--channels 512	--pretrained glove	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec --seed $run
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 23	--channels 512	--pretrained word2vec	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec	--supervised	--tunable --seed $run

    $PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext --seed $run
    $PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext	--tunable --seed $run
    $PY $LOG $dataset	$CNN	--learnable 23	--channels 512	--pretrained fasttext	--tunable --seed $run --droptype learn
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--supervised --seed $run
    $PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext	--supervised	--tunable --seed $run


    #$PY $LOG $dataset	$LSTM	--learnable 200 --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$LSTM	--learnable 23	--hidden 1024	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable --seed $run
    #$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised --seed $run

    #$PY $LOG $dataset	$ATTN	--learnable 200	 --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable --seed $run
    #$PY $LOG $dataset	$ATTN	--learnable 23	--hidden 1024	--pretrained glove	--tunable --seed $run --droptype learn
    #$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised --seed $run
    #$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run




done