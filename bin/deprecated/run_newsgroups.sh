# should be run from bin directory


EMB="../.vector_cache"
PY="python ../src/main.py"
LOG="--log-file ../log/r10runs.pretrained"
CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"

for run in {1..5}              # 0 is for plots, 1 is already performed in hyper parameter search
do


# -----------------------------------------------------------------------------------------------------------------------------------------
#
# 20_newsgroups
#
# -----------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset 20newsgroups"

#######################################################################
# CNN
#######################################################################
$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run

## GloVe
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove  --seed $run
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--glove-path $EMB   --tunable --seed $run
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	    --pretrained glove  --glove-path $EMB --tunable --seed $run --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--glove-path $EMB   --supervised --seed $run
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--glove-path $EMB   --supervised	--tunable --seed $run

## Word2Vec
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --tunable --seed $run
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --supervised --seed $run
$PY $LOG $dataset	$CNN	--channels 128	--pretrained gword2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised	--tunable --seed $run


#######################################################################
# LSTM
#######################################################################
$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256 --seed $run

## GloVe
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --glove-path $EMB    --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--glove-path $EMB   --supervised --seed $run
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--glove-path $EMB   --supervised	--tunable --seed $run

## Word2Vec
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin    --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --tunable --seed $run
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --tunable --seed $run --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --supervised --seed $run
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --supervised	--tunable --seed $run



#######################################################################
# ATTN
#######################################################################
$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run

## GloVe
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove --glove-path $EMB    --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--glove-path $EMB   --supervised --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--glove-path $EMB   --supervised	--tunable --seed $run

## Word2Vec
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--tunable --seed $run
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--supervised --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--supervised	--tunable --seed $run

# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------




# -----------------------------------------------------------------------------------------------------------------------------------------
#
# reuters21578
#
# -----------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset reuters21578"

#######################################################################
# CNN
#######################################################################
$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run

## GloVe
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove  --glove-path $EMB --seed $run
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--glove-path $EMB --tunable --seed $run
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	    --pretrained glove	--glove-path $EMB --tunable --seed $run --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--glove-path $EMB --supervised --seed $run
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--glove-path $EMB --supervised	--tunable --seed $run

## Word2Vec
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	    --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised --seed $run
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised	--tunable --seed $run


#######################################################################
# LSTM
#######################################################################
$PY $LOG $dataset	$LSTM	--learnable 200 --seed $run

## GloVe
$PY $LOG $dataset	$LSTM	--hidden 256    --pretrained glove  --glove-path $EMB --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256    --pretrained glove	--glove-path $EMB --tunable --seed $run
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	    --pretrained glove	--glove-path $EMB --tunable --seed $run --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--glove-path $EMB --supervised --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--glove-path $EMB --supervised	--tunable --seed $run

## Word2Vec
$PY $LOG $dataset	$LSTM	--hidden 256    --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256    --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	    --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised	--tunable --seed $run


#######################################################################
# ATTN
#######################################################################
$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run

## GloVe
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove  --glove-path $EMB --seed $run
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--glove-path $EMB --tunable --seed $run
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024       --pretrained glove  --glove-path $EMB	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256        --pretrained glove  --glove-path $EMB   --supervised --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--glove-path $EMB   --supervised	--tunable --seed $run

## Word2Vec
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024       --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	    --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--supervised --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised	--tunable --seed $run
# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------




# -----------------------------------------------------------------------------------------------------------------------------------------
#
# fastText
#
# -----------------------------------------------------------------------------------------------------------------------------------------

common="--dataset-dir ../fasttext/dataset --log-file ../log/10runs.fasttext"
python ../src/fasttext.py --dataset 20newsgroups	--learnable 50	--lr 0.5	--nepochs 50	--seed $run --pickle-path ../pickles/20newsgroups.pickle
python ../src/fasttext.py --dataset reuters21578	--learnable 200	--lr 0.5	--nepochs 40	--seed $run --pickle-path ../pickles/reuters21578.pickle

# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------

done
