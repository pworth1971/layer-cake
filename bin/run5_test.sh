# 
# should be run from bin directory
#

EMB="../.vector_cache"
PY="python ../src/layer_cake.py"
FT="python ../src/fasttext.py"
LOG="--log-file ../log/5runs.pretrained"
CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"
EP="10"

for run in {1..5}                   
do


# -----------------------------------------------------------------------------------------------------------------------------------------
#
# reuters21578
#
# -----------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset reuters21578"

#######################################################################
# CNN
#######################################################################
$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run  --nepochs $EP

## fastText
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec --tunable --seed $run --nepochs $EP
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	    --pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec --supervised --seed $run  --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 512	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec --supervised	--tunable --seed $run   --nepochs $EP

## GloVe
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove  --glove-path $EMB --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--glove-path $EMB --tunable --seed $run --nepochs $EP
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	    --pretrained glove	--glove-path $EMB --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--glove-path $EMB --supervised --seed $run  --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--glove-path $EMB --supervised	--tunable --seed $run   --nepochs $EP

## Word2Vec
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run --nepochs $EP
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	    --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised --seed $run  --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 512	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised	--tunable --seed $run   --nepochs $EP


#######################################################################
# LSTM
#######################################################################
$PY $LOG $dataset	$LSTM	--learnable 200 --seed $run --nepochs $EP

## fastText
$PY $LOG $dataset	$LSTM	--hidden 256    --pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256    --pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --tunable --seed $run --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	    --pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised --seed $run  --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised	--tunable --seed $run   --nepochs $EP

## GloVe
$PY $LOG $dataset	$LSTM	--hidden 256    --pretrained glove  --glove-path $EMB --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256    --pretrained glove	--glove-path $EMB --tunable --seed $run --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	    --pretrained glove	--glove-path $EMB --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--glove-path $EMB --supervised --seed $run  --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--glove-path $EMB --supervised	--tunable --seed $run   --nepochs $EP

## Word2Vec
$PY $LOG $dataset	$LSTM	--hidden 256    --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256    --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	    --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised --seed $run  --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised	--tunable --seed $run   --nepochs $EP


#######################################################################
# ATTN
#######################################################################
$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run    --nepochs $EP

## fastText
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --tunable --seed $run --nepochs $EP
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024       --pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256        --pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised	--tunable --seed $run   --nepochs $EP

## GloVe
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove  --glove-path $EMB --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--glove-path $EMB --tunable --seed $run --nepochs $EP
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024       --pretrained glove  --glove-path $EMB	--tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256        --pretrained glove  --glove-path $EMB   --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--glove-path $EMB   --supervised	--tunable --seed $run   --nepochs $EP

## Word2Vec
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run --nepochs $EP
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024       --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	    --pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised	--tunable --seed $run   --nepochs $EP

# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------




# -----------------------------------------------------------------------------------------------------------------------------------------
#
# 20_newsgroups
#
# -----------------------------------------------------------------------------------------------------------------------------------------
dataset="--dataset 20newsgroups"

#######################################################################
# CNN
#######################################################################
$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run  --nepochs $EP

## fastText
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --seed $run
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	    --pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised	--tunable --seed $run   --nepochs $EP

## GloVe
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove  --glove-path $EMB   --seed $run
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--glove-path $EMB   --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	    --pretrained glove  --glove-path $EMB --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--glove-path $EMB   --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--glove-path $EMB   --supervised	--tunable --seed $run   --nepochs $EP

## Word2Vec
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --tunable --seed $run --droptype learn    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$CNN	--channels 128	--pretrained gword2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --supervised	--tunable --seed $run   --nepochs $EP


#######################################################################
# LSTM
#######################################################################
$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256 --seed $run    --nepochs $EP

## fastText
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --seed $run --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec   --tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised	--tunable --seed $run   --nepochs $EP

## GloVe
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --glove-path $EMB    --seed $run --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--glove-path $EMB   --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--glove-path $EMB   --supervised	--tunable --seed $run   --nepochs $EP

## Word2Vec
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin    --seed $run    --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin   --supervised	--tunable --seed $run   --nepochs $EP



#######################################################################
# ATTN
#######################################################################
$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run    --nepochs $EP

## fastText
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --seed $run --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec   --tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained fasttext --fasttext-path $EMB/crawl-300d-2M.vec    --supervised	--tunable --seed $run   --nepochs $EP

## GloVe
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove --glove-path $EMB    --seed $run --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained glove	--glove-path $EMB   --tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--glove-path $EMB   --supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--glove-path $EMB   --supervised	--tunable --seed $run   --nepochs $EP

## Word2Vec
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--tunable --seed $run   --nepochs $EP
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--tunable --seed $run --droptype learn  --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--supervised --seed $run    --nepochs $EP
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin	--supervised	--tunable --seed $run   --nepochs $EP

# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------


done