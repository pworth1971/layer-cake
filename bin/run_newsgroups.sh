# should be run from bin directory


EMB="../.vector_cache"
PY="python ../src/main.py"
LOG="--log-file ../log/run_newsgroups.csv"
CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"

for run in {1..2}              # 0 is for plots, 1 is already performed in hyper parameter search
do

dataset="--dataset 20newsgroups"
#$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove --glove-path $EMB --seed $run
#$PY $LOG $dataset	$CNN	--channels 128	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run

#$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256 --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --glove-path $EMB --seed $run
#$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run

#$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove --glove-path $EMB --seed $run
#$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained word2vec --word2vec-path $EMB/GoogleNews-vectors-negative300.bin --seed $run

common="--dataset-dir ../fasttext/dataset --log-file ../log/fasttext.10runs.csv"
python ../src/fasttext.py --dataset 20newsgroups	--learnable 50	--lr 0.5	--nepochs 50	--seed $run --pickle-path ../pickles/20newsgroups.pickle
done
