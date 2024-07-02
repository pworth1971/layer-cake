PG="python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir ../.vector_cache --log-file ../log/svm_test.out --learner svm --mode glove"
echo $PG
$PG