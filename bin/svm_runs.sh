

PY="python ../src/svm_baseline"
LOG="--log-file ../log/svm_test.log"

for run in {1..5}                                   
do


# 20newsgroups (singl-label) basic TF-IDF with SVM:
python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir '../.vector_cache' --log-file '../log/svm_test.out' --learner svm --mode tfidf

# 20newsgroups (singl-label) basic TF-IDF with LR:
python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir '../.vector_cache' --log-file '../log/svm_test.out' --learner lr --mode tfidf

#
# NOT WORKING
# Supervised Term Weighting with Logistic Regression (Warning: not converging)
#python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir '../.vector_cache' --log-file '../log/svm_test.out' --learner lr --mode stw --tsr ig --stwmode wave --optimc
#

# Using GloVe Embeddings with SVM default params:
python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir '../.vector_cache' --log-file '../log/svm_test.out' --learner svm --mode glove 

# Using GloVe Embeddings with LR default params:
python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir '../.vector_cache' --log-file '../log/svm_test.out' --learner lr --mode glove 

# Using GloVe Embeddings + Supervised (WC) Embeddings with SVM default params:
python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir '../.vector_cache' --log-file '../log/svm_test.out' --learner svm --mode glove-sup

# Using GloVe Embeddings + Supervised (WC) Embeddings with LR default params:
python ../src/svm_baselines.py --dataset 20newsgroups --embedding-dir '../.vector_cache' --log-file '../log/svm_test.out' --learner lr --mode glove-sup


# BERT Embeddings Combined with SVM for Document Classification:
# python script_name.py --dataset rcv1 --learner svm --mode bert --combine-strategy mean --batch-size 256

# Forcing Re-computation of BERT Embeddings:
# python script_name.py --dataset rcv1 --learner svm --mode bert --force-embeddings



# python fasttext.py --dataset 20newsgroups	--learnable 50	--lr 0.5	--nepochs 200	--seed $run --pickle-path ../pickles/20newsgroups.pickle
# python fasttext.py --dataset jrcall	--learnable 200	--lr 0.25	--nepochs 200	--seed $run --pickle-path ../pickles/jrcall.pickle
# python fasttext.py --dataset ohsumed	--learnable 200	--lr 0.5	--nepochs 200	--seed $run --pickle-path ../pickles/ohsumed.pickle
# python fasttext.py --dataset rcv1	--learnable 50	--lr 0.5	--nepochs 100	--seed $run --pickle-path ../pickles/rcv1.pickle
# python fasttext.py --dataset reuters21578	--learnable 200	--lr 0.5	--nepochs 100	--seed $run --pickle-path ../pickles/reuters21578.pickle
# python fasttext.py --dataset wipo-sl-sc	--learnable 200	--lr 0.5	--nepochs 10	--seed $run --pickle-path ../pickles/wipo-sl-sc.pickle

done
