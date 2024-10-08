# Layer Cake


## Introduction 

Layer Cake is a platform for testing different word embeddings, within different underlying models, for a multi-label, multiclass text classification 
problem against four data sets. The core of the code and modules are forked from a github repo that was designed to test 'word-class embeddings' as 
described in a 2019 paper entitled 'Word-Class Embeddings for Multiclass Text Classification' authored by Alejandro Moreo, Andrea Esuli, and Fabrizio 
Sebastiani who at the time at least were part of the Istituto di Scienza e Tecnologie dell’Informazione Consiglio Nazionale delle Ricerche in Pisa Italy, 
which of course is where the Tower of Pisa is (random fact). 

Word-Class Embeddings (WCEs) are a form of supervised embeddings specially suited for 
multiclass text classification. WCEs are meant to be used as extensions (i.e., by concatenation) to pre-trained embeddings (e.g., GloVe or word2vec) 
embeddings in order to improve the performance of neural classifiers. Original paper available https://arxiv.org/abs/1911.11506, original repo available 
here https://github.com/AlexMoreo/word-class-embeddings. Layer Cake is designed to combine pretrained embeddings of different types, and different 
geometries, together against standard NLP data sets such that their relative performance can be evaluated.



## Platform Overview

The platform consists of three modules:

- ML Baselines: an NLP solve for the supported data sets using sklearn Support Vector Machine, Logistic Regression, and Naive Bayes models,
- Layer Cake: an NLKP solve for the supported data sets using pytorch constructed ATTN, CNN and LSTM based neural network models, and  
- Results Analysis: code to analyze the results and report on them - summary tables and charts effectively


### ML Baseline

The Machine Learning (ML) models were originally designed to baseline the NLP solve as a point of comparison against the neural models. We expanded 
the origianl capabilities in this department to include Logistic Regression (LR) as well as Naive Bayes (NB) models on top of the original Support 
Vector Machine (SVM) model which is quite handy and widely used for text classification problems - as is LR but primarily for binary classification
type problems.

The driver code for this module is in src/ml_class_baselines.py and it is a command line program that takes the following arguments:

usage: ml_class_baselines.py [-h] [--dataset N] [--pickle-dir str] [--log-file N] [--learner N] [--mode N] [--count] [--supervised] [--supervised-method dotn|ppmi|ig|chi] [--optimc] [--embedding-dir str] [--word2vec-path PATH] [--glove-path PATH] [--fasttext-path PATH] [--bert-path PATH] [--llama-path PATH] [--force-embeddings] [--batch-size int] [--force] [--nozscore] [--max-label-space int] [--scoring SCORING]

optional arguments:
  -h, --help            show this help message and exit
  --dataset N           dataset, one in {'reuters21578', 'rcv1', '20newsgroups', 'ohsumed'}
  --pickle-dir str      path where to load the pickled dataset from
  --log-file N          path to the application log file
  --learner N           learner (svm, lr, or nb)
  --mode N              mode, in [tfidf, sup, glove, glove-sup, bert, bert-sup, word2vec, word2vec-sup, fasttext, fasttext-sup, llama, llama-sup]
  --count               use CountVectorizer
  --supervised          use supervised embeddings
  --supervised-method   dotn|ppmi|ig|chi. method used to create the supervised matrix. Available methods include dotn (default), ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)
  --optimc              optimize the C parameter in the SVM
  --embedding-dir str   path where to load and save BERT document embeddings
  --word2vec-path PATH  path + filename to Word2Vec pretrained vectors (e.g. ../.vector_cache/GoogleNews-vectors-negative300.bin), used only with --pretrained word2vec
  --glove-path PATH     directory to pretrained glove embeddings (glove.840B.300d.txt.pt file), used only with --pretrained glove
  --fasttext-path PATH  path + filename to fastText pretrained vectors (e.g. --fasttext-path ../.vector_cache/crawl-300d-2M.vec), used only with --pretrained fasttext
  --bert-path PATH      directory to BERT pretrained vectors, used only with --pretrained bert
  --llama-path PATH     directory to LLaMA pretrained vectors, used only with --pretrained llama
  --force-embeddings    force the computation of embeddings even if a precomputed version is available
  --batch-size int      batch size for computation of BERT document embeddings
  --force               force the execution of the experiment even if a log already exists
  --nozscore            disables z-scoring form the computation of WCE
  --max-label-space int larger dimension allowed for the feature-label embedding (if larger, then PCA with this number of components is applied (default 300)
  --scoring SCORING     scoring parameter to GridSearchCV sklearn call. Must be one of sklearn scoring metricsd.

A sample command may look like the following: 

'python ../src/ml_class_baselines.py --log-file ../log/nb_20newsgroups.test --dataset 20newsgroups --pickle-dir ../pickles --embedding-dir ../.vector_cache --learner nb --mode llama-sup --llama-path ../.vector_cache --optimc'

This runs from the bin directory (which is presumed to be the run location of all scripts and programs) and looks to use the NB model against the 20newsgroups 
dataset, with llama pretrained embeddings as well as word-class embeddings.



### Layer Cake

Layer Cake is the module which tests the NLP solve against the neural models, with ATTN, CNN and LSTM models supported (see related paper or code for details). The
command line program supports the following options:

usage: layer_cake.py [-h] [--dataset str] [--batch-size int] [--batch-size-test int] [--nepochs int] [--patience int] [--plotmode] [--hidden int] [--channels int] [--lr float] [--weight_decay float] [--droptype DROPTYPE] [--dropprob [0.0, 1.0]] [--seed int] [--log-interval int] [--log-file str] [--pickle-dir str] [--test-each int] [--checkpoint-dir str] [--net str] [--pretrained glove|word2vec|fasttext|bert] [--supervised] [--supervised-method dotn|ppmi|ig|chi] [--learnable int] [--val-epochs int] [--word2vec-path PATH] [--glove-path PATH] [--fasttext-path PATH] [--bert-path PATH] [--llama-path PATH] [--max-label-space int] [--max-epoch-length int] [--force] [--tunable] [--nozscore] [--batch-file str]

optional arguments:
  -h, --help            show this help message and exit
  --dataset str         dataset, one in {'20newsgroups', 'ohsumed', 'reuters21578', 'rcv1'}
  --batch-size int      input batch size (default: 100)
  --batch-size-test int batch size for testing (default: 250)
  --nepochs int         number of epochs (default: 100)
  --patience int        patience for early-stop (default: 10)
  --plotmode            in plot mode, executes a long run in order to generate enough data to produce trend plots (test-each should be >0. This mode is used to produce plots, and does not perform a final evaluation on the test set other than those performed after test-each epochs).
  --hidden int          hidden lstm size (default: 512)
  --channels int        number of cnn out-channels (default: 256)
  --lr float            learning rate (default: 1e-3)
  --weight_decay float  weight decay (default: 0)
  --droptype DROPTYPE   chooses the type of dropout to apply after the embedding layer. Default is "sup" which only applies to word-class embeddings (if present). Other options include "none" which does not apply dropout (same as "sup" with no supervised embeddings), "full" which applies dropout to the entire embedding, or "learn" that applies dropout only to the learnable embedding.
  --dropprob [0.0, 1.0] dropout probability (default: 0.5)
  --seed int            random seed (default: 1)
  --log-interval int    how many batches to wait before printing training status
  --log-file str        path to the log csv file
  --pickle-dir str      if set, specifies the path where to save/load the dataset pickled (set to None if you prefer not to retain the pickle file)
  --test-each int       how many epochs to wait before invoking test (default: 0, only at the end)
  --checkpoint-dir str  path to the directory containing checkpoints
  --net str             net, one in {'cnn', 'attn', 'lstm'}
  --pretrained          glove|word2vec|fasttext|bert. pretrained embeddings, use "glove", "word2vec", "fasttext", "bert", or "llama" (default None)
  --supervised          use supervised embeddings
  --supervised-method   dotn|ppmi|ig|chi. method used to create the supervised matrix. Available methods include dotn (default), ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)
  --learnable int       dimension of the learnable embeddings (default 0)
  --val-epochs int      number of training epochs to perform on the validation set once training is over (default 1)
  --word2vec-path PATH  path + filename to Word2Vec pretrained vectors (e.g. ../.vector_cache/GoogleNews-vectors-negative300.bin), used only with --pretrained word2vec
  --glove-path PATH     directory to pretrained glove embeddings (glove.840B.300d.txt.pt file), used only with --pretrained glove
  --fasttext-path PATH  path + filename to fastText pretrained vectors (e.g. --fasttext-path ../.vector_cache/crawl-300d-2M.vec), used only with --pretrained fasttext
  --bert-path PATH      directory to BERT pretrained vectors (e.g. bert-base-uncased-20newsgroups.pkl), used only with --pretrained bert
  --llama-path PATH     directory to LLaMA pretrained vectors, used only with --pretrained llama
  --max-label-space int larger dimension allowed for the feature-label embedding (if larger, then PCA with this number of components is applied (default 300)
  --max-epoch-length intnumber of (batched) training steps before considering an epoch over (None: full epoch)
  --force               do not check if this experiment has already been run
  --tunable             pretrained embeddings are tunable from the beginning (default False, i.e., static)
  --nozscore            disables z-scoring form the computation of WCE
  --batch-file str      path to the config file used for batch processing of multiple experiments





### Results Analysis

The results_analysis module looks at the raw output of either the ML or NN (neural network) model runs and then parses the data to provide a summary
report and/or interactive, html charts that show the relative performance of different pretrained or other embeddings against different data sets and 
in different models. 

The command line program takes the following arguments:

usage: results_analysis.py [-h] [--output_dir OUTPUT_DIR] [-c] [-s] [-d] [--show] file_path

positional arguments:
  file_path             Path to the CSV file with the data

optional arguments:
  -h, --help            show this help message and exit
  --output_dir          OUTPUT_DIR. Directory to save the output files, default is "../out"
  -c, --charts          Generate charts
  -s, --summary         Generate summary
  -d                    debug mode
  --show                Display charts interactively (requires -c)

A sample run might look something like:

'python ../src/results_analysis.py --output_dir ../out -c --show ../log/ML/ml_baselines_newsgroups.test -d'

Again running from the bin directory, specifying the file to be processed (in this case ml_baselines_newsgroups.test in the ../log dircetory), that we are
running in debug mode (-d) and we are generaating charts (-c) which we are both saving (to the output directory ../out) and showing, in a browser. Note 
that to generate a summary file, along with the charts, simply add the -s option to the same command.




## Technical Requirements & Dependencies

This code was developed n both a Mac with Apple silicon (Apple M1 Max) as well as, for the neural models specifically which are GPU dependent, a Ubuntu Linux
host that has CUDA support, the latter of which is a core technical dependency for this code to run - along with the rest of the python and conda requirements 
outlined in setup.sh and requirements.txt. 

The code runs using the latest (as of summer of 2024) python 3.8 libraries. Specifically we use a miniconda python 3.8 environment which is built with the 
following commands (from startup.sh):


-- begin bin/startup.sh:

#!/bin/bash

conda init
conda create -n python38 python=3.8
conda activate python38
pip install scikit-learn fasttext transformers simpletransformers rdflib gensim fasttext matplotlib tabulate scipy datetime numpy pandas psutil GPUtil plotly
conda install pytorch torchtext cudatoolkit=11.8 -c pytorch -c nvidia
apt update
apt install zip

--- end bin/startup.sh


These should be run individually from a command line to make sure they take, and the CUDA library support should be confirmed once complete - if on a CUDA 
supported Linux environment. This can be done by starting a python environment (type 'python' at the command line) and then running the following code:

import torch
torch.cuda.is_available()

This should return True if on a CUDA enabled environment, or False otherwise (for example on my Mac which althogh has GPU support does not support CUDA which 
is a NVIDIA specific solution)



## Directory Tree


### /bin

Shell scripts and other commands.

### /src

All source code

### ./vector_cache

Cache for embeddings and their variants

### /pickles

Cache for datasets after they have been 'pickled'

### /log

Output log file

### /datasets

directory of dataset files


## Supported Data sets

TODO: describe supported data sets here


### BBC News (Single Label)

#### Download

https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive?resource=download

https://www.kaggle.com/competitions/learn-ai-bbc/data?select=BBC+News+Train.csv


#### Context
News article datasets, originating from BBC News, provided for use as benchmarks for machine learning research. The original data is processed to form a single csv 
file for ease of use, the news title and the related text file name is preserved along with the news content and its category. This dataset is made available for 
non-commercial and research purposes only. All rights, including copyright, in the content of the original articles are owned by the BBC.

#### Content
Consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.
Class Labels: 5 (business, entertainment, politics, sport, tech)

#### Acknowledgements
The original source of the data may be accessed through this link and it might be interesting to read the associated research article.

#### Associated Official Research Papers
D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.



### 20newsgroups (Single Label)

The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for training (or development) 
and the other one for testing (or for performance evaluation). The split between the train and test set is based upon a messages posted before 
and after a specific date.

We download the dataset from sklearn using sklearn.datasets.fetch_20newsgroups.

Single Label dataset, 20 categories

target_names: 

['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', \
  'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', \
  'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']





## Pre-Trained Word Embeddings

We use four variants of pre-trained word embeddings to drive and test the various models, and upon which we add - and test - additional embeddings 
(like word-class embeddings or Poincaire embeddings).


The code depends upon word embeddings, pre-trained, being accessible. They are kept in the ./vector_cache directory that sits right off the main dircetory. The following pre-trained embeddings are tested, and can be downloaded from the following URLs (as of June 2024)


### GloVe (Global Vectors for Word Representation)

Model: glove.42B.300d.txt

Dimension: 300

Architecture: GloVe is an unsupervised learning algorithm that generates word embeddings by aggregating global word-word co-occurrence statistics from a corpus. It learns representations by factoring the co-occurrence matrix of words in a corpus. Unlike Word2Vec, GloVe uses a matrix factorization approach rather than a neural network.

Training Data: The referenced GloVe model (glove.42B.300d.txt) was trained on a massive corpus containing 42 billion tokens from Common Crawl. Another popular GloVe model is trained on Wikipedia and Gigaword.

Training Objective: GloVe optimizes the embeddings such that word vectors are learned in a way that their dot products approximate word co-occurrence probabilities.

Pretrained embeddings described, and available for download here: https://nlp.stanford.edu/projects/glove/. 

We use, consistent with (Moreo et al 2019), the glove.840B.300d variant,
which is trained with Common Crawl inpiut - 840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB.  



### Word2Vec

Model: GoogleNews-vectors-negative300.bin

Dimension: 300

Architecture: Word2Vec uses a shallow neural network to create word embeddings. It has two major approaches:
- Skip-gram: Predicts context words given a target word.
- CBOW (Continuous Bag of Words): Predicts the target word based on surrounding context words. Word2Vec treats words as independent units and doesn't consider subword information (which FastText later improved upon).

Training Data: The GoogleNews-vectors-negative300.bin model was trained on part of the Google News dataset, consisting of 100 billion words. It provides 300-dimensional embeddings for 3 million words and phrases.

Training Objective: The training objective is to predict words in context (either target-to-context or context-to-target), effectively learning dense word embeddings based on co-occurrence patterns.

We use GoogleNews-vectors-negative300 (GoogleNews-vectors-negative300.bin) trained on Google News, dimension == 300, but other models are also available from gensim:

['fasttext-wiki-news-subwords-300',
 'conceptnet-numberbatch-17-06-300',
 'word2vec-ruscorpora-300',
 'word2vec-google-news-300',
 'glove-wiki-gigaword-50',
 'glove-wiki-gigaword-100',
 'glove-wiki-gigaword-200',
 'glove-wiki-gigaword-300',
 'glove-twitter-25',
 'glove-twitter-50',
 'glove-twitter-100',
 'glove-twitter-200',
 
Also kaggle download: https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300?resource=downloadn, or can use the unix command: 'wget https://figshare.com/ndownloader/files/10798046 -O GoogleNews-vectors-negative300.bin'

Other available downloads: https://wikipedia2vec.github.io/wikipedia2vec/pretrained/




### FASTTEXT

Model: crawl-300d-2M-subword.bin

Dimension: 300

Architecture: FastText is a shallow neural network embedding model developed by Facebook. It builds on Word2Vec, but its key innovation is the use of subword information (i.e., it learns vectors for character n-grams in addition to full words). This allows FastText to handle rare words or misspellings more robustly than Word2Vec or GloVe.

Training Data: The model referenced here is trained on Common Crawl with 2 million word vectors. FastText models are trained on text corpora that consist of billions of words from across the web.

Training Objective: FastText uses the skip-gram model with negative sampling (like Word2Vec) but augments it by using subwords. This allows the model to predict not just individual words but subword representations.

We use the crawl-300d-2M.vec set of fastText word embeddings which is 2 million word vectors (pre) trained on Common Crawl (600B tokens). These can
be downloaded, along with other English variants, from https://fasttext.cc/docs/en/english-vectors.html, or directly from https://fasttext.cc/docs/en/crawl-vectors.html



### BERT (Bidirectional Encoder Representations from Transformers)

For BERT we use the (English) bert-base-uncased model which can be downloaded here: https://github.com/google-research/bert.

Model: bert-base-cased

Dimension: 768

Architecture: BERT uses a transformer-based architecture with both encoder and decoder layers. The main innovation in BERT is its bidirectional training using the transformer encoder, 
which allows the model to look at both the left and right context of a token in all layers. It is trained on two tasks:

- Masked Language Modeling (MLM): A percentage of words in the input are randomly masked, and the model predicts the missing words.
- Next Sentence Prediction (NSP): The model is trained to predict whether two sentences follow each other in a sequence.

Training Data: BERT is pre-trained on the BooksCorpus (800M words) and English Wikipedia (2.5B words). This combination provides a large amount of general-domain text. The "cased" version 
of BERT means that the text is case sensitive, ie case is not lowered, and it uses a vocabulary that distinguishes case. The "base" version of BERT includes 12 transformer 
blocks (layers), with a hidden size of 768, and 12 self-attention heads, totaling about 110 million parameters. This makes it significantly smaller than the 
"large" version of BERT but still quite powerful.

BERT is trained using two unsupervised tasks:

1) Masked Language Model (MLM): Random tokens are masked out of the input, and the model is trained to predict the original token based on its context.
2) Next Sentence Prediction (NSP): Given pairs of sentences, the model predicts whether the second sentence in a pair is the actual next sentence in the original document.



### RoBERTa (Robustly Optimized BERT Pretraining Approach)

Model: roberta-base

Dimension: 768

Architecture: RoBERTa is a variant of BERT, but it improves on BERT's pretraining process by optimizing the architecture. It removes the NSP task and trains with much larger batch sizes and datasets for longer periods, resulting in improved performance on various benchmarks. Like BERT, it is based on a transformer encoder and uses bidirectional training.

Training Data: RoBERTa is trained on a larger and more diverse dataset compared to BERT. The dataset includes CommonCrawl News, OpenWebText, Stories, and Wikipedia, totaling over 160GB of text, making it significantly more data-rich.

Training Objective: RoBERTa is trained on the MLM task only, focusing on maximizing the model's ability to understand context within a sentence.



### LLAMA (LLaMA 2)

Model: meta-llama/Llama-2-7b-hf or meta-llama/Llama-2-13b-hf

Dimension: 4096

Architecture: LLaMA (Large Language Model Meta AI) is a transformer-based architecture designed to handle large-scale language tasks. It uses a decoder-only architecture, meaning it's similar to GPT models, where the entire input is processed as a sequence, and the model autoregressively predicts the next token. It focuses on efficient scaling and reduced inference costs compared to GPT-3.

Training Data: LLaMA 2 is trained on a diverse set of publicly available text data, including CommonCrawl, Wikipedia, books, research papers, and more. The model has access to hundreds of billions of tokens, making it one of the largest language models to date.

Training Objective: The model is trained using a standard autoregressive language modeling task, which means it predicts the next word in a sequence, given the previous context.




## License & Warranties

Code is licensed as is and is covered by the 'three clause BSD license' as reflected in the LICENSE file which is delivered with the sotfware. Any comments or 
questions should be addresed to pworth2022@fau.edu.






