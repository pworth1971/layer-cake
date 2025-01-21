# LAYER CAKE


## Introduction

Layer Cake is a platform for testing the performance of different language models, ie word embeddings, for teh problem of text classification.
Layer cake supports core machine learning models that are effective with text classification such as SVM , Logistic Regression and Naive Bayes, as well
as neural models, ie deep learning architectures, which support CNN, ATTN and LSTM approaches (with some limitations). The platform is designed to
support and test different model and data configurations, which include the tuning and testing of both hyperparametr settings for the various models 
as well as various combinations of embeddings and underlying dataset representations.

The core of the code and modules are forked from a github repo that was designed to test 'word-class embeddings' as described in a 2019 paper 
entitled 'Word-Class Embeddings for Multiclass Text Classification' authored by Alejandro Moreo, Andrea Esuli, and Fabrizio Sebastiani who at 
the time at least were part of the Istituto di Scienza e Tecnologie dell’Informazione Consiglio Nazionale delle Ricerche in Pisa Italy (which 
where the Tower of Pisa is - random fact). Word-Class Embeddings (WCEs) are a form of supervised embeddings specially suited for multiclass 
text classification. WCEs are meant to be used as extensions (i.e., by concatenation) to pre-trained embeddings (e.g., GloVe or word2vec) 
embeddings in order to improve the performance of neural classifiers. This work was outlined in the original paper which is available 
at https://arxiv.org/abs/1911.11506, with the original repo which we forked from available here: https://github.com/AlexMoreo/word-class-embeddings.

We extend this architecture to support fastText word embeddings, as well as Tranformer (token based) langugae models such as BERT, DistilBERT, RoBERTa,
XLNet and GPT2, levergaing the huggingface libraries (transformers) for model creation and testing. In doing so we adapt these WCEs to TCEs, or token 
class embeddings so that the underlying dataset representation data by the model, which uses a model sepcific tokenization strategy, can be properly 
aligned with the TCEs themselves for testing. Our findings are that the TCEs are not effective in this setting but we leave the possibility open that 
there is an alternative way to add them that could possiby be effective.


## Platform Overview

The platform consists of three modules:

- ML Baselines: an NLP solve for the supported data sets using sklearn Support Vector Machine, Logistic Regression, and Naive Bayes models,
- Layer Cake: an NLKP solve for the supported data sets using pytorch constructed ATTN, CNN and LSTM based neural network models, and  
- Results Analysis: code to analyze the results and report on them - summary tables and charts effectively


### ML Baseline: SVM, LR and NB Models

The Machine Learning (ML) models were originally designed to baseline the NLP solve as a point of comparison against the neural models. We expanded 
the origianl capabilities in this department to include Logistic Regression (LR) as well as Naive Bayes (NB) models on top of the original Support 
Vector Machine (SVM) model which is quite handy and widely used for text classification problems - as is LR but primarily for binary classification
type problems.

The driver code for this module is in src/ml_classification_test_v3.0.py and it is a command line program that takes the following arguments:



options:
  -h, --help                  show this help message and exit
  --dataset DATASET           dataset, one in ['20newsgroups', 'rcv1', 'reuters21578', 'bbc-news', 'ohsumed', 'imdb', 'arxiv', 'arxiv_protoformer']
  --pickle-dir DIR            path where to load the pickled dataset from
  --logfile FILE              path to the application log file
  --net MODEL                 model / learner, one of [svm, lr, or nb]
  --vtype VTYPE               dataset base vectorization strategy, in [tfidf, count]
  --pretrained MODEL          Language model to use, either "None" or in ['glove', 'word2vec', 'fasttext', 'bert', 'roberta', 'distilbert', 'xlnet', 'gpt2'], default None.
  --dataset-emb-comp POOLING  Pooling strategy for teh computation of dataset representation for transformer models. Eitehr "avg", or "summary" (cls), defaults to "avg".
  --embedding-dir DIR         path where to load and save embeddings
  --seed SEED                 random seed (default: 29)
  --mix MIX                   way to prepare the embeddings, in [vmode, cat, cat-wce, dot, dot-wce, solo, solo-wce, lsa, lsa-wce]. NB presumes --pretrained is set
  --cm                        create confusion matrix for underlying model
  --optimc                    optimize the model using relevant models params
  --force                     force the execution of the experiment even if a log already exists
  --supervised                Use supervised embeddings (TCEs
  --nozscore                  disables z-scoring form the computation of TCE
  --supervised-method         method used to create the supervised matrix, one of 'dotn', 'ppmi', 'ig', 'chi', default is dotn, [ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)]
  --max-label-space int       larger dimension allowed for the feature-label embedding (if larger, then PCA with this number of components is applied (default 300)


A sample command may look like the following: 

'python ../src/ml_class_baselines.py --log-file ../log/nb_20newsgroups.test --dataset 20newsgroups --pretrained bert --pickle-dir ../pickles --embedding-dir ../.vector_cache --net nb --mix vmode --seed 27 --optimc'

This runs from the bin directory (which is presumed to be the run location of all scripts and programs) and looks to use the NB model against the 20newsgroups dataset, using the bert language mdoel, caching data in ../pickels dirctory, using the 'vmode' mix (tfidf representation). The mixes are summarized as follows, note that the pooling strategy used by the engine to compute the document (dataset) representation is determined by the '--dataset-emb-comp' parameter, which defaults to 'avg', aka mean pooling.

- 'solo': uses the model embedding representation alone
- 'solo-wce': uses the model embedding representation concatenated with the WCEs / TCEs
- 'cat-doc': the vectorized text representation (either tfidf or count) concatenated with the embedding model representation
- 'cat-wce': the vectorized dataset representation (either tfidf or count) combined with the WCEs / TCEs
- 'cat-doc-wce': the vectorized dataset representation with the embedding model reprsentation and the WCEs / TCEs
- 'dot': projection of the vectorized doc / dataset representation into the embedding space using dot product operation
- 'dot-wce': projection of the vectorized doc / dataset representation into the embedding + WCE/TCE space using dot product operation
- 'lsa': using Latent Semantic Analysis approch, i,e LSA/SVD computation, for feature selection.
- 'lsa-wce': concatenation of LSA factorization of dataset with vectorized representation of dataset with WCE / TCEs

The script that was used to generate test data is in bin/ml_testing.ml and it covers most of the options needed to benchmark the three models across the supported datasets.



### Layer Cake: Static Word Embedding Models (GloVe, Word2Vec and fastText)

Layer Cake is the module which tests the NLP solve against static word embedding models (GloVe, Word2Vec and fastText) using three different neural architcetures: 
ATTN, CNN and LSTM models supported (see related paper or code for details). The command line program supports the following options:


(python312) peter@Peters-MacBook-Pro-M1-2021 bin % python ../src/wrd_layer_cake_v2.5.py --help   

	--- WORD_LAYER_CAKE Version: 2.5 ---
available_datasets: {'bbc-news', 'reuters21578', 'ohsumed', '20newsgroups', 'rcv1', 'arxiv', 'imdb', 'arxiv_protoformer'}

usage: 
wrd_layer_cake_v2.5.py [-h] [--dataset str] [--show-dist] [--vtype N] [--batch-size int] [--batch-size-test int] [--nepochs int] [--patience int] [--plotmode] [--hidden int] [--channels int] [--lr float] [--weight_decay float] [--droptype DROPTYPE] [--dropprob [0.0, 1.0]] [--seed int] [--log-interval int] [--log-file str] [--pickle-dir str] [--test-each int] [--checkpoint-dir str] [--net str] [--pretrained glove|word2vec|fasttext] [--supervised] [--sup-mode SUP_MODE] [--supervised-method dotn|ppmi|ig|chi] [--learnable int] [--val-epochs int] [--glove-path PATH] [--word2vec-path PATH] [--fasttext-path PATH] [--bert-path PATH] [--max-label-space int] [--max-epoch-length int] [--force] [--tunable] [--nozscore]


Neural text classification with Static Word Embeddings

options:
  -h, --help            show this help message and exit
  --dataset str         dataset, one in {'bbc-news', 'reuters21578', 'ohsumed', '20newsgroups', 'rcv1', 'arxiv', 'imdb', 'arxiv_protoformer'}
  --show-dist           Show dataset class distribution
  --vtype N             dataset base vectorization strategy, in [tfidf, count]
  --batch-size int      input batch size (default: 100)
  --batch-size-test int
                        batch size for testing (default: 250)
  --nepochs int         number of epochs (default: 200)
  --patience int        patience for early-stop (default: 10)
  --plotmode            in plot mode, executes a long run in order to generate enough data to produce trend plots (test-each should be >0. This mode is used to produce plots, and does not perform a final evaluation on the test set other than those performed after test-each epochs).
  --hidden int          hidden lstm size (default: 512)
  --channels int        number of cnn out-channels (default: 256)
  --lr float            learning rate (default: 1e-3)
  --weight_decay float  weight decay (default: 0)
  --droptype DROPTYPE   chooses the type of dropout to apply after the embedding layer. Default is "sup" which only applies to word-class embeddings (if present). Other options include "none" which does not apply dropout (same as "sup" with no supervised embeddings), "full" which applies dropout to the entire embedding, or
                        "learn" that applies dropout only to the learnable embedding.
  --dropprob [0.0, 1.0]
                        dropout probability (default: 0.5)
  --seed int            random seed (default: 29)
  --log-interval int    how many batches to wait before printing training status
  --log-file str        path to the log csv file
  --pickle-dir str      if set, specifies the path where to save/load the dataset pickled (set to None if you prefer not to retain the pickle file)
  --test-each int       how many epochs to wait before invoking test (default: 0, only at the end)
  --checkpoint-dir str  path to the directory containing checkpoints
  --net str             net, one in {'attn', 'cnn', 'lstm'}
  --pretrained glove|word2vec|fasttext
                        pretrained embeddings, use "glove", "word2vec", or "fasttext" (default None)
  --supervised          use supervised embeddings
  --sup-mode SUP_MODE   How WCEs are combined with model embeddings (cat)
  --supervised-method dotn|ppmi|ig|chi
                        method used to create the supervised matrix. Available methods include dotn (default), ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)
  --learnable int       dimension of the learnable embeddings (default 0)
  --val-epochs int      number of training epochs to perform on the validation set once training is over (default 1)
  --glove-path PATH     path to glove.840B.300d pretrained vectors (used only with --pretrained glove)
  --word2vec-path PATH  path to GoogleNews-vectors-negative300.bin pretrained vectors (used only with --pretrained word2vec)
  --fasttext-path PATH  path to crawl-300d-2M.vec pretrained vectors (used only with --pretrained fasttext)
  --bert-path PATH      Directory to BERT pretrained vectors, defaults to ../.vector_cache/BERT. Used only with --pretrained bert
  --max-label-space int
                        larger dimension allowed for the feature-label embedding (if larger, then PCA with this number of components is applied (default 300)
  --max-epoch-length int
                        number of (batched) training steps before considering an epoch over (None: full epoch)
  --force               do not check if this experiment has already been run
  --tunable             pretrained embeddings are tunable from the beginning (default False, i.e., static)
  --nozscore            disables z-scoring form the computation of WCE


NB: The script used to generate the data across the supported models, options and datasets can be found in bin/nn_test_legacy.sh



### Layer Cake: Dynamic Word Embedding Models, i.e. Transformers

We support BERT, RoBERTa, DistilBERT, XLNet and GPT2 base models across the same available datasets. The code is encapsulated in the 
trans_layer_cake python script which takes the following arguments:

(python312) peter@Peters-MacBook-Pro-M1-2021 bin % python ../src/trans_layer_cake_v13.5.py --help 

	--- TRANS_LAYER_CAKE Version: 13.5 ---

usage: 
trans_layer_cake_v13.5.py [-h] --dataset {20newsgroups,rcv1,reuters21578,bbc-news,ohsumed,imdb,arxiv,arxiv_protoformer} [--show-dist] [--vtype N] [--pretrained {bert,roberta,distilbert,xlnet,gpt2}] [--seed SEED] [--log-file LOG_FILE] [--force] [--weighted] --net str [--class-model] [--lr LR] [--weight_decay WEIGHT_DECAY] [--epochs EPOCHS] [--patience PATIENCE] [--dropprob [0.0, 1.0]] [--learnable int] [--droptype DROPTYPE] [--tunable] [--channels int] [--hidden int] [--pooling POOLING] [--supervised] [--sup-mode SUP_MODE] [--nozscore] [--supervised-method dotn|ppmi|ig|chi] [--max-label-space int] [--normalize-tces] [--tunable-tces] [--tce-weight-init [0.0, 1.0]]


Transformer Layer Cake: Text Classification with Transformer Models.

options:
  -h, --help            show this help message and exit
  --dataset {20newsgroups,rcv1,reuters21578,bbc-news,ohsumed,imdb,arxiv,arxiv_protoformer}
                        Dataset to use
  --show-dist           Show dataset class distribution
  --vtype N             dataset base vectorization strategy, in [tfidf, count]
  --pretrained {bert,roberta,distilbert,xlnet,gpt2}
                        supported language model types for dataset representation
  --seed SEED           Random seed
  --log-file LOG_FILE   Path to log file
  --force               do not check if this experiment has already been run
  --weighted            Whether or not to use class_weights in the loss funtion of the classifier (default False)
  --net str             supported models, either cnn, linear, hf.sc, (defaults to linear)
  --class-model         use simple transformer classifier if True, else use the HF Sequence Classifier (customizede) model.
  --lr LR               Learning rate
  --weight_decay WEIGHT_DECAY
                        Weight decay
  --epochs EPOCHS       Number of epochs
  --patience PATIENCE   Patience for early stopping
  --dropprob [0.0, 1.0]
                        dropout probability for classifier head (default: 0.5)
  --learnable int       dimension of the learnable embeddings (default 0)
  --droptype DROPTYPE   chooses the type of dropout to apply after the embedding layer. Default is "sup" which only applies to word-class embeddings (if present). Other options include "none" which does not apply dropout (same as "sup" with no supervised embeddings), "full" which applies dropout to the entire embedding, or
                        "learn" that applies dropout only to the learnable embedding.
  --tunable             Whether or not to set pretrained embeddings (as well as classifier layers), are tunable. Default False, i.e., static)
  --channels int        number of cnn out-channels (default: 256)
  --hidden int          hidden lstm size (default: 512)
  --pooling POOLING     type of pooling method, used only with --net linear. Default is "mean"
  --supervised          Use supervised embeddings (TCEs
  --sup-mode SUP_MODE   How to combine TCEs with model embeddings (in ['cat', 'add', 'dot'])
  --nozscore            disables z-scoring form the computation of TCE
  --supervised-method dotn|ppmi|ig|chi
                        method used to create the supervised matrix. Available methods include dotn (default), ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)
  --max-label-space int
                        larger dimension allowed for the feature-label embedding (if larger, then PCA with this number of components is applied (default 300)
  --normalize-tces      Whether or not to normalize the tce_matrix values to the std and mean of the model embeddings. Default False.
  --tunable-tces        whether or not to have the tce_matrix be tunable during training. Default False.
  --tce-weight-init [0.0, 1.0]
                        initialized tce embedding layer weight, used with --supervised and --tunable-tces. Default: 1.0

We have two models that are tested HF SEQ, Hugging Face Sequence Classifier, and a CNN variant which also leverages HuggingFace transformers, each of these
is driven and tested by the bin/trans_lc_hfseq_test.sh and bin/trans_lc_hfcnn_test.sh scripts which test the different variants of datasets and options for the
different models.



### Logging and Results Analysis

Each of the modules described above outputs results to log files, typically generated in the log/ directory which have a lot of information about the underlying 
system and program parameters, and model definitions, of each run. These log files are then in turn analyzed various charts and summary output capabilities are
provided by the src/results_analysis.py code. The results_analysis module looks at the raw output of either the ML or NN (neural network) model runs and then parses 
the data to provide a summary report and/or interactive, html charts that show the relative performance of different pretrained or other embeddings against 
different data sets and in different models. 


	----- Results Analysis -----
usage: results_analysis.py [-h] [-c] [-r] [-n] [-s] [-o] [-d] [-y YSTART] [-m RESULTS] [-show] file_path

Analyze model results and generate charts and/or summaries

positional arguments:
  file_path             Path to the TSV file with the data

options:
  -h, --help            show this help message and exit
  -c, --charts          Generate charts
  -r, --runtimes        Generate timrlapse charts
  -n, --neural          Output from Neural Nets
  -s, --summary         Generate summary
  -o, --output_dir      Directory to write output files. If not provided, defaults to ../out/ + the base file name of the input file.
  -d, --debug           debug mode
  -y YSTART, --ystart YSTART
                        Y-axis starting value for the charts (default: 0.6)
  -m RESULTS, --results RESULTS
                        Y-axis starting value for the charts (default: 0.6)
  -show                 Display charts interactively (requires -c)


A sample run might look something like:

python ../src/results_analysis.py ../log/lc_ml_test.test --charts --runtimes --summary --debug

But shell scripts for ML and NN model analysis are provided in the bin directory for reference.

Output defaults to /out directory and includes charts which show model and dataset performance (png, csv and html), as well as 
summary files in CSV and HTML output, 



## Technical Requirements & Dependencies

This code was developed n both a Mac with Apple silicon (Apple M1 and M3 chips) as well as, for the neural models specifically which are GPU dependent, a Ubuntu Linux
host that has CUDA support, the latter of which is a core technical dependency for this code to run - along with the rest of the python and conda requirements 
outlined in setup.sh and requirements.txt. 

The code runs using the latest (as of winter of 2025) python 3.12 libraries. Specifically we use a miniconda python 3.12 environment which is built 
with the bin/startup.sh script which can be run from a command line from a new system. The startup shell script runs on both Apple MacOS and Ubuntu Linux.



## Supported Data sets

Layer Cake is designed to combine pretrained embeddings of different types with different datasets across the text classification spectrum, with support
for a range of datasets that cover news, medical, and review type data, some of which are multi-label datasets where a given doc in a given dataset can
belong to multiple classes, or single-label data where a given doc for a given dataset can belong to just one class or label.

The following datasets are supported, each of which is generally available for research purposes but in some cases (like RCV1) must be requested
specifically from the provider.

As of the time of this writing the datasets we support are outlined below, each of which has its own source that must be downloaded and 
configured in the proper /datasets subdirectory.




#### BBC NEWS 
Description: A dataset consisting of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.
Classes: 5 (e.g., Business, Entertainment, Politics, Sport, Tech)
Type: Single-label classification
Size: Approximately 2,225 docs, 6.7MB
License: Typically used for educational and research purposes, though the specific license terms are not detailed on the download page.
Misc: good for testing as the relative size is small

#### REUTERS21578
Description: One of the most commonly used datasets for text categorization. It contains thousands of documents categorized into multiple classes which makes it a multi-label dataset.
Classes: 115
Type: Multi-label classification
Size: Roughly 21,578 docs, 64MB
Access: Reuters-21578 on UCI
License: Free for research purposes, but usage in commercial projects should be checked with Reuters.
Misc: Classes are not very well balanced and this causes problems with sone of the f1 summary data for some models

#### 20NEWSGROUPS
Description: A collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. We download the 
dataset from sklearn using sklearn.datasets.fetch_20newsgroups.
Classes: 20 (various topics such as sports, religion, hardware, etc.)
Type: Single-label classification
Size: About 20,000 docs, 15MB
Access: Available via Scikit-Learn's dataset utilities or 20 Newsgroups
License: Public domain
Misc: good baseline single-label test case, very well benchmarked

#### ARXIV
Description: A dataset derived from ArXiv papers, typically used for categorizing scientific papers into multiple classes based on their subjects.
Classes: 58 (scientific fields)
Type: Multi-label classification
Size: 5.5GB
License: Depends on the specifics of data usage; generally, data used for academic research without redistribution is allowed.
Misc: Special preprocessing requirements due to nature of underlying docs.

#### ARXIV PROTOFORMER
Description: A potentially derivative dataset from the ArXiv collection focusing on a smaller subset of topics or a specific preprocessing pipeline.
Classes: 10 (subset or specific topics within the broader ArXiv classification)
Type: Single-label classification
Size: 147 MB
Access and License: Likely a custom dataset; access and licensing would depend on the creator's setup or the project specifications.

#### OHSUMED
Description: A subset of the MEDLINE database, which is a bibliographic database of important, peer-reviewed medical literature maintained by the National Library of Medicine.
Classes: 23 (medical subject headings)
Type: Multi-label classification
Size: Approximately 348,000 citations (abstracts), 387 MB
Access: OHSUMED on UCI
License: Generally used for academic and research purposes; specific licensing terms would need to be confirmed.

#### IMDB
Description: A dataset for binary sentiment classification consisting of movie reviews from the IMDb site labeled as positive or negative.
Classes: 2 (Positive, Negative)
Type: Single-label classification
Size: 694 MB
Access: IMDb Reviews Dataset
License: For non-commercial use only.

#### RCV1
Description: An archive of over 800,000 manually categorized newswire stories made available by Reuters, Ltd. for research purposes.
Classes: 101 (various topics)
Type: Multi-label classification
Size: Over 800,000 docs, 7.4 GB
License: Available for research purposes; usage beyond this scope should be confirmed with the distributor (ie Reuters).
Misc: Very large dataset, requires significant compute power. Good for testing scalaability of models and underlying representation.






## Language Models in Scope

We use both static word embeddings as part of our testing (GloVe, Word2Vec and fastText) as well as dynamic embeddings, i.e. transformers (BERT, RoBERTa, DistilBERT, XLNet and 
GPT2). The code depends upon word embeddings, pre-trained, being accessible. They are kept in the ./vector_cache directory that sits right off the main dircetory. The following 
pre-trained embeddings are tested, and can be downloaded from the following URLs (as of June 2024)



#### GloVe (Global Vectors for Word Representation)
Pretrained embeddings described, and available for download here: https://nlp.stanford.edu/projects/glove/. We use, consistent with (Moreo et al 2019), the glove.840B.300d variant, which is 
trained with Common Crawl inpiut - 840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB.  

Model in use: GloVe 840B 300d

Architecture: GloVe is an unsupervised learning algorithm for obtaining vector representations for words by aggregating global word-word co-occurrence statistics from a corpus.

Training Data: The model is trained on 840 billion tokens from a dataset aggregated from web data (Common Crawl).

Salient Features: Each word is represented by a 300-dimensional vector. The model captures both semantic and syntactic information of words.

Reference: Pennington, Jeffrey, et al. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.
Training Objective: GloVe optimizes the embeddings such that word vectors are learned in a way that their dot products approximate word co-occurrence probabilities.



#### Word2Vec
Model in use: GoogleNews-vectors-negative300, trained on Google News, dimension == 300, but other models are also available from gensim:

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

Architecture: Word2Vec is a group of related models used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct 
linguistic contexts of words. Word2Vec uses a shallow neural network to create word embeddings. It has two major approaches:

Training Data: The GoogleNews-vectors-negative300.bin model was trained on part of the Google News dataset, consisting of 100 billion words. It provides 300-dimensional 
embeddings for 3 million words and phrases.

Training Objective: The training objective is to predict words in context (either target-to-context or context-to-target), effectively learning dense word 
embeddings based on co-occurrence patterns.

Salient Features: The model uses 300-dimensional vectors and is case-sensitive.

Reference: Mikolov, Tomas, et al. "Efficient Estimation of Word Representations in Vector Space." ICLR Workshop Papers. 2013.


#### FastText

Model in use: crawl-300d-2M-subword.bin

We use the crawl-300d-2M.vec set of fastText word embeddings which is 2 million word vectors (pre) trained on Common Crawl (600B tokens). These can
be downloaded, along with other English variants, from https://fasttext.cc/docs/en/english-vectors.html, or directly from https://fasttext.cc/docs/en/crawl-vectors.html

Dimension: 300

Architecture: FastText extends Word2Vec to consider subword information (character n-grams), allowing it to generate word embeddings for out-of-vocabulary words.
FastText is a shallow neural network embedding model developed by Facebook. It builds on Word2Vec, but its key innovation is the use of subword information (i.e., it 
learns vectors for character n-grams in addition to full words). This allows FastText to handle rare words or misspellings more robustly than Word2Vec or GloVe.

Training Data: The model referenced here is trained on Common Crawl with 2 million word vectors. FastText models are trained on text corpora that consist of billions of words from across 
the web. Trained on Common Crawl and Wikipedia using CBOW with position-weights, with character n-grams of length 5, a window of size 5, and 10 negatives.

Training Objective: FastText uses the skip-gram model with negative sampling (like Word2Vec) but augments it by using subwords. This allows the model to predict not just individual words but subword representations.

Salient Features: Produces 300-dimensional vectors, supports 157 languages, and is case-insensitive.

Reference: Bojanowski, Piotr, et al. "Enriching Word Vectors with Subword Information." Transactions of the Association for Computational Linguistics 5 (2017): 135-146.



#### BERT (Bidirectional Encoder Representations from Transformers)

For BERT we use the (English) bert-base-uncased model which can be downloaded here: https://github.com/google-research/bert.

Model in use: bert-base-uncased

Dimension: 768

Architecture: BERT uses a transformer-based architecture with both encoder and decoder layers. The main innovation in BERT is its bidirectional training using the transformer encoder, 
which allows the model to look at both the left and right context of a token in all layers. Architecture: BERT is a transformer-based model known for its deep bidirectionality, which allows 
it to contextually understand both the left and right context in all layers. It is trained on two tasks:

- Masked Language Modeling (MLM): A percentage of words in the input are randomly masked, and the model predicts the missing words.
- Next Sentence Prediction (NSP): The model is trained to predict whether two sentences follow each other in a sequence.

Training Data: BERT is pre-trained on the BooksCorpus (800M words) and English Wikipedia (2.5B words). This combination provides a large amount of general-domain text. The "cased" version 
of BERT means that the text is case sensitive, ie case is not lowered, and it uses a vocabulary that distinguishes case. The "base" version of BERT includes 12 transformer 
blocks (layers), with a hidden size of 768, and 12 self-attention heads, totaling about 110 million parameters. This makes it significantly smaller than the "large" version of 
BERT but still quite powerful.

BERT is trained using two unsupervised tasks:

1) Masked Language Model (MLM): Random tokens are masked out of the input, and the model is trained to predict the original token based on its context.
2) Next Sentence Prediction (NSP): Given pairs of sentences, the model predicts whether the second sentence in a pair is the actual next sentence in the original document.

Salient Features: The base model uses 12 layers (transformer blocks), has 768 hidden units, 12 heads, and is case-insensitive.

Reference: Devlin, Jacob, et al. "BERT: Pre-training of deep bidirectional transformers for language understanding." NAACL HLT 2019.


#### RoBERTa (Robustly Optimized BERT Approach)
Model in use: roberta-base

Dimension: 768

Architecture: RoBERTa is a variant of BERT, but it improves on BERT's pretraining process by optimizing the architecture. It removes the NSP task and trains with much larger batch 
sizes and datasets for longer periods, resulting in improved performance on various benchmarks. Like BERT, it is based on a transformer encoder and uses bidirectional training. RoBERTa 
iterates on BERT's architecture by modifying key hyperparameters, removing the next-sentence pretraining objective, and training with much larger mini-batches and learning rates.

Training Data: RoBERTa is trained on a larger and more diverse dataset compared to BERT. The dataset includes CommonCrawl News, OpenWebText, Stories, and Wikipedia, totaling over 
160GB of text, making it significantly more data-rich.

Training Objective: RoBERTa is trained on the MLM task only, focusing on maximizing the model's ability to understand context within a sentence.

Salient Features: Uses the same model size as BERT-base and is case-sensitive.

Reference: Liu, Yinhan, et al. "RoBERTa: A robustly optimized BERT pretraining approach." arXiv preprint arXiv:1907.11692 (2019).


#### DistilBERT
Model in use: distilbert-base-uncased

Architecture: DistilBERT is a smaller, faster, cheaper, and lighter version of BERT. It distills 40% of the size of BERT while retaining 97% of its performance, using a technique called knowledge distillation.

Training Data: Same as BERT.

Salient Features: Uses 6 layers, with 768 hidden units and is case-insensitive.

Reference: Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019).


#### XLNet
Model in use: xlnet-base-cased

Architecture: XLNet incorporates ideas from Transformer-XL, the state-of-the-art autoregressive model, into the pretraining objective of BERT by maximizing the expected likelihood over all permutations of the input sequence tokens.

Training Data: Trained on a larger corpus that includes BooksCorpus, English Wikipedia, Giga5, ClueWeb, and Common Crawl.

Salient Features: Uses 12 layers, with 768 hidden units and is case-sensitive.

Reference: Yang, Zhilin, et al. "XLNet: Generalized Autoregressive Pretraining for Language Understanding." NeurIPS 2019.


#### GPT-2 (Generative Pre-trained Transformer 2)
Model in use: gpt2

Architecture: GPT-2 is an unsupervised language model that uses the Transformer architecture. It generates synthetic text samples in response to the input text.

Training Data: Trained on a dataset called "WebText," a corpus consisting of over 8 million documents (40GB of text) scraped from the internet.

Salient Features: The base model has 12 layers, 768 hidden units, and is case-sensitive.

Reference: Radford, Alec, et al. "Language Models are Unsupervised Multitask Learners." OpenAI Blog (2019).



## License & Warranties

Code is licensed as is and is covered by the 'three clause BSD license' as reflected in the LICENSE file which is delivered with the sotfware. Any comments or 
questions should be addresed to pworth2022@fau.edu.






