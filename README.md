# Layer Cake

Benchmark platform for the testing of various models with various embeddings for a standard multi-class (and multi-label multi-calss) classification 
problem using a variety of pretrained word embeddings (GloVe, Word2Vec, FastText and BERT) across a variety of models (SVM, Logistic Regression, and 
CNN, LSTM and ATTN neural models, across several data sets (TODO: list final data sets here) in various combination. Code originally forked from, and 
greatly extended, from build and repo from the paper "Word-Class Embeddings for Multiclass Text Classification", publised in 2019, which studied the 
efficacy of adding supervised, topic based embeddings on top of pretrained embeddings to support multi-class (and multi-label multi-class) classifcation 
problem. 

Word-Class Embeddings (WCEs) are a form of supervised embeddings specially suited for multiclass text classification.
WCEs are meant to be used as extensions (i.e., by concatenation) to pre-trained embeddings (e.g., GloVe or word2vec) embeddings
in order to improve the performance of neural classifiers.

Original paper available https://arxiv.org/abs/1911.11506, original repo available here https://github.com/AlexMoreo/word-class-embeddings


We comnbine these embeddings with Poincare, dictionary trained embeddings (TODO: cite paper and repo here) and eveluate results.




## Technical Requirements & Dependencies

This code was built on a Linux, Ubuntu dist with CUDA GPU support, this is a dependency for the code to work (including setup.sh and other *.sh scripts). We 
use a miniconda python 3.8 environment with the following library dependencies, also mirrored in the startup.sh script in the /bin directory (along with other 
test scripts).


### python library dependencies

Pytorch
torchtext
#Cuda==
Scikit-learn
numpy==1.20.3
scipy==1.12
pandas==1.5
fastText
transformers
simpletransformers
rdflib
gensim
matplotlib
tabulate
datetime



## Directory Tree


### /bin

### /src

### ./vector_cache

### /pickles

### /log


## Supported Data sets

TODO: describe supported data sets here





## Pre-Trained Word Embeddings

We use four variants of pre-trained word embeddings to drive and test the various models, and upon which we add - and test - additional embeddings 
(like word-class embeddings or Poincaire embeddings).


The code depends upon word embeddings, pre-trained, being accessible. They are kept in the ./vector_cache directory that sits right off the main dircetory. The following pre-trained embeddings are tested, and can be downloaded from the following URLs (as of June 2024)



### Word2Vec

We use GoogleNews-vectors-negative300 (trained on Google News, dimension == 300, but other models are also available from gensim:

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



### GloVe

Pretrained embeddings described, and available for download here: https://nlp.stanford.edu/projects/glove/. 

We use, consistent with (Moreo et al 2019), the glove.840B.300d variant,
which is trained with Common Crawl inpiut - 840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB.  




### fastText

We use the crawl-300d-2M.vec set of fastText word embeddings which is 2 million word vectors (pre) trained on Common Crawl (600B tokens). These can
be downloaded, along with other English variants, from https://fasttext.cc/docs/en/english-vectors.html, or directly from https://fasttext.cc/docs/en/crawl-vectors.html



### BERT

For BERT we use the (English) bert-base-uncased model which can be downloaded here: https://github.com/google-research/bert.

The BERT-base-uncased model, developed by researchers at Google AI Language, is based on the BERT (Bidirectional Encoder Representations from Transformers) 
architecture, and is pre-trained on a large corpus that combines two specific text sources:

- BookCorpus: This dataset contains over 11,000 books, covering a diverse range of genres and topics. This corpus is particularly valuable for its broad and rich language usage.
- English Wikipedia: The entirety of the English Wikipedia is used (excluding lists, tables, and headers), which provides a vast range of knowledge across countless subjects.

The "uncased" version of BERT means that the text has been lowercased before tokenization, and it uses a vocabulary that does not distinguish case. This is typically 
beneficial for tasks where the capitalization of words doesn't add useful distinction, and it helps in reducing the model's vocabulary size. The "base" version of BERT 
includes 12 transformer blocks (layers), with a hidden size of 768, and 12 self-attention heads, totaling about 110 million parameters. This makes it significantly smaller 
than the "large" version of BERT but still quite powerful.

BERT is trained using two unsupervised tasks:

1) Masked Language Model (MLM): Random tokens are masked out of the input, and the model is trained to predict the original token based on its context.
2) Next Sentence Prediction (NSP): Given pairs of sentences, the model predicts whether the second sentence in a pair is the actual next sentence in the original document.





## Setup

TODO: Setup instructions 


## Test Run Instructions


## Results Analysis and Reporting



## Adding new embeddings into the test framework




## License & Warranties




