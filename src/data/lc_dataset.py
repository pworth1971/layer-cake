
"""
Dataset loading and managemeht routines to support Layer Cake ML model testing
#
# LCDataset class to support dataset management for Layer Cake ML Model processing 
# which includes word2vec, glove and fasttext, as well as BERT, RoBERTa, DistilBERT, XLNet, and GPT2 
# transformer models. The ML models are trained with scikit-learn libraries 
# 
#
"""

import os
import numpy as np
from tqdm import tqdm
import re
import pandas as pd
import string
from pathlib import Path
from urllib import request
import tarfile
import gzip
import shutil
import pickle


import torch

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from joblib import Parallel, delayed


#
# custom imports
#
from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1
from data.arxiv_reader import fetch_arxiv

from model.LCRepresentationModel import *

from data.lc_trans_dataset import _label_matrix, RANDOM_SEED
from util.common import preprocess


nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))



# ----------------------------------------------------------------------------------------------------------------------------------
#
# Constants
#
SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news", "ohsumed", "imdb", "arxiv", "arxiv_protoformer"]

DATASET_DIR = '../datasets/'                        # dataset directory

#
# Disable Hugging Face tokenizers parallelism to avoid fork issues
#
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MIN_DF_COUNT = 5                    # minimum document frequency count for a term to be included in the vocabulary
TEST_SIZE = 0.175                   # test size for train/test split
NUM_DL_WORKERS = 3                  # number of workers to handle DataLoader tasks
# ----------------------------------------------------------------------------------------------------------------------------------



class LCDataset:

    """
    A class to handle loading and preparing datasets for text classification.
    Supports multiple datasets including reuters21578, 20newsgroups, ohsumed, rcv1 (and imdb and arxiv coming).
    """

    dataset_available = SUPPORTED_DATASETS
    
    single_label_datasets = ['20newsgroups', 'bbc-news', 'arxiv_protoformer', 'imdb']
    multi_label_datasets = ['rcv1', 'reuters21578', 'ohsumed', 'arxiv']
    
    def __init__(self, 
                 name, 
                 vectorization_type='tfidf', 
                 pretrained=None, 
                 embedding_type='word', 
                 embedding_path=VECTOR_CACHE, 
                 embedding_comp_type='avg', 
                 seed=RANDOM_SEED):
        """
        Initializes the LCDataset object with the specified dataset and vectorization parameters. This method
        both loads the dataset into the respective LCDataset variables as well as sets up the proper model and 
        vectorizer objects and then vectorizes both the training (Xtr) and test (Xte) data using said vectorizer.
        NB that the training and test data is split in the relevant load_ method for the dataset

        - name: Name of the dataset to load.
        - vectorization_type: 'tfidf' or 'count', determines which vectorizer to use for tokenization.
        - pretrained: type of pretraiend emebddings (eg glove, word2vec, bert, etc)
        - embedding_type: 'word' for word-based embeddings (GloVe, Word2Vec, fastText) or 'token' for token-based models (BERT, RoBERTA, etc).
        - embedding_path: Path to the pretrained embeddings.
        - embedding_comp_type: 'avg' or 'weighted' for the type of embedding composition to use.
        - seed: Random seed for reproducibility.
        """

        print("\n\tinitializing LCDataset...")

        assert name in LCDataset.dataset_available, f'dataset {name} is not available'

        print(f"name: {name}, vectorization_type: {vectorization_type}, pretrained: {pretrained}, embedding_type: {embedding_type}, embedding_path: {embedding_path}, embedding_comp_type: {embedding_comp_type}, seed: {seed}")

        self.name = name
        self.vectorization_type = vectorization_type
        self.pretrained = pretrained
        self.embedding_type = embedding_type
        self.embedding_path = embedding_path
        self.pretrained_path = embedding_path
        self.embedding_comp_type = embedding_comp_type
        
        self.seed = seed
        
        self.loaded = False
        self.initialized = False

        # Setup device prioritizing CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.batch_size = DEFAULT_GPU_BATCH_SIZE
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.batch_size = DEFAULT_MPS_BATCH_SIZE
        else:
            self.device = torch.device("cpu")
            self.batch_size = DEFAULT_CPU_BATCH_SIZE

        print("device:", self.device)
        print("batch_size:", self.batch_size)

        #
        # load and preprocess the dataset
        #
        if name=='reuters21578':
            self._load_reuters()
        elif name == '20newsgroups':
            self._load_20news()
        elif name == 'rcv1':
            self._load_rcv1()
        elif name == 'ohsumed':
            self._load_ohsumed()
        elif name == 'bbc-news':
            self._load_bbc_news()
        elif name == 'imdb':
            self._load_imdb()
        elif name == 'arxiv':
            self._load_arxiv()
        elif name == 'arxiv_protoformer':
            self._load_arxiv_protoformer()
        else:
            raise ValueError(f"Dataset {name} not supported")
        
        #
        # instantiate representation model if token/transformer based 
        # we offload this code to another class
        #
        self.lcr_model = None          # custom representation model class object
    
        #
        # load LCRepresentation class
        #
        if (pretrained == 'word2vec'):
            print("Using Word2Vec language model embeddings...")
            
            self.lcr_model = Word2VecLCRepresentationModel(
                model_name=WORD2VEC_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )
        elif (pretrained == 'glove'):
            print("Using GloVe language model embeddings...")
            
            self.lcr_model = GloVeLCRepresentationModel(
                model_name=GLOVE_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )
        
        elif (pretrained == 'hyperbolic'):
            print("Using Hyperbolic dictionary model embeddings...")
            
            self.lcr_model = HyperbolicLCRepresentationModel(
                model_name=HYPERBOLIC_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )

        elif (pretrained == 'fasttext'):
            print("Using FastText language model embeddings...")

            self.lcr_model = FastTextGensimLCRepresentationModel(
                model_name=FASTTEXT_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )

        elif (pretrained == 'bert'):
            print("Using BERT language model embeddings...")

            self.lcr_model = BERTLCRepresentationModel(
                model_name=BERT_MODEL, 
                model_dir=embedding_path,
                vtype=vectorization_type
            )
        
        elif (pretrained == 'roberta'):
            print("Using RoBERTa language model embeddings...")

            self.lcr_model = RoBERTaLCRepresentationModel(
                model_name=ROBERTA_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )

        elif (pretrained == 'distilbert'):
            print("Using DistilBERT language model embeddings...")

            self.lcr_model = DistilBERTLCRepresentationModel(
                model_name=DISTILBERT_MODEL, 
                model_dir=embedding_path,
                vtype=vectorization_type
            )

        elif (pretrained == 'xlnet'):
            print("Using XLNet language model embeddings...")

            self.lcr_model = XLNetLCRepresentationModel(
                model_name=XLNET_MODEL, 
                model_dir=embedding_path,  
                vtype=vectorization_type
            )

        elif (pretrained == 'gpt2'):
            print("Using GPT2 langauge model embeddings...")

            self.lcr_model = GPT2LCRepresentationModel(
                model_name=GPT2_MODEL, 
                model_dir=embedding_path,  
                vtype=vectorization_type
            )

        elif (pretrained == 'llama'):
            print("Using Llama langauge model embeddings...")

            self.lcr_model = LlamaLCRepresentationModel(
                model_name=LLAMA_MODEL, 
                model_dir=embedding_path,  
                vtype=vectorization_type
            )

        elif (pretrained == 'deepseek'):
            print("Using DeepSeek language model embeddings...")

            self.lcr_model = DeepSeekLCRepresentationModel(
                model_name=DEEPSEEK_MODEL, 
                model_dir=embedding_path,  
                vtype=vectorization_type
            )
            
        else:

            print("Warning: pretrained not set, defaulting to GloVe pretrained embeddings...")
            
            self.lcr_model = GloVeLCRepresentationModel(
                model_name=GLOVE_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )

        #
        # initialize model, tokenizer and vectorizer
        #
        self.model = self.lcr_model.model
        self.max_length = self.lcr_model.max_length
        self.tokenizer = self.lcr_model.tokenizer           # note only transformer models have tokenizers
        self.vectorizer = self.lcr_model.vectorizer

        self.nC = self.num_labels
        print("nC:", self.nC)

        self.type = self.lcr_model.type
        print("type:", self.type)

        self.loaded = True
        self.initialized = True

        print("LCDataset initialized...\n")


    # Function to remove stopwords before tokenization
    def remove_stopwords(self, texts):
        
        print("removing stopwords...")
        
        filtered_texts = []
        for text in texts:
            filtered_words = [word for word in text.split() if word.lower() not in stop_words]
            filtered_texts.append(" ".join(filtered_words))
        return filtered_texts

    

    def vectorize(self, debug=False):
        """
        Build vector representation of data set using TF-IDF or CountVectorizer and constructing 
        the embeddings such that they align with pretrained embeddings tokenization method
        
        NB: We have a few different variants of the vectorizer vocabulary that we need, that seem
        to work in different situations, in particular with respect to BERT and LlaMa embeddings 
        when we perform the dot product operation to project the tfidf vectorized text into the pretrained 
        (dictionary / vocabulary) embedding space.
        
        In particular the TfidfVectorizer.get_feature_names_out() method from Scikit-Learn's TfidfVectorizer class 
        returns an array of feature names, which represent the terms (words or tokens) extracted from the input text 
        that the vectorizer was trained on. These feature names correspond to the columns of the matrix produced when 
        transforming text data into TF-IDF features. The array includes each term that appears in the vocabulary 
        after fitting the TfidfVectorizer on data. The terms are the individual words or n-grams (depending on the 
        configuration of the vectorizer) found during the text analysis. The terms are returned in the order of their 
        corresponding columns in the TF-IDF matrix. This order is determined by the internal sorting of the vocabulary, 
        which is typically alphabetical. For example, if the vocabulary contains the words "apple", "banana", and 
        "cherry", they will appear in that order. This method is useful for mapping the matrix columns back to their 
        respective terms. For instance, if you have a TF-IDF matrix where each row represents a document and each 
        column represents a term's TF-IDF score, get_feature_names_out() will tell you which term each column 
        corresponds to.
        """

        print("\tvectorizing dataset...")    

        print("model:\n", self.lcr_model)
        print("tokenizer:\n", self.tokenizer)
        print("vectorizer:\n", self.vectorizer)
        print("self.vectorization_type:", self.vectorization_type)
        print("self.embedding_type:", self.embedding_type)
        
        print("fitting training and test data with vectorizer...")

        # Fit and transform the text data
        self.Xtr_vectorized = self.vectorizer.fit_transform(self.Xtr)
        self.Xte_vectorized = self.vectorizer.transform(self.Xte)

        self.Xtr_vectorized.sort_indices()
        self.Xte_vectorized.sort_indices()

        print("Xtr_vectorized:", type(self.Xtr_vectorized), self.Xtr_vectorized.shape)
        print("Xte_vectorized:", type(self.Xte_vectorized), self.Xte_vectorized.shape)
        
        self.vocab_ = self.vectorizer.vocabulary_
        print("vocab_:", type(self.vocab_), len(self.vocab_))

        self.vocab_ndarr = self.vectorizer.get_feature_names_out()
        print("vocab_ndarr:", type(self.vocab_ndarr), len(self.vocab_ndarr))

        self.vocab = self.vocab_
        print("vocab:", type(self.vocab), len(self.vocab))

        self.vocabulary = self.vectorizer.vocabulary_
        print("vocabulary:", type(self.vocabulary), len(self.vocabulary))

        # Ensure X_vectorized is a sparse matrix (in case of word-based embeddings)
        if not isinstance(self.Xtr_vectorized, csr_matrix):
            self.Xtr_vectorized = csr_matrix(self.Xtr_vectorized)

        if not isinstance(self.Xte_vectorized, csr_matrix):
            self.Xte_vectorized = csr_matrix(self.Xte_vectorized)

        return self.Xtr_vectorized, self.Xte_vectorized
    

    def init_embedding_matrices(self):
        """
        Initialize the dataset with pretrained embeddings.
        """

        print("initializing embedding matrices...")
        
        self.pretrained_path = self.embedding_path
        print("self.pretrained:", self.pretrained)
        print("self.pretrained_path:", self.pretrained_path)

        # build the embedding vocabulary matrix to align with the dataset vocabulary and embedding type
    
        print("\n\tconstructing (pretrained) embeddings dataset vocabulary matrix...")    
        
        self.embedding_vocab_matrix = self.lcr_model.build_embedding_vocab_matrix()
           
        print("self.embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        #print(self.embedding_vocab_matrix)

        # generate pretrained embedding representation of dataset
        self.generate_dataset_embeddings()

        self.initialized = True

        #return self.embedding_vocab_matrix, self.token_to_index_mapping
        return self.embedding_vocab_matrix
    

    # -------------------------------------------------------------------------------------------------------------

    def generate_dataset_embeddings(self):
        """
        Generate embedding representation of dataset docs in three forms for both the training preprocessed data (Xtr)
        and the test preprocessed data (Xte). The three forms are:

        - summary_embeddings: CLS token embeddings from BERT or RoBERTa models.
        - avg_embeddings: Average of token embeddings from BERT or RoBERTa models.
        - weighted_embeddings: Weighted average of token embeddings using TF-IDF weights.

        Returns:
        - Xtr_summary_embeddings: CLS token embeddings for training data.
        - Xte_summary_embeddings: CLS token embeddings for test data.
        - Xtr_avg_embeddings: Average embeddings for training data.
        - Xte_avg_embeddings: Average embeddings for test data.
        """
        
        print("generating dataset embedding representation forms...")
        
        print("self.pretrained:", self.pretrained)
        print("self.pretrained_path:", self.pretrained_path)

        # generate dataset embedding representations depending on underlyinbg 
        # pretrained embedding language model - transformaer based and then word based
        if (self.pretrained in ['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2', 'llama', 'deepseek']):      

            print("generating token (transformer) based dataset representations...")

            self.Xtr_avg_embeddings, self.Xtr_summary_embeddings = self.lcr_model.encode_docs(self.Xtr)                                  
            self.Xte_avg_embeddings, self.Xte_summary_embeddings = self.lcr_model.encode_docs(self.Xte)

            # not supported weighted average comp method for transformer based models due to
            # complexity of vectorization and tokenization mapping across models 
            self.Xtr_weighted_embeddings = self.Xtr_avg_embeddings
            self.Xte_weighted_embeddings = self.Xte_avg_embeddings
            
        elif (self.pretrained in ['word2vec', 'glove', 'hyperbolic']):                        # word based embeddings
            
            print("generating word based dataset representations...")

            self.Xtr_weighted_embeddings, self.Xtr_avg_embeddings = self.lcr_model.encode_docs(
                self.Xtr, 
                self.embedding_vocab_matrix
            )
            
            self.Xte_weighted_embeddings, self.Xte_avg_embeddings = self.lcr_model.encode_docs(
                self.Xte, 
                self.embedding_vocab_matrix
            )

            # CLS token summary embeddings not supported in pretrained 
            # word embedding models like word2vec, GloVe or fasTtext
            self.Xtr_summary_embeddings = self.Xtr_avg_embeddings
            self.Xte_summary_embeddings = self.Xte_avg_embeddings
        
        elif (self.pretrained in ['fasttext']):                        # subword based embeddings
            
            print("generating subword based dataset representations...")

            self.Xtr_weighted_embeddings, self.Xtr_avg_embeddings = self.lcr_model.encode_docs(self.Xtr)
            self.Xte_weighted_embeddings, self.Xte_avg_embeddings = self.lcr_model.encode_docs(self.Xte)

            # CLS token summary embeddings not supported in pretrained 
            # word embedding models like word2vec, GloVe or fasTtext
            self.Xtr_summary_embeddings = self.Xtr_avg_embeddings
            self.Xte_summary_embeddings = self.Xte_avg_embeddings
        
        else:
            print("generating default (GloVe) based dataset representations...")

            self.Xtr_weighted_embeddings, self.Xtr_avg_embeddings = self.lcr_model.encode_docs(
                self.Xtr, 
                self.embedding_vocab_matrix
            )
            
            self.Xte_weighted_embeddings, self.Xte_avg_embeddings = self.lcr_model.encode_docs(
                self.Xte, 
                self.embedding_vocab_matrix
            )

            # CLS token summary embeddings not supported in pretrained 
            # word embedding models like word2vec, GloVe or fasTtext
            self.Xtr_summary_embeddings = self.Xtr_avg_embeddings
            self.Xte_summary_embeddings = self.Xte_avg_embeddings

        print("Xtr_avg_embeddings:", type(self.Xtr_avg_embeddings), self.Xtr_avg_embeddings.shape)
        print("Xte_avg_embeddings:", type(self.Xte_avg_embeddings), self.Xte_avg_embeddings.shape)

        print("self.Xtr_weighted_embeddings:", type(self.Xtr_weighted_embeddings), self.Xtr_weighted_embeddings.shape)
        print("self.Xte_weighted_embeddings:", type(self.Xte_weighted_embeddings), self.Xte_weighted_embeddings.shape)

        print("Xtr_summary_embeddings:", type(self.Xtr_summary_embeddings), self.Xtr_summary_embeddings.shape)
        print("Xte_summary_embeddings:", type(self.Xte_summary_embeddings), self.Xte_summary_embeddings.shape)
        
        return self.Xtr_weighted_embeddings, self.Xte_weighted_embeddings, self.Xtr_avg_embeddings, self.Xte_avg_embeddings, self.Xtr_summary_embeddings, self.Xte_summary_embeddings
    


    def get_initialized_neural_data(self):

        return self.word2index, self.out_of_vocabulary, self.unk_index, self.pad_index, self.devel_index, self.test_index


    def init_neural_data(self):

        """
        Indexes the dataset using either a tokenizer (for Transformer models) or a pretrained word-based embedding model.
        """

        print(f"init_neural_data()...")

        # For word-based models, use the dataset's vocabulary. For Transformer models, use the tokenizer's vocabulary.
        """
        if self.tokenizer:
            self.word2index = dict(self.tokenizer.get_vocab())
            self.unk_index = self.tokenizer.unk_token_id
            self.pad_index = self.tokenizer.pad_token_id
        else:
        """

        # build the vocabulary from the dataset and 
        # the pretrained language model (if not None)
        self.word2index = dict(self.vocabulary)                 # dataset vocabulary
        known_words = set(self.word2index.keys())

        print("self.word2index:", type(self.word2index), len(self.word2index))
        print("known_words:", type(known_words), len(known_words))
        
        if self.lcr_model is not None:
            print("self.lcr_model:\n: ", self.lcr_model)
            #print("self.lcr_model.vocabulary():\n: ", self.lcr_model.vocabulary())
            known_words.update(self.lcr_model.vocabulary())         # polymorphic behavior

        print("known_words:", type(known_words), len(known_words))
        
        self.word2index['UNKTOKEN'] = len(self.word2index)
        self.word2index['PADTOKEN'] = len(self.word2index)
        self.unk_index = self.word2index['UNKTOKEN']
        self.pad_index = self.word2index['PADTOKEN']

        self.out_of_vocabulary = dict()
        analyzer = self.analyzer()

        """
        # Define a helper function to tokenize and index documents
        def tokenize_and_index(documents):
            indices = []
            for doc in documents:
                if self.tokenizer:
                    # Use the transformer tokenizer for subword tokenization
                    tokens = self.tokenizer.encode(doc, truncation=True, padding=True, max_length=self.max_length)
                else:
                    # Use the dataset analyzer for word-based tokenization
                    tokens = self.analyzer()(doc)

                # Convert tokens to indices, handle OOVs
                indexed_tokens = [self.word2index.get(token, self.unk_index) for token in tokens]
                indices.append(indexed_tokens)
            return indices

        # Index development and test sets
        self.devel_index = tokenize_and_index(self.devel_raw)
        self.test_index = tokenize_and_index(self.test_raw)
        """

        self.devel_index = self.index(self.devel_raw, self.word2index, known_words, analyzer, self.unk_index, self.out_of_vocabulary)
        self.test_index = self.index(self.test_raw, self.word2index, known_words, analyzer, self.unk_index, self.out_of_vocabulary)

        print('[indexing complete]')

        return self.word2index, self.out_of_vocabulary, self.unk_index, self.pad_index, self.devel_index, self.test_index


    def index(self, data, vocab, known_words, analyzer, unk_index, out_of_vocabulary):
        """
        Index (i.e., replaces word strings with numerical indexes) a list of string documents
        
        :param data: list of string documents
        :param vocab: a fixed mapping [str]->[int] of words to indexes
        :param known_words: a set of known words (e.g., words that, despite not being included in the vocab, can be retained
        because they are anyway contained in a pre-trained embedding set that we know in advance)
        :param analyzer: the preprocessor in charge of transforming the document string into a chain of string words
        :param unk_index: the index of the 'unknown token', i.e., a symbol that characterizes all words that we cannot keep
        :param out_of_vocabulary: an incremental mapping [str]->[int] of words to indexes that will index all those words that
        are not in the original vocab but that are in the known_words
        :return:
        """
        indexes=[]
        vocabsize = len(vocab)
        unk_count = 0
        knw_count = 0
        out_count = 0
        pbar = tqdm(data, desc=f'indexing documents')
        for text in pbar:
            words = analyzer(text)
            index = []
            for word in words:
                if word in vocab:
                    idx = vocab[word]
                else:
                    if word in known_words:
                        if word not in out_of_vocabulary:
                            out_of_vocabulary[word] = vocabsize+len(out_of_vocabulary)
                        idx = out_of_vocabulary[word]
                        out_count += 1
                    else:
                        idx = unk_index
                        unk_count += 1
                index.append(idx)
            indexes.append(index)
            knw_count += len(index)
            pbar.set_description(f'[unk = {unk_count}/{knw_count}={(100.*unk_count/knw_count):.2f}%]'
                                f'[out = {out_count}/{knw_count}={(100.*out_count/knw_count):.2f}%]')
        return indexes



    def split_val_data(self, val_ratio=0.2, min_samples=20000, seed=1):
        """
        Split the validation data off of the training data (this is not cached in the pickle file).
        NB, must be called after init_neural_data()

        arguments:
            val_ratio: float, ratio of validation data to training data
            min: int, minimum number of samples in each class       
            seed: int, random seed for train_test_split

        returns:
            X_train, X_val, y_train, y_val: numpy arrays of training and validation data
        """

        print("split_val_data()...")

        val_size = min(int(len(self.devel_index) * val_ratio), min_samples)                   # dataset split tr/val/test
        X_train, X_val, y_train, y_val = train_test_split(
            self.devel_index, 
            self.devel_target, 
            test_size=val_size, 
            random_state=seed, 
            shuffle=True
        )

        return X_train, X_val, y_train, y_val

    #
    # -------------------------------------------------------------------------------------------------------------
    # data loading
    # -------------------------------------------------------------------------------------------------------------
    #


    # ------------------------------------------------------------------------------------------------------------------------
    # load_bbc_news()
    #
    # Load BBC NEWS dataset
    # ------------------------------------------------------------------------------------------------------------------------
    def _load_bbc_news(self):
        """
        Load and preprocess the BBC News dataset and set up the LCDataset data to be used by caller, namely 
        Xtr, Xte, y_sparse_train, y_sparse_test, the dataset vocabulary, and other relevant information 
        about the dataset needed by caller that comesd from the processed data
        
        Returns:
        - self.target_names: list iof categories 
        """    
     
        print(f'\n\tloading BBC News dataset from {DATASET_DIR}...')

        for dirname, _, filenames in os.walk(DATASET_DIR + 'bbc-news'):
            for filename in filenames:
                print(os.path.join(dirname, filename))

        self.classification_type = 'singlelabel'
        self.class_type = 'singlelabel'

        # Load datasets
        train_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Train.csv')
        #test_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Test.csv')

        print("train_set:", train_set.shape)
        print("train_set columns:\n", train_set.columns)
        # print("train_set:\n", train_set.head())
        
        print("Unique Categories:\n", train_set['Category'].unique())
        numCats = len(train_set['Category'].unique())
        print("# of categories:", numCats)
        
        # we split the train data into train and test here because that is
        # not done for us with the BBC News dataset (Test data is not labeled)
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            train_set['Text'], 
            train_set['Category'], 
            train_size = 1-TEST_SIZE, 
            random_state = self.seed
        )

        # reset indeces
        self.X_train_raw = self.X_train_raw.reset_index(drop=True)
        self.X_test_raw = self.X_test_raw.reset_index(drop=True)

        #
        # inspect the raw text
        #
        print("\t--- unprocessed text ---")
        print("self.X_train_raw:", type(self.X_train_raw), self.X_train_raw.shape)
        print("self.X_train_raw[0]:\n", self.X_train_raw[0])

        print("self.X_test_raw:", type(self.X_test_raw), self.X_test_raw.shape)
        print("self.X_test_raw[0]:\n", self.X_test_raw[0])

        #
        # preprocess: we remove stopwords, mask numbers and remove punctuation
        # if we are working with word based embeddings (fastText, Word2Vec, GloVe)
        #
        """
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = self._preprocess(pd.Series(self.X_train_raw), remove_punctuation=True)
            self.Xte = self._preprocess(pd.Series(self.X_test_raw), remove_punctuation=True)
        else:
            self.Xtr = self._preprocess(pd.Series(self.X_train_raw), remove_punctuation=False)
            self.Xte = self._preprocess(pd.Series(self.X_test_raw), remove_punctuation=False)
        """

        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = preprocess(
                pd.Series(self.X_train_raw), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(self.X_test_raw), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )    
        else:
            self.Xtr = preprocess(
                pd.Series(self.X_train_raw), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(self.X_test_raw), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            
        print("\t--- preprocessed text ---")
        print("self.Xtr:", type(self.Xtr), len(self.Xtr))
        print("self.Xtr[0]:\n", self.Xtr[0])
        
        print("self.Xte:", type(self.Xte), len(self.Xte))
        print("self.Xte[0]:\n", self.Xte[0])

        self.devel_raw = self.Xtr
        self.test_raw = self.Xte

        # Convert target labels to 1D arrays
        self.devel_target = np.array(self.y_train)  # Flattening the training labels into a 1D array
        self.test_target = np.array(self.y_test)    # Flattening the test labels into a 1D array

        # Use LabelEncoder to encode the labels into label IDs
        label_encoder = LabelEncoder()
        label_encoder.fit(self.devel_target)  # Fit on training labels

        # Transform labels to numeric IDs
        self.devel_target = label_encoder.transform(self.devel_target)
        self.test_target = label_encoder.transform(self.test_target)

        # Pass these reshaped arrays to the _label_matrix method
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1, 1), self.test_target.reshape(-1, 1))      
        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        print("self.labels:\n", self.labels)

        # Save the original label names (classes)
        self.target_names = label_encoder.classes_
        print("self.target_names (original labels):\n", self.target_names)
        
        #self.target_names = train_set['Category'].unique()       
        self.label_names = self.target_names           # set self.labels to the class label names   
        print("self.label_names:\n", self.label_names)

        self.labels = self.label_names
        self.num_label_names = len(self.label_names)
        self.num_labels = self.num_label_names
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        self.y_train_sparse = self.devel_labelmatrix
        self.y_test_sparse = self.test_labelmatrix
        print("self.y_train_sparse:", type(self.y_train_sparse), self.y_train_sparse.shape)
        print("self.y_test_sparse:", type(self.y_test_sparse), self.y_test_sparse.shape)

        # encode the one-hot encoded label array into a 1d array with label
        # Assuming self.y_train_sparse is a sparse matrix with one-hot encoding
        y_train_dense = self.y_train_sparse.toarray()
        y_test_dense = self.y_test_sparse.toarray()

        # Convert the one-hot encoded rows to class indices (assuming single-label per row)
        y_train_flat = np.argmax(y_train_dense, axis=1)
        y_test_flat = np.argmax(y_test_dense, axis=1)

        # Apply LabelEncoder to the flattened 1D array
        label_encoder = LabelEncoder()
        self.ytr_encoded = label_encoder.fit_transform(y_train_flat)
        self.yte_encoded = label_encoder.transform(y_test_flat)

        #print("label_encoder.classes_:", label_encoder.classes_)

        print("self.ytr_encoded:", type(self.ytr_encoded), self.ytr_encoded.shape)
        print("self.yte_encoded:", type(self.yte_encoded), self.yte_encoded.shape)

        return self.target_names


    def _load_20news(self):
        
        print("\n\tloading 20newsgroups dataset...")
        
        metadata = ('headers', 'footers', 'quotes')
        
        self.devel = fetch_20newsgroups(subset='train', remove=metadata)
        self.test = fetch_20newsgroups(subset='test', remove=metadata)

        self.classification_type = 'singlelabel'
        self.class_type = 'singlelabel'
        
        #
        # inspect the raw text
        #
        print("self.devel.data:", type(self.devel.data), len(self.devel.data))
        print("self.devel.data[0]:\n", type(self.devel.data[0]), self.devel.data[0])

        print("self.test.data:", type(self.test.data), len(self.test.data))
        print("self.test.data[0]:\n", type(self.test.data[0]), self.test.data[0])

        #
        # preprocess: we remove stopwords, mask numbers and remove punctuation
        # if we are working with word based embeddings (fastText, Word2Vec, GloVe)
        #
        """
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=True)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=True)
        else:
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=False)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=False)
        """
        
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = preprocess(
                pd.Series(self.devel.data), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(self.test.data), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )    
        else:
            self.Xtr = preprocess(
                pd.Series(self.devel.data), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(self.test.data), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            
        print("...preprocessed text...")
        print("self.Xtr:", type(self.Xtr), len(self.Xtr))
        print("self.Xtr[0]:\n", self.Xtr[0])
        
        print("self.Xte:", type(self.Xte), len(self.Xte))
        print("self.Xte[0]:\n", self.Xte[0])

        self.devel_raw = self.Xtr
        self.test_raw = self.Xte
                
        self.devel_target, self.test_target = self.devel.target, self.test.target        
        print("devel_target:", type(self.devel_target), len(self.devel_target))
        print("test_target:", type(self.test_target), len(self.test_target))

        print("encoded devel_target:", self.devel_target)
        print("encoded test_target:", self.test_target)

        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))
        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        print("self.labels:", type(self.labels), len(self.labels))

        self.label_names = self.devel.target_names           # set self.labels to the class label names
        print("self.label_names:\n", self.label_names) 

        self.target_names = self.label_names
        print("self.target_names:", self.target_names)

        self.num_labels = len(self.labels)
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        self.y_train_sparse = self.devel_labelmatrix
        self.y_test_sparse = self.test_labelmatrix

        print("self.y_train_sparse:", type(self.y_train_sparse), self.y_train_sparse.shape)
        print("self.y_test_sparse:", type(self.y_test_sparse), self.y_test_sparse.shape)

        # encode the one-hot encoded label array into a 1d array with label
        # Assuming self.y_train_sparse is a sparse matrix with one-hot encoding
        y_train_dense = self.y_train_sparse.toarray()
        y_test_dense = self.y_test_sparse.toarray()

        # Convert the one-hot encoded rows to class indices (assuming single-label per row)
        y_train_flat = np.argmax(y_train_dense, axis=1)
        y_test_flat = np.argmax(y_test_dense, axis=1)

        # Apply LabelEncoder to the flattened 1D array
        label_encoder = LabelEncoder()
        self.ytr_encoded = label_encoder.fit_transform(y_train_flat)
        self.yte_encoded = label_encoder.transform(y_test_flat)

        #print("label_encoder.classes_:", label_encoder.classes_)

        print("self.ytr_encoded:", type(self.ytr_encoded), self.ytr_encoded.shape)
        print("self.yte_encoded:", type(self.yte_encoded), self.yte_encoded.shape)

        return self.target_names


    def _load_imdb(self):
        
        print("\n\tloading IMDB dataset...")
        
        from datasets import load_dataset
        import os

        self.classification_type = 'singlelabel'
        self.class_type = 'singlelabel'

        data_path = os.path.join(DATASET_DIR, 'imdb')

        # Load IMDB dataset using the Hugging Face Datasets library
        imdb_dataset = load_dataset('imdb', cache_dir=data_path)

        # Split dataset into training and test data

        train_data = imdb_dataset['train']['text']
        train_target = np.array(imdb_dataset['train']['label'], dtype=np.int64)  # Convert to numpy array of type int64

        test_data = imdb_dataset['test']['text']
        test_target = np.array(imdb_dataset['test']['label'], dtype=np.int64)  # Convert to numpy array of type int64

        # Define target names
        target_names = ['negative', 'positive']
        num_classes = len(target_names)
    
        #
        # preprocess: we remove stopwords, mask numbers and remove punctuation
        # if we are working with word based embeddings (fastText, Word2Vec, GloVe)
        #
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = preprocess(
                pd.Series(train_data), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(test_data), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )    
        else:
            self.Xtr = preprocess(
                pd.Series(train_data), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(test_data), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            
        print("...preprocessed text...")
        print("self.Xtr:", type(self.Xtr), len(self.Xtr))
        print("self.Xtr[0]:\n", self.Xtr[0])
        
        print("self.Xte:", type(self.Xte), len(self.Xte))
        print("self.Xte[0]:\n", self.Xte[0])

        self.devel_raw = self.Xtr
        self.test_raw = self.Xte
                
        self.devel_target, self.test_target = train_target, test_target        
        print("self.devel_target:", type(self.devel_target), len(self.devel_target))
        print("self.test_target:", type(self.test_target), len(self.test_target))

        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))
        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        print("self.labels:", type(self.labels), len(self.labels))

        self.label_names = target_names           # set self.labels to the class label names
        print("self.label_names:\n", self.label_names) 

        self.target_names = self.label_names
        print("self.target_names:", self.target_names)

        self.num_labels = len(self.labels)
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        self.y_train_sparse = self.devel_labelmatrix
        self.y_test_sparse = self.test_labelmatrix

        print("self.y_train_sparse:", type(self.y_train_sparse), self.y_train_sparse.shape)
        print("self.y_test_sparse:", type(self.y_test_sparse), self.y_test_sparse.shape)

        # encode the one-hot encoded label array into a 1d array with label
        # Assuming self.y_train_sparse is a sparse matrix with one-hot encoding
        y_train_dense = self.y_train_sparse.toarray()
        y_test_dense = self.y_test_sparse.toarray()

        # Convert the one-hot encoded rows to class indices (assuming single-label per row)
        y_train_flat = np.argmax(y_train_dense, axis=1)
        y_test_flat = np.argmax(y_test_dense, axis=1)

        # Apply LabelEncoder to the flattened 1D array
        label_encoder = LabelEncoder()
        self.ytr_encoded = label_encoder.fit_transform(y_train_flat)
        self.yte_encoded = label_encoder.transform(y_test_flat)

        #print("label_encoder.classes_:", label_encoder.classes_)

        print("self.ytr_encoded:", type(self.ytr_encoded), self.ytr_encoded.shape)
        print("self.yte_encoded:", type(self.yte_encoded), self.yte_encoded.shape)

        return self.target_names
    

    def _load_arxiv(self):

        print("\n\tloading arxiv dataset...")

        import os

        data_path = os.path.join(DATASET_DIR, 'arxiv')
        
        print("data_path:", data_path)  
    
        self.classification_type = 'multilabel'
        self.class_type = 'multilabel'

        if (self.embedding_type in ['word', 'subword', 'sub-word']):

            # 
            # word based models (remove punctuation and stop words)
            #             
            self.Xtr, ytrain, self.Xte, ytest, target_names, num_classes = fetch_arxiv(
                                                                                data_path=data_path, 
                                                                                test_size=TEST_SIZE, 
                                                                                seed=self.seed,
                                                                                static=True, 
                                                                                array=True)

        else:
            #
            # token based models (leave pucntiation and stop words)
            #
            self.Xtr, ytrain, self.Xte, ytest, target_names, num_classes = fetch_arxiv(
                                                                                data_path=data_path, 
                                                                                test_size=TEST_SIZE, 
                                                                                seed=self.seed,
                                                                                static=False, 
                                                                                array=True)

        self.devel_raw = self.Xtr
        self.test_raw = self.Xte
        
        #self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(ytrain, ytest)
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = ytrain, ytest, target_names

        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        print("labels:\n", self.labels)

        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix
        print("devel_target:", type(self.devel_target), self.devel_target.shape)
        print("test_target:", type(self.test_target), self.test_target.shape)

        self.label_names = target_names                       # Set labels to class label names
        print("self.label_names:\n", self.label_names)
        
        self.target_names = self.label_names
        print("target_names:", type(self.target_names), len(self.target_names))

        self.num_labels = num_classes
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        # Now self.devel_target is already a dense NumPy array 
        # with shape (9603, 115), so no need for MultiLabelBinarizer.

        self.y_train = self.devel_target                                    # Transform multi-label targets into a binary matrix
        self.y_test = self.test_target                                      # Transform multi-label targets into a binary matrix
        print("self.y_train:", type(self.y_train), self.y_train.shape)
        print("self.y_test:", type(self.y_test), self.y_test.shape)
        
        # Convert Y to a sparse matrix
        self.y_train_sparse = csr_matrix(self.y_train)                                       # without Transpose to match the expected shape
        self.y_test_sparse = csr_matrix(self.y_test)                                         # without Transpose to match the expected shape
        print("self.y_test_sparse:", type(self.y_test_sparse), self.y_test_sparse.shape)
        print("self.y_train_sparse:", type(self.y_train_sparse), self.y_train_sparse.shape)

        self.ytr_encoded = self.y_train
        self.yte_encoded = self.y_test
        print("self.ytr_encoded:", type(self.ytr_encoded), self.ytr_encoded.shape)
        print("self.yte_encoded:", type(self.yte_encoded), self.yte_encoded.shape)

        return self.label_names
    

    def _load_arxiv_protoformer(self):

        print("\n\tloading arxiv_protoformer dataset...")

        import os

        self.classification_type = 'singlelabel'
        self.class_type = 'singlelabel'
        
        #
        # dataset from https://paperswithcode.com/dataset/arxiv-10
        #
        data_path = os.path.join(DATASET_DIR, 'arxiv_protoformer')

        file_path = data_path + '/arxiv100.csv'
        print("file_path:", file_path)

        # Load datasets
        full_data_set = pd.read_csv(file_path)
        
        target_names = full_data_set['label'].unique()
        
        """
        self.num_classes = len(full_data_set['label'].unique())
        print(f"self.num_classes: {len(self.target_names)}")
        print("self.target_names:", self.target_names)
        """

        papers_dataframe = pd.DataFrame({
            'title': full_data_set['title'],
            'abstract': full_data_set['abstract'],
            'label': full_data_set['label']
        })

        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())

        print("proeprocessing...")
        """

        # preprocess text
        #papers_dataframe['abstract'] = papers_dataframe['abstract'].apply(lambda x: x.replace("\n",""))
        #papers_dataframe['abstract'] = papers_dataframe['abstract'].apply(lambda x: x.strip())
        papers_dataframe['text'] = papers_dataframe['title'] + '. ' + papers_dataframe['abstract']

        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())
        """

        # Ensure the 'categories' column value counts are calculated and indexed properly
        categories_counts = papers_dataframe['label'].value_counts().reset_index(name="count")

        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            # 
            # word based models (remove punctuation and stop words)
            #             
            papers_dataframe['text'] = preprocess(
                papers_dataframe['text'],
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                remove_special_chars=True,          # we do this for arxiv data
                array=True
            )

        else:
            #
            # token based models (leave punctiation and stop words)
            #
            papers_dataframe['text'] = preprocess(
                papers_dataframe['text'],
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                remove_special_chars=True,           # we do this for arxiv data
                array=True                           # return as array
            )
        
        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())        
        """

        # we split the train data into train and test here because that is
        # not done for us with the BBC News dataset (Test data is not labeled)
        self.Xtr, self.Xte, self.y_train, self.y_test = train_test_split(
            papers_dataframe['text'], 
            papers_dataframe['label'], 
            train_size = 1-TEST_SIZE, 
            random_state = self.seed,
        )
        
        self.devel_raw = self.Xtr
        self.test_raw = self.Xte

        # Convert target labels to 1D arrays
        self.devel_target = np.array(self.y_train)  # Flattening the training labels into a 1D array
        self.test_target = np.array(self.y_test)    # Flattening the test labels into a 1D array

        # Use LabelEncoder to encode the labels into label IDs
        label_encoder = LabelEncoder()
        label_encoder.fit(self.devel_target)  # Fit on training labels

        # Transform labels to numeric IDs
        self.devel_target = label_encoder.transform(self.devel_target)
        self.test_target = label_encoder.transform(self.test_target)

        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))
        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        print("self.labels:", type(self.labels), len(self.labels))

        # Save the original label names (classes)
        self.target_names = label_encoder.classes_
        print("self.target_names (original labels):\n", self.target_names)
        
        #self.target_names = train_set['Category'].unique()       
        self.label_names = self.target_names           # set self.labels to the class label names   
        print("self.label_names:\n", self.label_names)

        self.labels = self.label_names
        self.num_labels = len(self.labels)
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        self.y_train_sparse = self.devel_labelmatrix
        self.y_test_sparse = self.test_labelmatrix
        print("self.y_train_sparse:", type(self.y_train_sparse), self.y_train_sparse.shape)
        print("self.y_test_sparse:", type(self.y_test_sparse), self.y_test_sparse.shape)

        # encode the one-hot encoded label array into a 1d array with label
        # Assuming self.y_train_sparse is a sparse matrix with one-hot encoding
        y_train_dense = self.y_train_sparse.toarray()
        y_test_dense = self.y_test_sparse.toarray()

        # Convert the one-hot encoded rows to class indices (assuming single-label per row)
        y_train_flat = np.argmax(y_train_dense, axis=1)
        y_test_flat = np.argmax(y_test_dense, axis=1)

        # Apply LabelEncoder to the flattened 1D array
        label_encoder = LabelEncoder()
        self.ytr_encoded = label_encoder.fit_transform(y_train_flat)
        self.yte_encoded = label_encoder.transform(y_test_flat)

        #print("label_encoder.classes_:", label_encoder.classes_)

        print("self.ytr_encoded:", type(self.ytr_encoded), self.ytr_encoded.shape)
        print("self.yte_encoded:", type(self.yte_encoded), self.yte_encoded.shape)

        return self.target_names
    

    def _load_reuters(self):

        print("\n\tloading reuters21578 dataset...")

        data_path = os.path.join(DATASET_DIR, 'reuters21578')
        
        print("data_path:", data_path)  

        self.devel = fetch_reuters21578(subset='train', data_path=data_path)
        self.test = fetch_reuters21578(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.class_type = 'multilabel'

        #
        # inspect the raw text
        #
        print("self.devel.data:", type(self.devel.data), len(self.devel.data))
        print("self.devel.data[0]:\n", type(self.devel.data[0]), self.devel.data[0])

        print("self.test.data:", type(self.test.data), len(self.test.data))
        print("self.test.data[0]:\n", type(self.test.data[0]), self.test.data[0])

        #
        # preprocess: we remove stopwords, mask numbers and remove punctuation
        # if we are working with word based embeddings (fastText, Word2Vec, GloVe)
        #
        """
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=True)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=True)
        else:
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=False)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=False)
        """

        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = preprocess(
                pd.Series(self.devel.data), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(self.test.data), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )    
        else:
            self.Xtr = preprocess(
                pd.Series(self.devel.data), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(self.test.data), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            
        print("...preprocessed text...")
        print("self.Xtr:", type(self.Xtr), len(self.Xtr))
        print("self.Xtr[0]:\n", self.Xtr[0])
        
        print("self.Xte:", type(self.Xte), len(self.Xte))
        print("self.Xte[0]:\n", self.Xte[0])
        
        self.devel_raw = self.Xtr
        self.test_raw = self.Xte
        
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel.target, self.test.target)
        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        print("labels:\n", self.labels)

        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix
        print("devel_target:", type(self.devel_target), self.devel_target.shape)
        print("test_target:", type(self.test_target), self.test_target.shape)

        # Convert sparse targets to dense arrays
        print("Converting devel_target from sparse matrix to dense array...")
        self.devel_target = self.devel_target.toarray()                           # Convert to dense
        self.test_target = self.test_target.toarray()                             # Convert to dense
        print("devel_target (after processing):", type(self.devel_target), self.devel_target.shape)
        print("test_target (after processing):", type(self.test_target), self.test_target.shape)

        self.label_names = self.devel.target_names              # Set labels to class label names
        print("self.label_names:\n", self.label_names)
        
        self.target_names = self.label_names
        print("target_names:", type(self.target_names), len(self.target_names))

        self.num_labels = len(self.labels)
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        # Now self.devel_target is already a dense NumPy array 
        # with shape (9603, 115), so no need for MultiLabelBinarizer.

        self.y_train = self.devel_target                                    # Transform multi-label targets into a binary matrix
        self.y_test = self.test_target                                      # Transform multi-label targets into a binary matrix
        print("self.y_train:", type(self.y_train), self.y_train.shape)
        print("self.y_test:", type(self.y_test), self.y_test.shape)
        
        # Convert Y to a sparse matrix
        self.y_train_sparse = csr_matrix(self.y_train)                                       # without Transpose to match the expected shape
        self.y_test_sparse = csr_matrix(self.y_test)                                         # without Transpose to match the expected shape
        print("self.y_test_sparse:", type(self.y_test_sparse), self.y_test_sparse.shape)
        print("self.y_train_sparse:", type(self.y_train_sparse), self.y_train_sparse.shape)

        self.ytr_encoded = self.y_train
        self.yte_encoded = self.y_test
        print("self.ytr_encoded:", type(self.ytr_encoded), self.ytr_encoded.shape)
        print("self.yte_encoded:", type(self.yte_encoded), self.yte_encoded.shape)

        return self.label_names


    def _load_ohsumed(self):

        print("\n\tloading ohsumed dataset...")

        #data_path = os.path.join(get_data_home(), 'ohsumed50k')
        data_path = os.path.join(DATASET_DIR, 'ohsumed50k')

        print("data_path:", data_path)  

        self.devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        self.test = fetch_ohsumed50k(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.class_type = 'multilabel'

        #
        # inspect the raw text
        #
        print("\t--- unprocessed text ---")
        print("self.devel.data:", type(self.devel.data), len(self.devel.data))
        print("self.devel.data[0]:\n", self.devel.data[0])

        print("self.test.data:", type(self.test.data), len(self.test.data))
        print("self.test.data[0]:\n", self.test.data[0])

        #
        # preprocess: we remove stopwords, mask numbers and remove punctuation
        # if we are working with word based embeddings (fastText, Word2Vec, GloVe)
        #
        """
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=True)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=True)
        else:
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=False)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=False)
        """
        
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = preprocess(
                pd.Series(self.devel.data), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(self.test.data), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )    
        else:
            self.Xtr = preprocess(
                pd.Series(self.devel.data), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(self.test.data), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            
        print("\t--- preprocessed text ---")
        print("self.Xtr:", type(self.Xtr), len(self.Xtr))
        print("self.Xtr[0]:\n", self.Xtr[0])
        
        print("self.Xte:", type(self.Xte), len(self.Xte))
        print("self.Xte[0]:\n", self.Xte[0])

        self.devel_raw = self.Xtr
        self.test_raw = self.Xte

        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel.target, self.test.target)
        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        print("labels:\n", self.labels)

        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix
        print("devel_target:", type(self.devel_target), self.devel_target.shape)
        print("test_target:", type(self.test_target), self.test_target.shape)
        
        # Convert sparse targets to dense arrays
        print("Converting devel_target from sparse matrix to dense array...")
        self.devel_target = self.devel_target.toarray()                           # Convert to dense
        self.test_target = self.test_target.toarray()                             # Convert to dense
        print("devel_target (after processing):", type(self.devel_target), self.devel_target.shape)
        print("test_target (after processing):", type(self.test_target), self.test_target.shape)

        self.label_names = self.devel.target_names                                  # set self.labels to the class label names
        print("self.label_names:\n", self.label_names)        

        self.target_names = self.label_names
        print("target_names:", type(self.target_names), len(self.target_names))
        
        self.num_labels = len(self.labels)
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        #
        # Now self.devel_target is already a dense NumPy array so no need for MultiLabelBinarizer.
        #
        self.y_train = self.devel_target                                    # Transform multi-label targets into a binary matrix
        self.y_test = self.test_target                                      # Transform multi-label targets into a binary matrix
        print("self.y_train:", type(self.y_train), self.y_train.shape)
        print("self.y_test:", type(self.y_test), self.y_test.shape)

        # Convert Y to a sparse matrix
        self.y_train_sparse = csr_matrix(self.y_train)                                       # without Transpose to match the expected shape
        self.y_test_sparse = csr_matrix(self.y_test)                                         # without Transpose to match the expected shape
        print("self.y_test_sparse:", type(self.y_test_sparse), self.y_test_sparse.shape)
        print("self.y_train_sparse:", type(self.y_train_sparse), self.y_train_sparse.shape)

        self.ytr_encoded = self.y_train
        self.yte_encoded = self.y_test
        print("self.ytr_encoded:", type(self.ytr_encoded), self.ytr_encoded.shape)
        print("self.yte_encoded:", type(self.yte_encoded), self.yte_encoded.shape)

        return self.label_names


    def _load_rcv1(self):

        print("\n\tloading rcv1 dataset...")

        #data_path = '../datasets/RCV1-v2/rcv1/'               
        
        data_path = os.path.join(DATASET_DIR, 'rcv1')

        print("data_path:", data_path)

        """
        print('Downloading rcv1v2-ids.dat.gz...')
        self.download_file(
            'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a07-rcv1-doc-ids/rcv1v2-ids.dat.gz', 
            data_path)

        print('Downloading rcv1-v2.topics.qrels.gz...')
        self.download_file(
            'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz', 
            data_path)
        """

        # ----------------------------------------------------------------
        # we presume tar balls are downloaded into this directory already
        # NB: we only do this once
        """
        print("extracting files...")
        self.extract_gz(data_path + '/' +  'rcv1v2-ids.dat.gz')
        self.extract_gz(data_path + '/' + 'rcv1-v2.topics.qrels.gz')
        self.extract_tar(data_path + '/' + 'rcv1.tar.xz')
        self.extract_tar(data_path + '/' + 'RCV1.tar.bz2')
        """
        # ----------------------------------------------------------------

        #print("fetching training and test data...")
        self.devel = fetch_RCV1(subset='train', data_path=data_path, debug=True)
        self.test = fetch_RCV1(subset='test', data_path=data_path, debug=True)

        self.classification_type = 'multilabel'
        self.class_type = 'multilabel'

        #
        # inspect the raw text
        #
        print("\t--- unprocessed text ---")
        print("self.devel.data:", type(self.devel.data), len(self.devel.data))
        print("self.devel.data[0]:\n", self.devel.data[0])

        print("self.test.data:", type(self.test.data), len(self.test.data))
        print("self.test.data[0]:\n", self.test.data[0])

        #
        # preprocess: we remove stopwords, mask numbers and remove punctuation
        # if we are working with word based embeddings (fastText, Word2Vec, GloVe)
        #
        """
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=True)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=True)
        else:
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=False)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=False)
        """
        
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = preprocess(
                pd.Series(self.devel.data), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(self.test.data), 
                remove_punctuation=True,
                lowercase=True,
                remove_stopwords=True,
                array=True                              # return as numpy array
                )    
        else:
            self.Xtr = preprocess(
                pd.Series(self.devel.data), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            self.Xte = preprocess(
                pd.Series(self.test.data), 
                remove_punctuation=False,
                lowercase=False,
                remove_stopwords=False,
                array=True                              # return as numpy array
                )
            
        print("\t--- preprocessed text ---")
        print("self.Xtr:", type(self.Xtr), len(self.Xtr))
        print("self.Xtr[0]:\n", self.Xtr[0])
        
        print("self.Xte:", type(self.Xte), len(self.Xte))
        print("self.Xte[0]:\n", self.Xte[0])

        self.devel_raw = self.Xtr
        self.test_raw = self.Xte

        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel.target, self.test.target)
        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        print("labels:\n", self.labels)

        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix
        print("devel_target:", type(self.devel_target), self.devel_target.shape)
        print("test_target:", type(self.test_target), self.test_target.shape)

        # Convert sparse targets to dense arrays
        print("Converting devel_target from sparse matrix to dense array...")
        self.devel_target = self.devel_target.toarray()                           # Convert to dense
        self.test_target = self.test_target.toarray()                             # Convert to dense
        print("devel_target (after processing):", type(self.devel_target), self.devel_target.shape)
        print("test_target (after processing):", type(self.test_target), self.test_target.shape)

        self.label_names = self.devel.target_names              # Set labels to class label names
        print("self.label_names:\n", self.label_names)
        
        self.target_names = self.label_names
        print("target_names:", type(self.target_names), len(self.target_names))

        self.num_labels = len(self.labels)
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        # Now self.devel_target is already a dense NumPy array 
        # with shape (9603, 115), so no need for MultiLabelBinarizer.

        self.y_train = self.devel_target                                    # Transform multi-label targets into a binary matrix
        self.y_test = self.test_target                                      # Transform multi-label targets into a binary matrix
        print("self.y_train:", type(self.y_train), self.y_train.shape)
        print("self.y_test:", type(self.y_test), self.y_test.shape)
        
        # Convert Y to a sparse matrix
        self.y_train_sparse = csr_matrix(self.y_train)                                       # without Transpose to match the expected shape
        self.y_test_sparse = csr_matrix(self.y_test)                                         # without Transpose to match the expected shape
        print("self.y_test_sparse:", type(self.y_test_sparse), self.y_test_sparse.shape)
        print("self.y_train_sparse:", type(self.y_train_sparse), self.y_train_sparse.shape)

        self.ytr_encoded = self.y_train
        self.yte_encoded = self.y_test
        print("self.ytr_encoded:", type(self.ytr_encoded), self.ytr_encoded.shape)
        print("self.yte_encoded:", type(self.yte_encoded), self.yte_encoded.shape)

        return self.label_names


    def analyzer(self):
        return self._vectorizer.build_analyzer()
    


    # ----------------------------------------------------------------------------------------------------------------------------------------------------------

    def save(self, pickle_file):
        """
        Save the LCDataset instance to a pickle file.
        """
        print(f'Saving {pickle_file} to pickle...')

        # Combine the relevant attributes, including lcr_model, into a dictionary for serialization
        lc_pt_data = {
            'name': self.name,
            'classification_type': self.classification_type,
            'vectorization_type': self.vectorization_type,
            'vectorizer': self.vectorizer,
            'nC': self.nC,
            'type': self.type,
            'pretrained': self.pretrained,
            'embedding_type': self.embedding_type,
            'embedding_path': self.embedding_path,
            'embedding_comp_type': self.embedding_comp_type,
            'devel_raw': self.devel_raw,
            'test_raw': self.test_raw,
            'Xtr': self.Xtr,
            'Xte': self.Xte,
            'devel_target': self.devel_target,
            'test_target': self.test_target,
            'devel_labelmatrix': self.devel_labelmatrix,
            'test_labelmatrix': self.test_labelmatrix,
            'vectorizer': self.vectorizer,
            'Xtr_vectorized': self.Xtr_vectorized,
            'Xte_vectorized': self.Xte_vectorized,
            'y_train_sparse': self.y_train_sparse,
            'y_test_sparse': self.y_test_sparse,
            'ytr_encoded': self.ytr_encoded,
            'yte_encoded': self.yte_encoded,
            'target_names': self.target_names,
            'class_type': self.class_type,
            'vocabulary': self.vocabulary,
            'tokenizer': self.tokenizer,
            'max_length': self.max_length,
            'embedding_vocab_matrix': self.embedding_vocab_matrix,
            'Xtr_weighted_embeddings': self.Xtr_weighted_embeddings,
            'Xte_weighted_embeddings': self.Xte_weighted_embeddings,
            'Xtr_avg_embeddings': self.Xtr_avg_embeddings,
            'Xte_avg_embeddings': self.Xte_avg_embeddings,
            'Xtr_summary_embeddings': self.Xtr_summary_embeddings,
            'Xte_summary_embeddings': self.Xte_summary_embeddings,
            'word2index': self.word2index,
            'out_of_vocabulary': self.out_of_vocabulary,
            'unk_index': self.unk_index,
            'pad_index': self.pad_index,
            'devel_index': self.devel_index,
            'test_index': self.test_index,
            'lcr_model': self.lcr_model,                    # Include the lcr_model object
            'loaded': self.loaded,
            'initialized': self.initialized
        }

        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(lc_pt_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Data successfully pickled at:", pickle_file)
        except Exception as e:
            print(f'Error: Failed to save pickle file {pickle_file}. Exception: {e}')



    @classmethod
    def load(cls, pickle_file):
        """
        Load the LCDataset instance from a pickle file.
        """
        print(f"Loading pickle file: {pickle_file}...")

        try:
            with open(pickle_file, 'rb') as f:
                data_loaded = pickle.load(f)

            # Create a new instance
            lcd = cls.__new__(cls)  # Bypass __init__

            # Set attributes based on the loaded data
            lcd.name = data_loaded['name']
            lcd.classification_type = data_loaded['classification_type']
            lcd.vectorization_type = data_loaded['vectorization_type']
            lcd.vectorizer = data_loaded['vectorizer']
            lcd.nC = data_loaded['nC']
            lcd.type = data_loaded['type']
            lcd.pretrained = data_loaded['pretrained']
            lcd.embedding_type = data_loaded['embedding_type']
            lcd.embedding_path = data_loaded['embedding_path']
            lcd.embedding_comp_type = data_loaded['embedding_comp_type']
            lcd.devel_raw = data_loaded['devel_raw']
            lcd.test_raw = data_loaded['test_raw']
            lcd.Xtr = data_loaded['Xtr']
            lcd.Xte = data_loaded['Xte']
            lcd.devel_target = data_loaded['devel_target']
            lcd.test_target = data_loaded['test_target']
            lcd.devel_labelmatrix = data_loaded.get('devel_labelmatrix')
            lcd.test_labelmatrix = data_loaded.get('test_labelmatrix')
            lcd.vectorizer = data_loaded['vectorizer']  
            lcd.Xtr_vectorized = data_loaded['Xtr_vectorized']
            lcd.Xte_vectorized = data_loaded['Xte_vectorized']
            lcd.y_train_sparse = data_loaded['y_train_sparse']
            lcd.y_test_sparse = data_loaded['y_test_sparse']
            lcd.ytr_encoded = data_loaded['ytr_encoded']
            lcd.yte_encoded = data_loaded['yte_encoded']
            lcd.target_names = data_loaded['target_names']
            lcd.class_type = data_loaded['class_type']
            lcd.vocabulary = data_loaded['vocabulary']
            lcd.tokenizer = data_loaded['tokenizer']
            lcd.max_length = data_loaded['max_length']
            lcd.embedding_vocab_matrix = data_loaded['embedding_vocab_matrix']
            lcd.Xtr_weighted_embeddings = data_loaded['Xtr_weighted_embeddings']
            lcd.Xte_weighted_embeddings = data_loaded['Xte_weighted_embeddings']
            lcd.Xtr_avg_embeddings = data_loaded['Xtr_avg_embeddings']
            lcd.Xte_avg_embeddings = data_loaded['Xte_avg_embeddings']
            lcd.Xtr_summary_embeddings = data_loaded['Xtr_summary_embeddings']
            lcd.Xte_summary_embeddings = data_loaded['Xte_summary_embeddings']
            lcd.word2index = data_loaded['word2index']
            lcd.out_of_vocabulary = data_loaded['out_of_vocabulary']
            lcd.unk_index = data_loaded['unk_index']
            lcd.pad_index = data_loaded['pad_index']
            lcd.devel_index = data_loaded['devel_index']
            lcd.test_index = data_loaded['test_index']
            lcd.lcr_model = data_loaded['lcr_model']                                # Restore the lcr_model object
            lcd.loaded = data_loaded['loaded']
            lcd.initialized = data_loaded['initialized']

            return lcd  # Return the LCDataset instance

        except Exception as e:
            print(f"Error: Failed to load pickle file {pickle_file}. Exception: {e}")
            return None




        

    # ------------------------------------------------------------------------------------------------------------------------
    # ancillary support methods
    # ------------------------------------------------------------------------------------------------------------------------

    def split(self):
        
        return train_test_split(
            self.X,
            self.y, 
            test_size = TEST_SIZE, 
            random_state = 60,
            shuffle=True
            #stratify=self.y
            )

    def _missing_values(self, df):
        """
        Calculate the percentage of missing values for each column in a DataFrame.
        
        Args:
        df (pd.DataFrame): The input DataFrame to analyze.
        
        Returns:
        pd.DataFrame: A DataFrame containing the total count and percentage of missing values for each column.
        """
        # Calculate total missing values and their percentage
        total = df.isnull().sum()
        percent = (total / len(df) * 100)
        
        # Create a DataFrame with the results
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        
        # Sort the DataFrame by percentage of missing values (descending)
        missing_data = missing_data.sort_values('Percent', ascending=False)
        
        # Filter out columns with no missing values
        missing_data = missing_data[missing_data['Total'] > 0]
        
        print("Columns with missing values:")
        print(missing_data)
        
        return missing_data


    def _remove_punctuation(self, x):
        punctuationfree="".join([i for i in x if i not in string.punctuation])
        return punctuationfree


    # Function to lemmatize text with memory optimization
    def _lemmatization(self, texts, chunk_size=1000):
        lmtzr = WordNetLemmatizer()
        
        num_chunks = len(texts) // chunk_size + 1
        #print(f"Number of chunks: {num_chunks}")
        for i in range(num_chunks):
            chunk = texts[i*chunk_size:(i+1)*chunk_size]
            texts[i*chunk_size:(i+1)*chunk_size] = [' '.join([lmtzr.lemmatize(word) for word in text.split()]) for text in chunk]
        
        return texts


    def _preprocess(self, text_series: pd.Series, remove_punctuation=True):
        """
        Preprocess a pandas Series of texts by removing punctuation and stopwords, leavig numbers unmasked.
        We do NOT lowercase the text or tokenize the text, ensuring that the text remains in its original form.

        Parameters:
        - text_series: A pandas Series containing text data (strings).

        Returns:
        - processed_texts: A NumPy array containing processed text strings.
        """

        print("_preprocessing...")
        print("text_series:", type(text_series), text_series.shape)

        # Load stop words once outside the loop
        stop_words = set(stopwords.words('english'))
        punctuation_table = str.maketrans('', '', string.punctuation)  # Translation table to remove punctuation

        # Function to process each text (masking numbers, removing punctuation, and stopwords)
        def process_text(text):

            # Remove punctuation
            if (remove_punctuation):
                text = text.translate(punctuation_table)

            # Remove stopwords without tokenizing or lowercasing
            for stopword in stop_words:
                text = re.sub(r'\b' + re.escape(stopword) + r'\b', '', text)

            # Ensure extra spaces are removed after stopwords are deleted
            return ' '.join(text.split())

        # Use Parallel processing with multiple cores
        processed_texts = Parallel(n_jobs=-1)(delayed(process_text)(text) for text in text_series)

        # Return as NumPy array
        return np.array(processed_texts)


    def show(self):
        nTr_docs = len(self.devel_raw)
        nTe_docs = len(self.test_raw)
        #nfeats = len(self._vectorizer.vocabulary_)
        nfeats = len(self.vectorizer.vocabulary_)
        nC = self.devel_labelmatrix.shape[1]
        nD=nTr_docs+nTe_docs
        print(f'{self.classification_type}, nD={nD}=({nTr_docs}+{nTe_docs}), nF={nfeats}, nC={nC}')
        return self


    def get_label_names(self):
        """
        Returns the labels and their associated names (catgeories) associated with t
        he dataset. Useful for plotting confusion matrices and more.
        """
        if hasattr(self, 'label_names'):
            return self.label_names

    def get_labels(self):
        if hasattr(self, 'labels'):
            return self.labels
        else:
            return None

    def is_loaded(self):
        if hasattr(self, 'loaded'):
            return self.loaded
        else:
            return False

    def analyzer(self):
        #return self._vectorizer.build_analyzer()
        return self.vectorizer.build_analyzer()


    def download_file(self, url, path):
        file_name = url.split('/')[-1]
        filename_with_path = path + "/" + file_name

        print("file: ", {filename_with_path})

        file = Path(filename_with_path)

        if not file.exists():
            print('File %s does not exist. Downloading ...\n', file_name)
            file_data = request.urlopen(url)
            data_to_write = file_data.read()

            with file.open('wb') as f:
                f.write(data_to_write)
        else:
            print('File %s already existed.\n', file_name)


    def extract_tar(self, path):
        path = Path(path)
        dir_name = '.'.join(path.name.split('.')[:-2])
        dir_output = path.parent/dir_name
        if not dir_output.exists():
            if path.exists():
                tf = tarfile.open(str(path))
                tf.extractall(path.parent)
            else:
                print('ERROR: File %s is required. \n', path.name)


    def extract_gz(self, path):
        path = Path(path)
        file_output_name = '.'.join(path.name.split('.')[:-1])
        file_name = path.name
        if not (path.parent/file_output_name).exists():
            print('Extracting %s ...\n', file_name)

            with gzip.open(str(path), 'rb') as f_in:
                with open(str(path.parent/file_output_name), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

#
# end class
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def loadpt_data(dataset, vtype='tfidf', pretrained=None, embedding_path=VECTOR_CACHE, emb_type='word', embedding_comp_type='avg', seed=RANDOM_SEED):

    print(f'\n\tloading dataset:{dataset} with embedding:{pretrained} data...')
    
    # Determine the model name based on the pretrained option
    model_name_mapping = {
        'glove': GLOVE_MODEL,
        'word2vec': WORD2VEC_MODEL,
        'fasttext': FASTTEXT_MODEL,
        'hyperbolic': HYPERBOLIC_MODEL,
        'bert': BERT_MODEL,
        'roberta': ROBERTA_MODEL,
        'distilbert': DISTILBERT_MODEL,
        'xlnet': XLNET_MODEL,
        'gpt2': GPT2_MODEL,
        'llama': LLAMA_MODEL,
        'deepseek': DEEPSEEK_MODEL
    }

    # Look up the model name, or raise an error if not found
    model_name = model_name_mapping.get(pretrained)
    print("model_name:", model_name)
    
    pickle_file_name = f'{dataset}_{vtype}_{pretrained}_{model_name}.pickle'.replace("/", "_")
    pickle_file = PICKLE_DIR + pickle_file_name
    print("pickle_file:", pickle_file)

    # If the pickle file exists, load the dataset from it
    if os.path.exists(pickle_file):
        print(f"Loading LCDataset data from '{pickle_file}'...")
        lcd = LCDataset.load(pickle_file)
        if lcd:
            return lcd
        else:
            print("Error loading pickle file.")
            return None

    else:
        print(f"'{pickle_file}' not found, loading {dataset}...")
        
        lcd = LCDataset(
            name=dataset,                                       # dataset name 
            vectorization_type=vtype,                           # vectorization type (one of 'tfidf', 'count')
            embedding_type=emb_type,                            # embedding type (one of 'word', 'token')
            pretrained=pretrained,                              # pretrained embeddings (model type or None)
            embedding_path=embedding_path,                      # path to embeddings
            embedding_comp_type=embedding_comp_type,            # embedding computation type (one of 'avg', 'summary')
            seed=seed
        )    

        lcd.vectorize()                             # vectorize the dataset
        lcd.init_embedding_matrices()               # initialize the embedding matrices
        lcd.init_neural_data()                      # initialize the neural data inputs

        # Save the dataset instance to a pickle file
        lcd.save(pickle_file)

        return lcd







def load_fasttext_format(path):
    print(f'loading {path}')
    labels,docs=[],[]
    for line in tqdm(open(path, 'rt').readlines()):
        space = line.strip().find(' ')
        label = int(line[:space].replace('__label__',''))-1
        labels.append(label)
        docs.append(line[space+1:])
    labels=np.asarray(labels,dtype=int)
    return docs,labels




    
