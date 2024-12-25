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

from scipy.sparse import csr_matrix

import pickle

from scipy.special._precompute.expn_asy import generate_A

import torch

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from joblib import Parallel, delayed

from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1

from model.LCRepresentationModel import *



#SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news", "ohsumed", "imdb", "arxiv", "cmu_movie_corpus"]
SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news", "ohsumed", "imdb", "arxiv"]

DATASET_DIR = '../datasets/'                        # dataset directory



#
# Disable Hugging Face tokenizers parallelism to avoid fork issues
#
os.environ["TOKENIZERS_PARALLELISM"] = "false"


MIN_DF_COUNT = 5                    # minimum document frequency count for a term to be included in the vocabulary
TEST_SIZE = 0.175                   # test size for train/test split
NUM_DL_WORKERS = 3                  # number of workers to handle DataLoader tasks




nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))


#
# LCDataset class for legacy Neural Model (supports word based models) which includes 
# word2vec, glove and fasttext integrated with SVM and Logistic Regression ML models and 
# CNN, LSTM and ATTN based neural models. The ML models are trained with scikit-learn
# and also support various transformer models as well
#

class LCDataset:

    """
    A class to handle loading and preparing datasets for text classification.
    Supports multiple datasets including reuters21578, 20newsgroups, ohsumed, rcv1 (and imdb and arxiv coming).
    """

    dataset_available = {'reuters21578', '20newsgroups', 'ohsumed', 'rcv1', 'bbc-news'}

    def __init__(self, name, vectorization_type='tfidf', pretrained=None, embedding_type='word', embedding_path=VECTOR_CACHE, embedding_comp_type='avg'):
        """
        Initializes the LCDataset object with the specified dataset and vectorization parameters. This method
        both loads the dataset into the respective LCDataset variables as well as sets up the proper model and 
        vectorizer objects and then vectorizes both the training (Xtr) and test (Xte) data using said vectorizer.
        NB that the training and test data is split in the relevant load_ method for the dataset

        - name: Name of the dataset to load.
        - vectorization_type: 'tfidf' or 'count', determines which vectorizer to use for tokenization.
        - embedding_type: 'word' for word-based embeddings (GloVe, Word2Vec, fastText) or 'token' for 
        token-based models (BERT, RoBERTA, LLaMa).
        """

        print("\n\tinitializing LCDataset...")

        assert name in LCDataset.dataset_available, f'dataset {name} is not available'

        self.name = name
        print("self.name:", self.name) 

        self.vectorization_type = vectorization_type
        print("vectorization_type:", self.vectorization_type)

        self.pretrained = pretrained
        print("pretrained:", self.pretrained)

        self.embedding_type = embedding_type
        print("embedding_type:", self.embedding_type)

        self.embedding_path = embedding_path
        print("embedding_path:", self.embedding_path)

        self.pretrained_path = embedding_path
        print("pretrained_path:", self.pretrained_path)

        self.embedding_comp_type = embedding_comp_type
        print("embedding_comp_type:", self.embedding_comp_type)

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

        #
        # instantiate representation model if token/transformer based 
        # we offload this code to another class
        #
        self.lcr_model = None          # custom representation model class object
    
        if (pretrained == 'word2vec'):
            print("Using Word2Vec pretrained embeddings...")
            
            self.lcr_model = Word2VecLCRepresentationModel(
                model_name=WORD2VEC_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )
        elif (pretrained == 'glove'):
            print("Using GloVe pretrained embeddings...")
            
            self.lcr_model = GloVeLCRepresentationModel(
                model_name=GLOVE_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )

        elif (pretrained == 'fasttext'):
            print("Using FastText pretrained embeddings with subwords...")

            self.lcr_model = FastTextGensimLCRepresentationModel(
                model_name=FASTTEXT_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )

        elif (pretrained == 'bert'):
            print("Using BERT pretrained embeddings...")

            self.lcr_model = BERTLCRepresentationModel(
                model_name=BERT_MODEL, 
                model_dir=embedding_path,
                vtype=vectorization_type
            )
        
        elif (pretrained == 'roberta'):
            print("Using RoBERTa pretrained embeddings...")

            self.lcr_model = RoBERTaLCRepresentationModel(
                model_name=ROBERTA_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )

        elif (pretrained == 'distilbert'):
            print("Using DistilBERT pretrained embeddings...")

            self.lcr_model = DistilBERTLCRepresentationModel(
                model_name=DISTILBERT_MODEL, 
                model_dir=embedding_path,
                vtype=vectorization_type
            )

        elif (pretrained == 'gpt2'):
            print("Using GPT2 pretrained embeddings...")

            self.lcr_model = GPT2LCRepresentationModel(
                model_name=GPT2_MODEL, 
                model_dir=embedding_path,  
                vtype=vectorization_type
            )

        elif (pretrained == 'xlnet'):
            print("Using XLNet pretrained embeddings...")

            self.lcr_model = XLNetLCRepresentationModel(
                model_name=XLNET_MODEL, 
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

        #print("LCDataset initialized...\n")



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
        if (self.pretrained in ['fasttext', 'bert', 'roberta', 'distilbert', 'xlnet', 'llama']):      

            print("generating token / subword based dataset representations...")

            self.Xtr_avg_embeddings, self.Xtr_summary_embeddings = self.lcr_model.encode_docs(self.Xtr)                                  
            self.Xte_avg_embeddings, self.Xte_summary_embeddings = self.lcr_model.encode_docs(self.Xte)

            # not supported weighted average comp method for transformer based models due to
            # complexity of vectorization and tokenization mapping across models 
            # word embedding models like word2vec, GloVe or fasTtext
            self.Xtr_weighted_embeddings = self.Xtr_avg_embeddings
            self.Xte_weighted_embeddings = self.Xte_avg_embeddings
            
        elif (self.pretrained in ['word2vec', 'glove']):                        # word based embeddings
            
            print("generating word based dataset representations...")

            self.Xtr_weighted_embeddings, self.Xtr_avg_embeddings = self.lcr_model.encode_docs(
                #self.Xtr.tolist(),
                self.Xtr, 
                self.embedding_vocab_matrix
            )
            
            self.Xte_weighted_embeddings, self.Xte_avg_embeddings = self.lcr_model.encode_docs(
                #self.Xte.tolist(),
                self.Xte, 
                self.embedding_vocab_matrix
            )

            # CLS token summary embeddings not supported in pretrained 
            # word embedding models like word2vec, GloVe or fasTtext
            self.Xtr_summary_embeddings = self.Xtr_avg_embeddings
            self.Xte_summary_embeddings = self.Xte_avg_embeddings

        elif (self.pretrained in ['gpt2']):                        # GPT2, does not include a summary embedding token option

            print("generating GPT2 based dataset repressentations...")

            self.Xtr_avg_embeddings, first_tokens = self.lcr_model.encode_docs(
                self.Xtr, 
                self.embedding_vocab_matrix
            )
            
            self.Xte_avg_embeddings, first_tokens = self.lcr_model.encode_docs(
                self.Xte, 
                self.embedding_vocab_matrix
            )

            # not supported weighted average comp method for transformer based models due to
            # complexity of vectorization and tokenization mapping across models 
            # word embedding models like word2vec, GloVe or fasTtext
            self.Xtr_weighted_embeddings = self.Xtr_avg_embeddings
            self.Xte_weighted_embeddings = self.Xte_avg_embeddings

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
        print("Xtr_summary_embeddings:", type(self.Xtr_summary_embeddings), self.Xtr_summary_embeddings.shape)
        print("Xte_avg_embeddings:", type(self.Xte_avg_embeddings), self.Xte_avg_embeddings.shape)
        print("Xte_summary_embeddings:", type(self.Xte_summary_embeddings), self.Xte_summary_embeddings.shape)
        print("self.Xtr_weighted_embeddings:", type(self.Xtr_weighted_embeddings), self.Xtr_weighted_embeddings.shape)
        print("self.Xte_weighted_embeddings:", type(self.Xte_weighted_embeddings), self.Xte_weighted_embeddings.shape)

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
            print("self.lcr_model\n: ", self.model)
            known_words.update(self.lcr_model.vocabulary())         # polymorphic behavior

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
            random_state = seed
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
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = self._preprocess(pd.Series(self.X_train_raw), remove_punctuation=True)
            self.Xte = self._preprocess(pd.Series(self.X_test_raw), remove_punctuation=True)
        else:
            self.Xtr = self._preprocess(pd.Series(self.X_train_raw), remove_punctuation=False)
            self.Xte = self._preprocess(pd.Series(self.X_test_raw), remove_punctuation=False)

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
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=True)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=True)
        else:
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=False)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=False)

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
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=True)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=True)
        else:
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=False)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=False)

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
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=True)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=True)
        else:
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=False)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=False)

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
        if (self.embedding_type in ['word', 'subword', 'sub-word']):
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=True)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=True)
        else:
            self.Xtr = self._preprocess(pd.Series(self.devel.data), remove_punctuation=False)
            self.Xte = self._preprocess(pd.Series(self.test.data), remove_punctuation=False)

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
    

    def _preprocess_deprecated(self, text_series: pd.Series):
        """
        Preprocess a pandas Series of texts by removing punctuation, stopwords, and masking numbers.
        We do NOT lowercase the text or tokenize the text, ensuring that the text remains in its original form.

        Parameters:
        - text_series: A pandas Series containing text data (strings).

        Returns:
        - processed_texts: A NumPy array containing processed text strings.
        """

        print("Preprocessing text without tokenization...")
        print("text_series:", type(text_series), text_series.shape)

        # Load stop words once outside the loop
        stop_words = set(stopwords.words('english'))
        punctuation_table = str.maketrans('', '', string.punctuation)  # Translation table to remove punctuation

        # Function to mask numbers in the text
        def _mask_numbers(text, number_mask='numbermask'):
            mask = re.compile(r'\b[0-9][0-9.,-]*\b')
            return mask.sub(number_mask, text)

        # Function to process each text (masking numbers, removing punctuation, and stopwords)
        def process_text(text):
            # Mask numbers
            text = _mask_numbers(text)

            # Remove punctuation
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



def loadpt_data(dataset, vtype='tfidf', pretrained=None, embedding_path=VECTOR_CACHE, emb_type='word', embedding_comp_type='avg'):

    print(f'\n\tloading dataset:{dataset} with embedding:{pretrained} data...')
    
    # Determine the model name based on the pretrained option
    model_name_mapping = {
        'glove': GLOVE_MODEL,
        'word2vec': WORD2VEC_MODEL,
        'fasttext': FASTTEXT_MODEL,
        'bert': BERT_MODEL,
        'roberta': ROBERTA_MODEL,
        'distilbert': DISTILBERT_MODEL,
        'gpt2': GPT2_MODEL,
        'xlnet': XLNET_MODEL,
        'llama': LLAMA_MODEL
    }

    # Look up the model name, or raise an error if not found
    model_name = model_name_mapping.get(pretrained)
    print("model_name:", model_name)
    
    pickle_file_name = f'{dataset}_{vtype}_{pretrained}_{model_name}.pickle'.replace("/", "_")
    pickle_file = PICKLE_DIR + pickle_file_name
    print("pickle_file:", pickle_file)

    # If the pickle file exists, load the dataset from it
    if os.path.exists(pickle_file):
        print(f"Loading tokenized data from '{pickle_file}'...")
        lcd = LCDataset.load(pickle_file)
        if lcd:
            return lcd
        else:
            print("Error loading pickle file.")
            return None

    else:
        print(f"'{pickle_file}' not found, loading {dataset}...")
        
        lcd = LCDataset(
            name=dataset,                               # dataset name 
            vectorization_type=vtype,                   # vectorization type (one of 'tfidf', 'count')
            embedding_type=emb_type,                    # embedding type (one of 'word', 'token')
            pretrained=pretrained,                      # pretrained embeddings (model type or None)
            embedding_path=embedding_path,              # path to embeddings
            embedding_comp_type=embedding_comp_type     # embedding computation type (one of 'avg', 'weighted', 'summary')
        )    

        lcd.vectorize()                             # vectorize the dataset
        lcd.init_embedding_matrices()               # initialize the embedding matrices
        lcd.init_neural_data()                      # initialize the neural data inputs

        # Save the dataset instance to a pickle file
        lcd.save(pickle_file)

        return lcd






def preprocess(text_series: pd.Series, remove_punctuation=True, lowercase=False, remove_stopwords=False):
    """
    Preprocess a pandas Series of texts by removing punctuation, optionally lowercasing, and optionally removing stopwords.
    Numbers are not masked, and text remains in its original form unless modified by these options.

    Parameters:
    - text_series: A pandas Series containing text data (strings).
    - remove_punctuation: Boolean indicating whether to remove punctuation.
    - lowercase: Boolean indicating whether to convert text to lowercase.
    - remove_stopwords: Boolean indicating whether to remove stopwords.

    Returns:
    - processed_texts: A list containing processed text strings.
    """

    print("preprocessing...")
    print("text_series:", type(text_series), text_series.shape)
    
    # Load stop words once outside the loop
    stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    punctuation_table = str.maketrans('', '', string.punctuation)  # Translation table to remove punctuation

    # Function to process each text
    def process_text(text):
        if lowercase:
            text = text.lower()

        if remove_punctuation:
            text = text.translate(punctuation_table)

        if remove_stopwords:
            for stopword in stop_words:
                text = re.sub(r'\b' + re.escape(stopword) + r'\b', '', text)

        # Ensure extra spaces are removed after stopwords are deleted
        return ' '.join(text.split())

    # Use Parallel processing with multiple cores
    processed_texts = Parallel(n_jobs=-1)(delayed(process_text)(text) for text in text_series)

    # Return as a list
    return list(processed_texts)



def _mask_numbers(data, number_mask='[NUM]'):
    """
    Masks numbers in the given text data with a placeholder.
    """
    mask = re.compile(r'\b[0-9][0-9.,-]*\b')
    return [mask.sub(number_mask, text) for text in data]


def preprocess_text(data):
    """
    Preprocess the text data by converting to lowercase, masking numbers, 
    removing punctuation, and removing stopwords.
    """
    import re
    from nltk.corpus import stopwords
    from string import punctuation

    stop_words = set(stopwords.words('english'))
    punct_table = str.maketrans("", "", punctuation)

    def _remove_punctuation_and_stopwords(data):
        """
        Removes punctuation and stopwords from the text data.
        """
        cleaned = []
        for text in data:
            # Remove punctuation and lowercase text
            text = text.translate(punct_table).lower()
            # Remove stopwords
            tokens = text.split()
            tokens = [word for word in tokens if word not in stop_words]
            cleaned.append(" ".join(tokens))
        return cleaned

    # Apply preprocessing steps
    masked = _mask_numbers(data)
    cleaned = _remove_punctuation_and_stopwords(masked)
    return cleaned




def _preprocess_old(text_series: pd.Series, remove_punctuation=True):
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





def _label_matrix(tr_target, te_target):
    """
    Converts multi-label target data into a binary matrix format using MultiLabelBinarizer.
    
    Input:
    - tr_target: A list (or iterable) of multi-label sets for the training data. 
                 Each element is a list, tuple, or set of labels assigned to a sample.
                 Example: [["label1", "label2"], ["label3"], ["label2", "label4"]]
    - te_target: A list (or iterable) of multi-label sets for the test data.
                 Each element is a list, tuple, or set of labels assigned to a sample.
                 Example: [["label1"], ["label3", "label4"]]
    
    Output:
    - ytr: A binary label matrix for the training data where each column represents a label.
           The matrix has shape (n_samples, n_classes).
    - yte: A binary label matrix for the test data where each column represents a label.
           The matrix has shape (n_samples, n_classes).
    - mlb.classes_: A list of all unique classes (labels) across the training data.
    """
    
    """
    print("_label_matrix...")
    print("tr_target:", tr_target)
    print("te_target:", te_target)
    """

    mlb = MultiLabelBinarizer(sparse_output=True)
    
    ytr = mlb.fit_transform(tr_target)
    yte = mlb.transform(te_target)

    """
    print("ytr:", type(ytr), ytr.shape)
    print("yte:", type(yte), yte.shape)

    print("MultiLabelBinarizer.classes_:\n", mlb.classes_)
    """
    
    return ytr, yte, mlb.classes_





# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Load dataset method for transformer based neural models
#
def trans_lc_load_dataset(name, seed):

    print("\n\tLoading dataset for transformer:", name)
    print("seed:", seed)

    if name == "20newsgroups":

        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

        target_names = list(set(train_data.target_names))  # Ensures unique class names
        num_classes = len(target_names)
  
        # Preprocess text data
        """
        train_data_processed = preprocess_text(train_data.data)
        test_data_processed = preprocess_text(test_data.data)
        """

        train_data_processed = preprocess(
            pd.Series(train_data.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False
            )

        test_data_processed = preprocess(
            pd.Series(test_data.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False
            )

        class_type = 'single-label'

        return (train_data_processed, train_data.target), (test_data_processed, test_data.target), num_classes, target_names, class_type
        

    elif name == "reuters21578":
        
        import os

        data_path = os.path.join(DATASET_DIR, 'reuters21578')    
        print("data_path:", data_path)  

        class_type = 'multi-label'

        train_labelled_docs = fetch_reuters21578(subset='train', data_path=data_path)
        test_labelled_docs = fetch_reuters21578(subset='test', data_path=data_path)

        train_data = preprocess(
            pd.Series(train_labelled_docs.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)
        
        test_data = preprocess(
            pd.Series(test_labelled_docs.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)
        
        """
        train_data = preprocess_text(train_labelled_docs.data)
        test_data = preprocess_text(list(test_labelled_docs.data))
        """

        train_target = train_labelled_docs.target
        test_target = test_labelled_docs.target
        
        train_target, test_target, labels = _label_matrix(train_target, test_target)

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        target_names = train_labelled_docs.target_names
        num_classes = len(target_names)
        
        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type
        
    elif name == "ohsumed":

        import os
        
        data_path = os.path.join(DATASET_DIR, 'ohsumed50k')

        devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        test = fetch_ohsumed50k(subset='test', data_path=data_path)

        """
        train_data = preprocess_text(devel.data)
        test_data = preprocess_text(test.data)
        """
        
        """
        train_data = _preprocess(pd.Series(devel.data), remove_punctuation=False)
        test_data = _preprocess(pd.Series(test.data), remove_punctuation=False)
        """

        train_data = preprocess(
            pd.Series(devel.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)
        
        test_data = preprocess(
            pd.Series(test.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)

        train_target, test_target = devel.target, test.target
        class_type = 'multi-label'
        
        train_target, test_target, target_names = _label_matrix(train_target, test_target)

        target_names = devel.target_names

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        num_classes = len(target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type
    
    elif name == "bbc-news":

        import os

        for dirname, _, filenames in os.walk(DATASET_DIR + 'bbc-news'):
            for filename in filenames:
                print(os.path.join(dirname, filename))

        class_type = 'single-label'

        # Load datasets
        train_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Train.csv')
        #test_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Test.csv')
        
        target_names = train_set['Category'].unique()
        num_classes = len(train_set['Category'].unique())
        #print(f"num_classes: {len(target_names)}")
        #print("class_names:", target_names)
        
        # we split the train data into train and test here because that is
        # not done for us with the BBC News dataset (Test data is not labeled)
        train_data, test_data, train_target, test_target = train_test_split(
            train_set['Text'], 
            train_set['Category'], 
            train_size = 1-TEST_SIZE, 
            random_state = seed,
        )

        """
        train_data = preprocess_text(train_data.tolist())
        test_data = preprocess_text(test_data.tolist())
        """

        train_data = preprocess(
            train_data, 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )

        test_data = preprocess(
            test_data, 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )
        
        #
        # set up label targets
        # Convert target labels to 1D arrays
        train_target_arr = np.array(train_target)  # Flattening the training labels into a 1D array
        test_target_arr = np.array(test_target)    # Flattening the test labels into a 1D array

        # Use LabelEncoder to encode the labels into label IDs
        label_encoder = LabelEncoder()
        label_encoder.fit(train_target_arr)  # Fit on training labels

        # Transform labels to numeric IDs
        train_target_encoded = label_encoder.transform(train_target_arr)
        test_target_encoded = label_encoder.transform(test_target_arr)

        return (train_data, train_target_encoded), (test_data, test_target_encoded), num_classes, target_names, class_type

    elif name == "rcv1":

        import os

        data_path = os.path.join(DATASET_DIR, 'rcv1')
        
        class_type = 'multi-label'

        devel = fetch_RCV1(subset='train', data_path=data_path)
        test = fetch_RCV1(subset='test', data_path=data_path)

        """
        train_data = preprocess_text(devel.data)
        test_data = preprocess_text(test.data)
        """

        train_data = preprocess(
            pd.Series(devel.data), 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )

        test_data = preprocess(
            pd.Series(test.data), 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )
        
        train_target, test_target = devel.target, test.target
                
        train_target, test_target, target_names = _label_matrix(train_target, test_target)

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        num_classes = len(target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type

    elif name == 'imdb':

        from datasets import load_dataset
        import os

        class_type = 'single-label'

        data_path = os.path.join(DATASET_DIR, 'imdb')

        # Load IMDB dataset using the Hugging Face Datasets library
        imdb_dataset = load_dataset('imdb', cache_dir=data_path)

        #train_data = preprocess_text(imdb_dataset['train']['text'])
        #train_data = _preprocess(imdb_dataset['train']['text'], remove_punctuation=False)
        train_data = imdb_dataset['train']['text']

        # Split dataset into training and test data
        #train_data = imdb_dataset['train']['text']
        train_target = np.array(imdb_dataset['train']['label'], dtype=np.int64)  # Convert to numpy array of type int64

        #test_data = imdb_dataset['test']['text']
        #test_data = preprocess_text(imdb_dataset['test']['text'])
        #test_data = _preprocess(imdb_dataset['test']['text'], remove_punctuation=False)
        test_data = imdb_dataset['test']['text']

        test_target = np.array(imdb_dataset['test']['label'], dtype=np.int64)  # Convert to numpy array of type int64

        train_data = preprocess(
            pd.Series(train_data), 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )

        test_data = preprocess(
            pd.Series(test_data), 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )

        # Define target names
        target_names = ['negative', 'positive']
        num_classes = len(target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type

    elif name == 'arxiv':

        import os, json, re

        class_type = 'multi-label'

        sci_field_map = {'astro-ph': 'Astrophysics',
                'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
                'astro-ph.EP': 'Earth and Planetary Astrophysics',
                'astro-ph.GA': 'Astrophysics of Galaxies',
                'astro-ph.HE': 'High Energy Astrophysical Phenomena',
                'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
                'astro-ph.SR': 'Solar and Stellar Astrophysics',
                'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
                'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
                'cond-mat.mtrl-sci': 'Materials Science',
                'cond-mat.other': 'Other Condensed Matter',
                'cond-mat.quant-gas': 'Quantum Gases',
                'cond-mat.soft': 'Soft Condensed Matter',
                'cond-mat.stat-mech': 'Statistical Mechanics',
                'cond-mat.str-el': 'Strongly Correlated Electrons',
                'cond-mat.supr-con': 'Superconductivity',
                'cs.AI': 'Artificial Intelligence',
                'cs.AR': 'Hardware Architecture',
                'cs.CC': 'Computational Complexity',
                'cs.CE': 'Computational Engineering, Finance, and Science',
                'cs.CG': 'Computational Geometry',
                'cs.CL': 'Computation and Language',
                'cs.CR': 'Cryptography and Security',
                'cs.CV': 'Computer Vision and Pattern Recognition',
                'cs.CY': 'Computers and Society',
                'cs.DB': 'Databases',
                'cs.DC': 'Distributed, Parallel, and Cluster Computing',
                'cs.DL': 'Digital Libraries',
                'cs.DM': 'Discrete Mathematics',
                'cs.DS': 'Data Structures and Algorithms',
                'cs.ET': 'Emerging Technologies',
                'cs.FL': 'Formal Languages and Automata Theory',
                'cs.GL': 'General Literature',
                'cs.GR': 'Graphics',
                'cs.GT': 'Computer Science and Game Theory',
                'cs.HC': 'Human-Computer Interaction',
                'cs.IR': 'Information Retrieval',
                'cs.IT': 'Information Theory',
                'cs.LG': 'Machine Learning',
                'cs.LO': 'Logic in Computer Science',
                'cs.MA': 'Multiagent Systems',
                'cs.MM': 'Multimedia',
                'cs.MS': 'Mathematical Software',
                'cs.NA': 'Numerical Analysis',
                'cs.NE': 'Neural and Evolutionary Computing',
                'cs.NI': 'Networking and Internet Architecture',
                'cs.OH': 'Other Computer Science',
                'cs.OS': 'Operating Systems',
                'cs.PF': 'Performance',
                'cs.PL': 'Programming Languages',
                'cs.RO': 'Robotics',
                'cs.SC': 'Symbolic Computation',
                'cs.SD': 'Sound',
                'cs.SE': 'Software Engineering',
                'cs.SI': 'Social and Information Networks',
                'cs.SY': 'Systems and Control',
                'econ.EM': 'Econometrics',
                'eess.AS': 'Audio and Speech Processing',
                'eess.IV': 'Image and Video Processing',
                'eess.SP': 'Signal Processing',
                'gr-qc': 'General Relativity and Quantum Cosmology',
                'hep-ex': 'High Energy Physics - Experiment',
                'hep-lat': 'High Energy Physics - Lattice',
                'hep-ph': 'High Energy Physics - Phenomenology',
                'hep-th': 'High Energy Physics - Theory',
                'math.AC': 'Commutative Algebra',
                'math.AG': 'Algebraic Geometry',
                'math.AP': 'Analysis of PDEs',
                'math.AT': 'Algebraic Topology',
                'math.CA': 'Classical Analysis and ODEs',
                'math.CO': 'Combinatorics',
                'math.CT': 'Category Theory',
                'math.CV': 'Complex Variables',
                'math.DG': 'Differential Geometry',
                'math.DS': 'Dynamical Systems',
                'math.FA': 'Functional Analysis',
                'math.GM': 'General Mathematics',
                'math.GN': 'General Topology',
                'math.GR': 'Group Theory',
                'math.GT': 'Geometric Topology',
                'math.HO': 'History and Overview',
                'math.IT': 'Information Theory',
                'math.KT': 'K-Theory and Homology',
                'math.LO': 'Logic',
                'math.MG': 'Metric Geometry',
                'math.MP': 'Mathematical Physics',
                'math.NA': 'Numerical Analysis',
                'math.NT': 'Number Theory',
                'math.OA': 'Operator Algebras',
                'math.OC': 'Optimization and Control',
                'math.PR': 'Probability',
                'math.QA': 'Quantum Algebra',
                'math.RA': 'Rings and Algebras',
                'math.RT': 'Representation Theory',
                'math.SG': 'Symplectic Geometry',
                'math.SP': 'Spectral Theory',
                'math.ST': 'Statistics Theory',
                'math-ph': 'Mathematical Physics',
                'nlin.AO': 'Adaptation and Self-Organizing Systems',
                'nlin.CD': 'Chaotic Dynamics',
                'nlin.CG': 'Cellular Automata and Lattice Gases',
                'nlin.PS': 'Pattern Formation and Solitons',
                'nlin.SI': 'Exactly Solvable and Integrable Systems',
                'nucl-ex': 'Nuclear Experiment',
                'nucl-th': 'Nuclear Theory',
                'physics.acc-ph': 'Accelerator Physics',
                'physics.ao-ph': 'Atmospheric and Oceanic Physics',
                'physics.app-ph': 'Applied Physics',
                'physics.atm-clus': 'Atomic and Molecular Clusters',
                'physics.atom-ph': 'Atomic Physics',
                'physics.bio-ph': 'Biological Physics',
                'physics.chem-ph': 'Chemical Physics',
                'physics.class-ph': 'Classical Physics',
                'physics.comp-ph': 'Computational Physics',
                'physics.data-an': 'Data Analysis, Statistics and Probability',
                'physics.ed-ph': 'Physics Education',
                'physics.flu-dyn': 'Fluid Dynamics',
                'physics.gen-ph': 'General Physics',
                'physics.geo-ph': 'Geophysics',
                'physics.hist-ph': 'History and Philosophy of Physics',
                'physics.ins-det': 'Instrumentation and Detectors',
                'physics.med-ph': 'Medical Physics',
                'physics.optics': 'Optics',
                'physics.plasm-ph': 'Plasma Physics',
                'physics.pop-ph': 'Popular Physics',
                'physics.soc-ph': 'Physics and Society',
                'physics.space-ph': 'Space Physics',
                'q-bio.BM': 'Biomolecules',
                'q-bio.CB': 'Cell Behavior',
                'q-bio.GN': 'Genomics',
                'q-bio.MN': 'Molecular Networks',
                'q-bio.NC': 'Neurons and Cognition',
                'q-bio.OT': 'Other Quantitative Biology',
                'q-bio.PE': 'Populations and Evolution',
                'q-bio.QM': 'Quantitative Methods',
                'q-bio.SC': 'Subcellular Processes',
                'q-bio.TO': 'Tissues and Organs',
                'q-fin.CP': 'Computational Finance',
                'q-fin.EC': 'Economics',
                'q-fin.GN': 'General Finance',
                'q-fin.MF': 'Mathematical Finance',
                'q-fin.PM': 'Portfolio Management',
                'q-fin.PR': 'Pricing of Securities',
                'q-fin.RM': 'Risk Management',
                'q-fin.ST': 'Statistical Finance',
                'q-fin.TR': 'Trading and Market Microstructure',
                'quant-ph': 'Quantum Physics',
                'stat.AP': 'Applications',
                'stat.CO': 'Computation',
                'stat.ME': 'Methodology',
                'stat.ML': 'Machine Learning',
                'stat.OT': 'Other Statistics',
                'stat.TH': 'Statistics Theory'}
        #
        # code from
        # https://www.kaggle.com/code/jampaniramprasad/arxiv-abstract-classification-using-roberta
        #
        data_path = os.path.join(DATASET_DIR, 'arxiv')

        file_path = data_path + '/arxiv-metadata-oai-snapshot.json'
        #print("file_path:", file_path)

        # Preprocessing function for text cleaning
        def clean_text(text):
            text = text.lower()                                                       # Convert text to lowercase
            text = re.sub(r'\d+', '<NUM>', text)                                      # Mask numbers
            text = re.sub(r'\$\{[^}]*\}|\$|\\[a-z]+|[{}]', '', text)                  # Remove LaTeX-like symbols
            text = re.sub(r'\s+', ' ', text).strip()                                  # Remove extra spaces
            return text

        # Generator function with progress bar
        def get_data(file_path, preprocess=False):
            with open(file_path, 'r') as f:
                # Use tqdm to wrap the file iterator
                for line in tqdm(f, desc="Loading dataset", unit="line"):
                    paper = json.loads(line)
                    if preprocess:
                        # Apply text cleaning to relevant fields
                        paper['title'] = clean_text(paper.get('title', ''))
                        paper['abstract'] = clean_text(paper.get('abstract', ''))
                    yield paper

        paper_metadata = get_data(file_path, preprocess=True)
        #print("paper_metadata:", type(paper_metadata))

        """
        def load_dataset(file_path):
            data = []
            with open(file_path, 'r') as f:
                for line in tqdm(f, desc="Loading dataset", unit="line", total=2626136):            # Approximate total
                    data.append(json.loads(line))
            return data

        dataset = load_dataset(file_path)
        """

        # Using `yield` to load the JSON file in a loop to prevent 
        # Python memory issues if JSON is loaded directly
        def get_raw_data():
            with open(file_path, 'r') as f:
                for thing in f:
                    yield thing

        #paper_metadata = get_data()

        """
        for paper in paper_metadata:
            for k, v in json.loads(paper).items():
                print(f'{k}: {v} \n')
            break
        """

        paper_titles = []
        paper_intro = []
        paper_type = []

        paper_categories = np.array(list(sci_field_map.keys())).flatten()

        metadata_of_paper = get_raw_data()
        for paper in tqdm(metadata_of_paper):
            papers_dict = json.loads(paper)
            category = papers_dict.get('categories')
            try:
                try:
                    year = int(papers_dict.get('journal-ref')[-4:])
                except:
                    year = int(papers_dict.get('journal-ref')[-5:-1])

                if category in paper_categories and 2010<year<2021:
                    paper_titles.append(papers_dict.get('title'))
                    paper_intro.append(papers_dict.get('abstract'))
                    paper_type.append(papers_dict.get('categories'))
            except:
                pass 

        """
        print("paper_titles:", paper_titles[:5])
        print("paper_intro:", paper_intro[:5])
        print("paper_type:", paper_type[:5])
        print(len(paper_titles), len(paper_intro), len(paper_type))
        """
        papers_dataframe = pd.DataFrame({
            'title': paper_titles,
            'abstract': paper_intro,
            'categories': paper_type
        })

        # preprocess text
        papers_dataframe['abstract'] = papers_dataframe['abstract'].apply(lambda x: x.replace("\n",""))
        papers_dataframe['abstract'] = papers_dataframe['abstract'].apply(lambda x: x.strip())
        papers_dataframe['text'] = papers_dataframe['title'] + '. ' + papers_dataframe['abstract']

        papers_dataframe['categories'] = papers_dataframe['categories'].apply(lambda x: tuple(x.split()))
        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())
        """

        # Ensure the 'categories' column value counts are calculated and indexed properly
        categories_counts = papers_dataframe['categories'].value_counts().reset_index(name="count")
        """
        print("categories_counts:", categories_counts.shape)
        print(categories_counts.head())
        """

        # Filter for categories with a count greater than 250
        shortlisted_categories = categories_counts.query("count > 250")["categories"].tolist()
        print("shortlisted_categories:", shortlisted_categories)

        # Choosing paper categories based on their frequency & eliminating categories with very few papers
        #shortlisted_categories = papers_dataframe['categories'].value_counts().reset_index(name="count").query("count > 250")["index"].tolist()
        papers_dataframe = papers_dataframe[papers_dataframe["categories"].isin(shortlisted_categories)].reset_index(drop=True)
        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())
        """

        # clean the text, remove special chracters, etc
        def clean_text(text):
            # Mask numbers
            text = re.sub(r'\d+', '<NUM>', text)
            # Remove special LaTeX-like symbols and tags
            text = re.sub(r'\$\{[^}]*\}|\$|\\[a-z]+|[{}]', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # Apply cleaning to dataset texts
        papers_dataframe['text'] = papers_dataframe['text'].apply(clean_text)
        
        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())
        """

        # Shuffle DataFrame
        papers_dataframe = papers_dataframe.sample(frac=1).reset_index(drop=True)

        # Sample roughtly equal number of texts from different paper categories (to reduce class imbalance issues)
        papers_dataframe = papers_dataframe.groupby('categories').head(250).reset_index(drop=True)

        # encode categories using MultiLabelBinarizer
        multi_label_encoder = MultiLabelBinarizer()
        multi_label_encoder.fit(papers_dataframe['categories'])
        papers_dataframe['categories_encoded'] = papers_dataframe['categories'].apply(lambda x: multi_label_encoder.transform([x])[0])

        papers_dataframe = papers_dataframe[["text", "categories", "categories_encoded"]]
        del paper_titles, paper_intro, paper_type
        print(papers_dataframe.head())

        # Convert encoded labels to a 2D array
        y = np.vstack(papers_dataframe['categories_encoded'].values)
        #y = papers_dataframe['categories_encoded'].values

        # Retrieve target names and number of classes
        target_names = multi_label_encoder.classes_
        num_classes = len(target_names)

        # split dataset into training and test set
        xtrain, xtest, ytrain, ytest = train_test_split(papers_dataframe['text'], y, test_size=TEST_SIZE, random_state=seed)

        return (xtrain.tolist(), ytrain), (xtest.tolist(), ytest), num_classes, target_names, class_type
    

    elif name == 'cmu_movie_corpus':                # TODO, not working with model, need to fix
        """
        Load and process the CMU Movie Corpus for multi-label classification.

        from 
        https://github.com/prateekjoshi565/movie_genre_prediction/blob/master/Movie_Genre_Prediction.ipynb

        Returns:
            train_data (list): List of movie plots (text) for training.
            train_target (numpy.ndarray): Multi-label binary matrix for training labels.
            test_data (list): List of movie plots (text) for testing.
            test_target (numpy.ndarray): Multi-label binary matrix for testing labels.
            num_classes (int): Number of unique genres (classes).
            target_names (list): List of genre names.
            class_type (str): Classification type ('multi-label').
        """

        import csv
        import json
        import os
        import re

        # Classification type
        class_type = "multi-label"

        data_path = os.path.join(DATASET_DIR, 'cmu_movie_corpus')
        print("data_path:", data_path)

        # Ensure the dataset files exist
        tsv_file = os.path.join(data_path, "movie.metadata.tsv")
        if not os.path.exists(tsv_file):
            raise FileNotFoundError(f"Dataset file not found at {tsv_file}. Please download the dataset as per the article instructions.")

        meta = pd.read_csv(tsv_file, sep = '\t', header = None)
        meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]        # rename columns

        #print("meta:\n", meta.head())

        file_path_2 = data_path + '/plot_summaries.txt'
        plots = []
        with open(file_path_2, 'r') as f:
            reader = csv.reader(f, dialect='excel-tab') 
            for row in tqdm(reader):
                plots.append(row)


        movie_id = []
        plot = []

        for i in tqdm(plots):
            movie_id.append(i[0])
            plot.append(i[1])

        movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})

        #print("movies:\n", movies.head())

        # change datatype of 'movie_id'
        meta['movie_id'] = meta['movie_id'].astype(str)

        # merge meta with movies
        movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

        #print("movies:\n", movies.head())

        # get genre data
        genres = []
        for i in movies['genre']:
            genres.append(list(json.loads(i).values()))
        movies['genre_new'] = genres

        # remove samples with 0 genre tags
        movies_new = movies[~(movies['genre_new'].str.len() == 0)]

        """
        print("movies:", movies.shape)
        print("movies_new:", movies_new.shape)
        
        print("movies_new:\n", movies_new.head())
        """

        # get all genre tags in a list
        all_genres = sum(genres,[])
        len(set(all_genres))
        #print("all_genres:", all_genres)

        # function for text cleaning
        def clean_text(text):
            # remove backslash-apostrophe
            text = re.sub("\'", "", text)
            # remove everything alphabets
            text = re.sub("[^a-zA-Z]"," ",text)
            # remove whitespaces
            text = ' '.join(text.split())
            # convert text to lowercase
            text = text.lower()
            
            return text

        movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))
        #print("movies_new:", movies_new.head())
        
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        # function to remove stopwords
        def remove_stopwords(text):
            no_stopword_text = [w for w in text.split() if not w in stop_words]
            return ' '.join(no_stopword_text)
        
        movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))
        
        mlb = MultiLabelBinarizer()
        mlb.fit(movies_new['genre_new'])

        # transform target variable
        y = mlb.transform(movies_new['genre_new'])

        # Retrieve target names and number of classes
        target_names = mlb.classes_
        num_classes = len(target_names)

        # split dataset into training and test set
        xtrain, xtest, ytrain, ytest = train_test_split(movies_new['clean_plot'], y, test_size=TEST_SIZE, random_state=seed)

        """
        xtrain = _preprocess(pd.Series(xtrain), remove_punctuation=False)
        xtest = _preprocess(pd.Series(xtest), remove_punctuation=False)  
        """

        return (xtrain.tolist(), ytrain), (xtest.tolist(), ytest), num_classes, target_names, class_type
    else:
        raise ValueError("Unsupported dataset:", name)






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




    
