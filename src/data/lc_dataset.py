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


from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from joblib import Parallel, delayed

from gensim.scripts.glove2word2vec import glove2word2vec

from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, LlamaTokenizerFast, RobertaTokenizerFast
from transformers import BertModel, LlamaModel, RobertaModel

from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1

from model.LCRepresentationModel import *

from util.common import VECTOR_CACHE, DATASET_DIR

import fasttext
import fasttext.util
from scipy.special._precompute.expn_asy import generate_A


# Disable Hugging Face tokenizers parallelism to avoid fork issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"



MIN_DF_COUNT = 5
MAX_VOCAB_SIZE = 15000                                      # max feature size for TF-IDF vectorization
TEST_SIZE = 0.25


NUM_DL_WORKERS = 3      # number of workers to handle DataLoader tasks






nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))


class LCDataset:
    """
    A class to handle loading and preparing datasets for text classification.
    Supports multiple datasets including Reuters, 20 Newsgroups, Ohsumed, RCV1, and WIPO.
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
            self.batch_size = MPS_BATCH_SIZE
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
    
        if (pretrained == 'word2vec':
            print("Using Word2Vec pretrained embeddings...")
            
            self.lcr_model = WordLCRepresentationModel(
                model_name=WORD2VEC_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )
        elif pretrained == 'glove':
            print("Using GloVe pretrained embeddings...")
            
            self.lcr_model = WordLCRepresentationModel(
                model_name=GLOVE_MODEL, 
                model_dir=embedding_path, 
                vtype=vectorization_type
            )

        elif pretrained == 'fasttext':
            print("Using FastText pretrained embeddings with subwords...")
            
            self.lcr_model = SubWordLCRepresentationModel(
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

        elif (pretrained == 'llama'):
            print("Using LLaMA pretrained embeddings...")
            self.lcr_model = LlaMaLCRepresentationModel(
                model_name=LLAMA_MODEL, 
                model_dir=embedding_path,  
                vtype=vectorization_type
            )
        
        else:
            self.lcr_model = None
            raise ValueError(f'Invalid pretrained type: {pretrained}. Failed to instantiate representation model.')

        self.model = self.lcr_model.model
        self.tokenizer = self.lcr_model.tokenizer           # note only transformer models have tokenizers
        self.vectorizer = self.lccr_model.vectorizer

        self.nC = self.num_labels
        print("nC:", self.nC)

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

    

    def vectorize(self, debug=True):
        """
        Build vector representation of data set using TF-IDF or CountVectorizer and constructing 
        the embeddings such that they align with pretrained embeddings tokenization method
        """

        print("\n\t vectorizing dataset...")

        
        if (debug):
            print("model:\n", self.model)
            print("tokenizer:\n", self.tokenizer)
            print("vectorizer:\n", self.vectorizer)
            print("self.vectorization_type:", self.vectorization_type)
            print("self.embedding_type:", self.embedding_type)
        
        print("fitting training data...")
        print("Xtr:", type(self.Xtr), self.Xtr.shape)
        print("Xte:", type(self.Xte), self.Xte.shape)

        # Fit and transform the text data
        self.Xtr_vectorized = self.vectorizer.fit_transform(self.Xtr)
        self.Xte_vectorized = self.vectorizer.transform(self.Xte)
        print("Xtr_vectorized:", type(self.Xtr_vectorized), self.Xtr_vectorized.shape)
        print("Xte_vectorized:", type(self.Xte_vectorized), self.Xte_vectorized.shape)

        """
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

        print("\n\tinitializing embedding matrices...")
        
        self.pretrained_path = self.embedding_path
        print("self.pretrained:", self.pretrained)=
        print("self.pretrained_path:", self.pretrained_path)

        # build the embedding vocabulary matrix to align with the dataset vocabulary and embedding type
        self.build_embedding_vocab_matrix()

        # generate pretrained embedding representation of dataset
        self.generate_dataset_embeddings()

        self.initialized = True

    


    #
    # -------------------------------------------------------------------------------------------------------------
    # embedding vocab construction
    # -------------------------------------------------------------------------------------------------------------
    #



    # -------------------------------------------------------------------------------------------------------------
    # vocab embedding matrix construction helper functions
    #
    def process_batch(self, batch_words):

        #tokenize and prepare inputs
        inputs = self.tokenizer(batch_words, return_tensors='pt', padding=True, truncation=True, max_length=TOKEN_TOKENIZER_MAX_LENGTH)
        
        # move input tensors to proper device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)  # Ensure all inputs are on the same device
        max_vocab_size = self.model.config.vocab_size

        # check for out of range tokens
        out_of_range_ids = input_ids[input_ids >= max_vocab_size]
        if len(out_of_range_ids) > 0:
            print("Warning: The following input IDs are out of range for the model's vocabulary:")
            for out_id in out_of_range_ids.unique():
                token = self.tokenizer.decode(out_id.item())
                print(f"Token '{token}' has ID {out_id.item()} which is out of range (vocab size: {max_vocab_size}).")
        
        # Perform inference, ensuring that the model is on the same device as the inputs
        self.model.to(self.device)
        with torch.no_grad():
            #outputs = model(**inputs)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Return the embeddings and ensure they are on the CPU for further processing
        return outputs.last_hidden_state[:, 0,:].cpu().numpy()
        

    def build_embedding_vocab_matrix_core(self, vocab, batch_size=MPS_BATCH_SIZE):
        
        print("core embedding dataset vocab matrix construction...")
        
        print("vocab:", type(vocab), len(vocab))
        print("batch_size:", batch_size)
        
        embedding_vocab_matrix = np.zeros((len(vocab), self.model.config.hidden_size))
        batch_words = []
        word_indices = []

        with tqdm(total=len(vocab), desc="Processing embedding vocab matrix construction batches") as pbar:
            for word, idx in vocab.items():
                batch_words.append(word)
                word_indices.append(idx)

                if len(batch_words) == batch_size:
                    embeddings = self.process_batch(batch_words)
                    for i, embedding in zip(word_indices, embeddings):
                        if i < len(embedding_vocab_matrix):
                            embedding_vocab_matrix[i] = embedding
                        else:
                            print(f"IndexError: Skipping index {i} as it's out of bounds for embedding_vocab_matrix.")
                    
                    batch_words = []
                    word_indices = []
                    pbar.update(batch_size)

            if batch_words:
                embeddings = self.process_batch(batch_words)
                for i, embedding in zip(word_indices, embeddings):
                    if i < len(embedding_vocab_matrix):
                        embedding_vocab_matrix[i] = embedding
                    else:
                        print(f"IndexError: Skipping index {i} as it's out of bounds for the embedding_vocab_matrix.")
                
                pbar.update(len(batch_words))

        return embedding_vocab_matrix



    def build_embedding_vocab_matrix(self):
        """
        build the vector representation of the dataset vocabulary, ie the representation of the features that 
        we can use to add information to the embeddings that we feed into the model (depending upon 'mode')

        Returns:
        - embedding_vocab_matrix: Matrix of embeddings for the dataset vocabulary.
        """

        print("\n\tconstructing (pretrained) embeddings dataset vocabulary matrix...")    

        print("self.pretrained:", self.pretrained)
        print("self.pretrained_path:", self.pretrained_path)
        
        if (self.pretrained in ['word2vec', 'glove']):
            
            print("word based embeddings...")

            print("model:", type(self.model))
            print("model.vector_size:", self.model.vector_size)

            self.embedding_dim = self.model.vector_size
            self.vocab_size = len(self.vectorizer.vocabulary_)
            print("embedding_dim:", self.embedding_dim)
            print("vocab_size:", self.vocab_size)

            print("creating (pretrained) embedding matrix which aligns with dataset vocabulary...")
            self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))

            # Extract pretrained embeddings for words in the vocabulary
            for word, idx in self.vectorizer.vocabulary_.items():
                
                # For Word2Vec and GloVe (retrieving word vectors)
                if self.pretrained in ['word2vec', 'glove']:
                    if word in self.model.key_to_index:  # Check if the word is in the model's vocabulary
                        self.embedding_vocab_matrix[idx] = self.model[word]
                    else:
                        # Handle OOV words with a random embedding
                        self.embedding_vocab_matrix[idx] = np.random.normal(size=(self.embedding_dim,))
                
                # For FastText (supports subword-level embeddings for OOV words)
                elif self.pretrained == 'fasttext':
                    if word in self.model.wv.key_to_index:  # If the word exists in the FastText vocabulary
                        self.embedding_vocab_matrix[idx] = self.model.wv[word]
                    else:
                        # Get subword-based embedding for OOV words
                        self.embedding_vocab_matrix[idx] = self.model.wv[word]
                    
        elif (self.pretrained == 'fastText'):

            print(f"subword based embeddings...")

            # create the embedding matrix that aligns with the vocabulary
            self.vocab_size = len(self.vectorizer.vocabulary_)
            print('vocab_size:', self.vocab_size)
            self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))

            for word, idx in self.vectorizer.vocabulary_.items():
                # FastText automatically handles subwords, so retrieve the word vector directly
                self.embedding_vocab_matrix[idx] = self.model.get_word_vector(word)


        elif self.pretrained in ['bert', 'roberta', 'llama']:
                        
            print("token based embeddings...
                  
            print("self.model:\n", self.model)
            print("self.tokenizer:\n", self.tokenizer)

            print("Tokenizer vocab size:", self.tokenizer.vocab_size)
            print("Model vocab size:", self.model.config.vocab_size)

            self.model.eval()  # Set model to evaluation mode            
            self.model = self.model.to(self.device)
            print(f"Using device: {self.device}")  # To confirm which device is being used

            #
            # NB we use different embedding vocab matrices here depending upon the pretrained model
            #
            if (self.pretrained in ['bert', 'roberta']):                                                                # BERT and RoBERTa

                self.embedding_dim = self.model.config.hidden_size  # Get the embedding dimension size
                self.vocab_size = len(self.vectorizer.vocabulary_)

                self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))
                
                print("embedding_dim:", self.embedding_dim)
                print("dataset vocab size:", self.vocab_size)
                #print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)         

                self.embedding_vocab_matrix_orig = self.build_embedding_vocab_matrix_core(self.vocab, batch_size=MPS_BATCH_SIZE)
                print("embedding_vocab_matrix_orig:", type(self.embedding_vocab_matrix_orig), self.embedding_vocab_matrix_orig.shape)

                self.embedding_vocab_matrix = self.embedding_vocab_matrix_orig
                
            elif self.pretrained == 'llama':

                print("creating vocabulary list of LLaMA encoded tokens based on the vectorizer vocabulary...")
                
                self.embedding_dim = self.model.config.hidden_size  # Get the embedding dimension size
                self.vocab_size = len(self.vectorizer.vocabulary_)

                print("embedding_dim:", self.embedding_dim)
                print("dataset vocab size:", self.vocab_size)
                #print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)         

                self.llama_vocab_embeddings = {}                # Initialize an empty dictionary to store the embeddings

                # Collect all tokens that need encoding
                # NBL we use the tokenizer's vocabulary 
                # instead of vocab_ndarr (which comes from 
                # TfidfVectorizer features_out)
                tokens = list(self.vectorizer.vocabulary_.keys())       # Get the vocabulary keys (tokens)

                #tokens = list(self.vocab_ndarr)
                num_tokens = len(tokens)

                print("tokens:", type(tokens), len(tokens))
                print("num_tokens:", num_tokens)

                batch_size = BATCH_SIZE

                # Process tokens in batches
                for batch_start in tqdm(range(0, num_tokens, batch_size), desc="batch encoding dataset vocabulary using LLaMa..."):
                    batch_tokens = tokens[batch_start: batch_start + batch_size]
                    
                    # Tokenize the entire batch at once
                    input_ids = self.tokenizer(
                        batch_tokens, 
                        return_tensors='pt', 
                        padding=True, 
                        truncation=True,
                        max_length=TOKEN_TOKENIZER_MAX_LENGTH       # specify max_length for truncation
                    ).input_ids.to(self.device)
                    
                    # Get model outputs for the batch
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                    
                    # Extract and store the embeddings for each token in the batch
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    
                    # Store each embedding in the dictionary
                    for token, embedding in zip(batch_tokens, batch_embeddings):
                        self.llama_vocab_embeddings[token] = embedding

                print("llama_vocab_embeddings:", type(self.llama_vocab_embeddings), len(self.llama_vocab_embeddings))

                # Function to convert llama_vocab_embeddings (dict) to a numpy matrix
                def convert_dict_to_matrix(vocab_embeddings, vocab, embedding_dim):
                    print("converting dict to matrix...")

                    # Assuming all embeddings have the same dimension and it's correctly 4096 as per the LLaMA model dimension
                    embedding_matrix = np.zeros((len(vocab), embedding_dim))  # Shape (vocab_size, embedding_dim)

                    for i, token in enumerate(vocab):
                        if token in vocab_embeddings:
                            # Direct assignment of the embedding which is already in the correct shape (4096,)
                            embedding_matrix[i, :] = vocab_embeddings[token]
                        else:
                            # Initialize missing tokens with zeros or a small random value
                            embedding_matrix[i, :] = np.zeros(embedding_dim)

                    return embedding_matrix

                # Convert the dictionary of embeddings to a matrix
                self.embedding_vocab_matrix_new = convert_dict_to_matrix(self.llama_vocab_embeddings, self.vocab_ndarr, self.embedding_dim)
                print("embedding_vocab_matrix_new:", type(self.embedding_vocab_matrix_new), self.embedding_vocab_matrix_new.shape)

                # Assign the new embedding matrix
                self.embedding_vocab_matrix = self.embedding_vocab_matrix_new

        else:
            raise ValueError("Invalid pretrained type.")
        
        # should be a numpy array 
        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)

        return self.embedding_vocab_matrix
            


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
        if (self.pretrained in ['bert', 'llama', 'roberta']):    

            print("generating transformer / token based dataset representations...")

            self.Xtr_avg_embeddings, self.Xtr_summary_embeddings = self.lcr_model.encode_sentences_opt(self.Xtr.tolist())                                  
            self.Xte_avg_embeddings, self.Xte_summary_embeddings = self.lcr_model.encode_sentences_opt(self.Xte.tolist())

            # not supported weighted average comp method for transformer based models due to
            # complexity of vectorization and tokenization mapping across models 
            # word embedding models like word2vec, GloVe or fasTtext
            self.Xtr_weighted_embeddings = self.Xtr_avg_embeddings
            self.Xte_weighted_embeddings = self.Xte_avg_embeddings
            
        elif (self.pretrained in ['word2vec', 'glove', 'fasttext']):                        # word (and subword) based embeddins
            
            print("generating word / subword based dataset repressentations...")
            
            self.Xtr_weighted_embeddings, self.Xtr_avg_embeddings = lcr_model.encode_docs(self.Xtr.tolist(), self.embedding_vocab_matrix)
            self.self.Xte_weighted_embeddings, self.Xte_avg_embeddings = lcr_model.encode_docs(self.Xte.tolist(), self.embedding_vocab_matrix)

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

        #self.X_raw = train_set['Text'].tolist()
        #self.X_raw = train_set['Text']        
        #print("X_raw:", type(self.X_raw), len(self.X_raw))
        #print("X_raw[0]:\n", self.X_raw[0])
        
        # we split the train data into train and test here because that is
        # not done for us with the BBC News dataset (Test data is not labeled)
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            train_set['Text'], 
            train_set['Category'], 
            train_size = 1-TEST_SIZE, 
            random_state = 1
        )

        print("Train Size =", self.X_train_raw.shape, "\nTest Size = ", self.X_test_raw.shape)

        # preprocess the text (stopword removal, mask numbers, etc)
        self.Xtr = self._preprocess(self.X_train_raw)
        print("Xtr:", type(self.Xtr), self.Xtr.shape)
        print("Xtr[0]:\n", self.Xtr[0])

        self.Xte = self._preprocess(self.X_test_raw)
        print("Xte:", type(self.Xte), self.Xte.shape)
        print("Xte[0]:\n", self.Xte[0])       

        self.devel_raw = self.Xtr
        self.test_raw = self.Xte

        self.devel_target = self.y_train
        self.test_target = self.y_test

        # ** Convert single-label targets to a format suitable for _label_matrix **
        # Each label should be wrapped in a list for compatibility with _label_matrix (even for single-label classification)
        self.devel_target_formatted = [[label] for label in self.devel_target]
        self.test_target_formatted = [[label] for label in self.test_target]

        # Generate label matrix for the training and test sets
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(
            self.devel_target_formatted, self.test_target_formatted
        )

        self.target_names = train_set['Category'].unique()       
        self.label_names = self.target_names           # set self.labels to the class label names        
        self.labels = self.label_names
        self.num_label_names = len(self.label_names)
        self.num_labels = self.num_label_names
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        print("encoding labels...")

        # Encode the labels
        label_encoder = LabelEncoder()
        self.y_train = label_encoder.fit_transform(self.devel_target)
        self.y_test = label_encoder.transform(self.test_target)
        print("self.y_train:", type(self.y_train), self.y_train.shape)
        print("self.y_test:", type(self.y_test), self.y_test.shape)

        # Convert y matrices to sparse matrices
        self.y_train_sparse = csr_matrix(self.y_train).T                                        # Transpose to match the expected shape
        self.y_test_sparse = csr_matrix(self.y_test).T                                          # Transpose to match the expected shape
        print("self.y_train_sparse:", type(self.y_train_sparse), self.y_train_sparse.shape)
        print("self.y_test_sparse:", type(self.y_test_sparse), self.y_test_sparse.shape)

        return self.target_names


    def _load_20news(self):
        
        print("\n\tloading 20newsgroups dataset...")
        
        metadata = ('headers', 'footers', 'quotes')
        
        self.devel = fetch_20newsgroups(subset='train', remove=metadata)
        self.test = fetch_20newsgroups(subset='test', remove=metadata)

        self.classification_type = 'singlelabel'
        self.class_type = 'singlelabel'
        
        # training data
        self.Xtr = self._preprocess(pd.Series(self.devel.data))
        print("self.Xtr:", type(self.Xtr), self.Xtr.shape)
        print("self.Xtr[0]:\n", self.Xtr[0])

        # test data
        self.Xte = self._preprocess(pd.Series(self.test.data))
        print("self.Xte:", type(self.Xte), self.Xte.shape)
        print("self.Xte[0]:\n", self.Xte[0])

        self.devel_raw = self.Xtr
        self.test_raw = self.Xte

        #self.devel_raw, self.test_raw = mask_numbers(self.devel.data), mask_numbers(self.test.data)
        self.devel_target, self.test_target = self.devel.target, self.test.target        
        print("devel_target:", type(self.devel_target), len(self.devel_target))
        print("test_target:", type(self.test_target), len(self.test_target))

        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))
        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        print("self.labels:", type(self.labels), len(self.labels))

        self.label_names = self.devel.target_names           # set self.labels to the class label names
        print("self.label_names:\n", self.label_names) 

        self.target_names = self.label_names
        print("target_names:", type(self.target_names), len(self.target_names))

        self.num_labels = len(self.labels)
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        print("encoding labels...")

        # Encode the labels
        label_encoder = LabelEncoder()
        self.y_train = label_encoder.fit_transform(self.devel_target)
        self.y_test = label_encoder.transform(self.test_target)
        print("self.y_train:", type(self.y_train), self.y_train.shape)
        print("self.y_test:", type(self.y_test), self.y_test.shape)

        # Convert y matrices to sparse matrices
        self.y_train_sparse = csr_matrix(self.y_train).T                                                    # Transpose to match the expected shape
        self.y_test_sparse = csr_matrix(self.y_test).T                                                      # Transpose to match the expected shape
        print("self.y_train_sparse:", type(self.y_train_sparse), self.y_train_sparse.shape)
        print("self.y_test_sparse:", type(self.y_test_sparse), self.y_test_sparse.shape)

        return self.target_names




    def _load_reuters(self):

        print("\n\tloading reuters21578 dataset...")

        data_path = os.path.join(DATASET_DIR, 'reuters21578')
        
        print("data_path:", data_path)  

        self.devel = fetch_reuters21578(subset='train', data_path=data_path)
        self.test = fetch_reuters21578(subset='test', data_path=data_path)

        #print("dev target names:", type(devel), devel.target_names)
        #print("test target names:", type(test), test.target_names)

        self.classification_type = 'multilabel'
        self.class_type = 'multilabel'
        
        # training data
        self.Xtr = self._preprocess(pd.Series(self.devel.data))
        print("self.Xtr:", type(self.Xtr), self.Xtr.shape)
        print("self.Xtr[0]:\n", self.Xtr[0])

        # test data
        self.Xte = self._preprocess(pd.Series(self.test.data))
        print("self.Xte:", type(self.Xte), self.Xte.shape)
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


        """
        self.y = self.devel_target
        print("y:", type(self.y), self.y.shape)

        # Convert Y to a sparse matrix
        self.y_sparse = csr_matrix(self.y)  # No need for transpose
        print("y_sparse:", type(self.y_sparse), self.y_sparse.shape)
        """

        return self.label_names



    def _load_ohsumed(self):

        print("\n\tloading ohsumed dataset...")

        #data_path = os.path.join(get_data_home(), 'ohsumed50k')
        data_path = os.path.join(DATASET_DIR, 'ohsumed50k')

        print("data_path:", data_path)  

        self.classification_type = 'multilabel'
        self.class_type = 'multilabel'
        
        self.devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        self.test = fetch_ohsumed50k(subset='test', data_path=data_path)

        # training data
        self.Xtr = self._preprocess(pd.Series(self.devel.data))
        print("self.Xtr:", type(self.Xtr), self.Xtr.shape)
        print("self.Xtr[0]:\n", self.Xtr[0])

        # test data
        self.Xte = self._preprocess(pd.Series(self.test.data))
        print("self.Xte:", type(self.Xte), self.Xte.shape)
        print("self.Xte[0]:\n", self.Xte[0])

        self.devel_raw = self.Xtr
        self.test_raw = self.Xte

        self.devel_target, self.test_target = self.devel.target, self.test.target
        print("devel_target:", type(self.devel_target), len(self.devel_target))
        print("test_target:", type(self.test_target), len(self.test_target))

        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel.target, self.test.target)
        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        print("labels:\n", self.labels)

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

        print("encoding labels...")
        # Encode the labels using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        self.y_train = mlb.fit_transform(self.devel_target)     # Transform multi-label targets into a binary matrix
        self.y_test = mlb.transform(self.test_target)           # Transform multi-label targets into a binary matrix
        print("self.y_train:", type(self.y_train), self.y_train.shape)
        print("self.y_test:", type(self.y_test), self.y_test.shape)
        
        # Convert Y to a sparse matrix
        self.y_train_sparse = csr_matrix(self.y_train)                                       # without Transpose to match the expected shape
        self.y_test_sparse = csr_matrix(self.y_test)                                         # without Transpose to match the expected shape
        print("self.y_test_sparse:", type(self.y_test_sparse), self.y_test_sparse.shape)
        print("self.y_train_sparse:", type(self.y_train_sparse), self.y_train_sparse.shape)

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

        """
        print("training data:", type(devel))
        print("training data:", type(devel.data), len(devel.data))
        print("training targets:", type(devel.target), len(devel.target))

        print("testing data:", type(test))
        print("testing data:", type(test.data), len(test.data))
        print("testing targets:", type(test.target), len(test.target))
        """
        
        # training data
        self.Xtr = self._preprocess(pd.Series(self.devel.data))
        print("self.Xtr:", type(self.Xtr), self.Xtr.shape)
        print("self.Xtr[0]:\n", self.Xtr[0])

        # test data
        self.Xte = self._preprocess(pd.Series(self.test.data))
        print("self.Xte:", type(self.Xte), self.Xte.shape)
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


        """
        self.y = self.devel_target
        print("y:", type(self.y), self.y.shape)

        # Convert Y to a sparse matrix
        self.y_sparse = csr_matrix(self.y)  # No need for transpose
        print("y_sparse:", type(self.y_sparse), self.y_sparse.shape)
        """

        return self.label_names





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



    def _preprocess(self, text_series: pd.Series):
        """
        Preprocess a pandas Series of texts by removing punctuation and stopwords.

        Parameters:
        - text_series: A pandas Series containing text data (strings).

        Returns:
        - processed_texts: A NumPy array containing processed text strings.
        """
        print("preprocessing text...")
        print("text_series:", type(text_series), text_series.shape)

        # Load stop words once outside the loop
        stop_words = set(stopwords.words('english'))
        punctuation_table = str.maketrans('', '', string.punctuation)  # Translation table to remove punctuation

        # Function to process each text (removing punctuation and stopwords)
        def process_text(text):
            # Remove punctuation
            text = text.translate(punctuation_table)
            
            # Tokenize and remove stopwords
            filtered_words = [word for word in text.lower().split() if word not in stop_words]
            
            # Join back into a single string
            return ' '.join(filtered_words)

        # Use Parallel processing with multiple cores
        processed_texts = Parallel(n_jobs=-1)(delayed(process_text)(text) for text in text_series)

        # Return as NumPy array
        return np.array(processed_texts)
    


    def _preprocess_deprecated(self, text_series):
        """
        Preprocess a pandas Series of texts: tokenizing, removing punctuation, stopwords, 
        and applying a custom number masking function.

        Parameters:
        - text_series: A pandas Series containing text data (strings).

        Returns:
        - processed_texts: A NumPy array containing processed text strings with the shape property.
        """
        print("preprocessing text...")
        print("text_series:", type(text_series), text_series.shape)

        # Load stop words once outside the loop
        stop_words = set(stopwords.words('english'))

        # Vectorize punctuation removal
        text_series = text_series.apply(self._remove_punctuation)

        # Parallelize tokenization and stop word removal using joblib for multi-core processing
        def process_text(text):
            # Tokenize text
            #tokenized_text = word_tokenize(text.lower())
            
            # Remove stop words
            #filtered_words = [word for word in tokenized_text if word not in stop_words]
            filtered_words = [word for word in text if word not in stop_words]
            
            # Join back into a string
            filtered_text = ' '.join(filtered_words)
            
            # Apply custom number masking
            #masked_text = _mask_numbers([filtered_text])  # Input as list to fit mask_numbers function signature
            #return masked_text[0]

            return filtered_text

        # Parallel processing with multiple cores
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
    
            
    @classmethod
    def load_nn(cls, dataset_name, vectorization_type='tfidf', embedding_type='word', base_pickle_path=None):

        print("Dataset::load():", dataset_name, base_pickle_path)

        print("vectorization_type:", vectorization_type)
        print("embedding_type:", embedding_type)

        # Create a pickle path that includes the vectorization type
        # NB we assume the /pickles directory exists already
        if base_pickle_path:
            full_pickle_path = f"{base_pickle_path}{'/'}{dataset_name}_{vectorization_type}.pkl"
            pickle_file_name = f"{dataset_name}_{vectorization_type}.pkl"
        else:
            full_pickle_path = None
            pickle_file_name = None

        print("pickle_file_name:", pickle_file_name)

        # not None so we are going to create the pickle file, 
        # by dataset and vectorization type
        if full_pickle_path:
            print("full_pickle_path: ", {full_pickle_path})

            if os.path.exists(full_pickle_path):                                        # pickle file exists, load it
                print(f'loading pickled dataset from {full_pickle_path}')
                dataset = pickle.load(open(full_pickle_path, 'rb'))
            else:                                                                       # pickle file does not exist, create it, load it, and dump it
                print(f'fetching dataset and dumping it into {full_pickle_path}')
                dataset = LCDataset(name=dataset_name, vectorization_type=vectorization_type, embedding_type=embedding_type)

                print('dumping')
                #pickle.dump(dataset, open(pickle_path, 'wb', pickle.HIGHEST_PROTOCOL))
                # Open the file for writing and write the pickle data
                try:
                    with open(full_pickle_path, 'wb', pickle.HIGHEST_PROTOCOL) as file:
                        pickle.dump(dataset, file)
                    print("data successfully pickled at:", full_pickle_path)
                except Exception as e:
                    print(f'\n\tERROR: Exception raised, failed to pickle data: {e}')

        else:
            print(f'loading dataset {dataset_name}')
            dataset = LCDataset(name=dataset_name, vectorization_type=vectorization_type, embedding_type=embedding_type)

        return dataset

#
# end class
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------



def save_to_pickle(Xtr_raw, Xte_raw, Xtr_vectorized, Xte_vectorized, y_train, y_test, target_names, class_type, embedding_matrix, Xtr_weighted_embeddings, 
                   Xte_weighted_embeddings, Xtr_avg_embeddings, Xte_avg_embeddings, Xtr_summary_embeddings, Xte_summary_embeddings, pickle_file):
    
    print(f'saving {pickle_file} to pickle...')

    # Open the file for writing and write the pickle data
    try:

        # Combine multiple variables into a dictionary
        lc_pt_data = {
            'Xtr_raw': Xtr_raw,
            'Xte_raw': Xte_raw,
            'Xtr_vectorized': Xtr_vectorized,
            'Xte_vectorized': Xte_vectorized,
            'y_train': y_train,
            'y_test': y_test,
            'target_names': target_names,
            'class_type': class_type,
            'embedding_matrix': embedding_matrix,
            'Xtr_weighted_embeddings': Xtr_weighted_embeddings,
            'Xte_weighted_embeddings': Xte_weighted_embeddings,
            'Xtr_avg_embeddings': Xtr_avg_embeddings,
            'Xte_avg_embeddings': Xte_avg_embeddings,
            'Xtr_summary_embeddings': Xtr_summary_embeddings,
            'Xte_summary_embeddings': Xte_summary_embeddings
        }

        with open(pickle_file, 'wb', pickle.HIGHEST_PROTOCOL) as f:
            pickle.dump(lc_pt_data, f)
        
        print("data successfully pickled at:", pickle_file)

        return True

    except Exception as e:
        print(f'Exception raised, failed to pickle data to {pickle_file}: {e}')
        return False
        
   

def load_from_pickle(pickle_file):
    
    print(f"Loading pickle file: {pickle_file}...")

    try:
        # open the pickle file for reading
        with open(pickle_file, 'rb') as f:
            data_loaded = pickle.load(f)                # load the data from the pickle file

        # Access the individual variables
        Xtr_raw = data_loaded['Xtr_raw']
        Xte_raw = data_loaded['Xte_raw']
        Xtr_vectorized = data_loaded['Xtr_vectorized']
        Xte_vectorized = data_loaded['Xte_vectorized']
        y_train = data_loaded['y_train']
        y_test = data_loaded['y_test']
        target_names = data_loaded['target_names']
        class_type = data_loaded['class_type']
        embedding_matrix = data_loaded['embedding_matrix']
        Xtr_weighted_embeddings = data_loaded['Xtr_weighted_embeddings']
        Xte_weighted_embeddings = data_loaded['Xte_weighted_embeddings']
        Xtr_avg_embeddings = data_loaded['Xtr_avg_embeddings']
        Xte_avg_embeddings = data_loaded['Xte_avg_embeddings']
        Xtr_summary_embeddings = data_loaded['Xtr_summary_embeddings']
        Xte_summary_embeddings = data_loaded['Xte_summary_embeddings']

    except EOFError:
        print("\n\tError: Unexpected end of file while reading the pickle file.")
        return None

    print("embedding_matrix:", type(embedding_matrix), embedding_matrix.shape)
    #print("embedding_matrix[0]:\n", embedding_matrix[0])

    return Xtr_raw, Xte_raw, Xtr_vectorized, Xte_vectorized, y_train, y_test, target_names, class_type, embedding_matrix, Xtr_weighted_embeddings, Xte_weighted_embeddings, Xtr_avg_embeddings, Xte_avg_embeddings, Xtr_summary_embeddings, Xte_summary_embeddings



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
    
    print("_label_matrix...")

    mlb = MultiLabelBinarizer(sparse_output=True)
    
    ytr = mlb.fit_transform(tr_target)
    yte = mlb.transform(te_target)

    print("ytr:", type(ytr), ytr.shape)
    print("yte:", type(yte), yte.shape)

    print("MultiLabelBinarizer.classes_:", mlb.classes_)

    return ytr, yte, mlb.classes_


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


def _mask_numbers(data, number_mask='numbermask'):
    """
    Masks numbers in the given text data with a placeholder.
    """
    mask = re.compile(r'\b[0-9][0-9.,-]*\b')
    masked = [mask.sub(number_mask, text) for text in data]

    return masked

    
