import os
import numpy as np
from tqdm import tqdm
import re
import pandas as pd
import string

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1

from pathlib import Path
from urllib import request
import tarfile
import gzip
import shutil

from scipy.sparse import csr_matrix

import pickle

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from gensim.models import KeyedVectors

from contextlib import nullcontext

import torch

from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, LlamaTokenizerFast, RobertaTokenizerFast
from transformers import BertModel, LlamaModel, RobertaModel


VECTOR_CACHE = '../.vector_cache'
DATASET_DIR = '../datasets/'
PICKLE_DIR = '../pickles/'

MAX_VOCAB_SIZE = 10000                                      # max feature size for TF-IDF vectorization

BERT_MODEL = 'bert-base-uncased'                            # dimension = 768
LLAMA_MODEL = 'meta-llama/Llama-2-7b-hf'                    # dimension = 4096
#LLAMA_MODEL = 'meta-llama/Llama-2-13b-hf'
ROBERTA_MODEL = 'roberta-base'                              # dimension = 768

TOKEN_TOKENIZER_MAX_LENGTH = 512

TEST_SIZE = 0.3

# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 16
DEFAULT_GPU_BATCH_SIZE = 16
MPS_BATCH_SIZE = 32

#
# tokens for LLAMA model access, must be requested from huggingface
#
from huggingface_hub import login

HF_TOKEN = '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'
HF_TOKEN2 = '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'


stop_words = set(stopwords.words('english'))



# ------------------------------------------------------------------------------------------------------------------------
# BERT Embeddings functions
# ------------------------------------------------------------------------------------------------------------------------
class TextDataset(Dataset):

    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten()
        }



class LCDataset:
    """
    A class to handle loading and preparing datasets for text classification.
    Supports multiple datasets including Reuters, 20 Newsgroups, Ohsumed, RCV1, and WIPO.
    """

    dataset_available = {'reuters21578', '20newsgroups', 'ohsumed', 'rcv1', 'bbc-news'}
    

    def __init__(self, name, vectorization_type, embedding_type):
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

        print("initializing LCDataset...")

        assert name in LCDataset.dataset_available, f'dataset {name} is not available'

        self.name = name
        print("self.name:", self.name) 

        self.vectorization_type = vectorization_type
        print("vectorization_type:", self.vectorization_type)

        self.embedding_type = embedding_type
        print("embedding_type:", self.embedding_type)

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

        self.vectorize()

        self.nC = self.num_labels
        print("nC:", self.nC)

        self.loaded = True



    def init_embedding_matrices(self, pretrained=None, pretrained_path=None):
        """
        Initialize the dataset with pretrained embeddings.
        
        Parameters:
        - pretrained: 'word2vec', 'glove', 'fasttext', 'bert', 'roberta' or 'llama' for the pretrained embeddings to use.
        - pretrained_path: Path to the pretrained embeddings file.
        """
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        print("pretrained:", self.pretrained)
        print("pretrained_path:", self.pretrained_path)

        # build the embedding vocabulary matrix to align with the dataset vocabulary and embedding type
        self.build_embedding_vocab_matrix()

        # generate pretrained embedding representation of dataset
        self.generate_dataset_embeddings()

        self.initialized = True

    

    def show(self):
        nTr_docs = len(self.devel_raw)
        nTe_docs = len(self.test_raw)
        #nfeats = len(self._vectorizer.vocabulary_)
        nfeats = len(self.vectorizer.vocabulary_)
        nC = self.devel_labelmatrix.shape[1]
        nD=nTr_docs+nTe_docs
        print(f'{self.classification_type}, nD={nD}=({nTr_docs}+{nTe_docs}), nF={nfeats}, nC={nC}')
        return self

    # Function to remove stopwords before tokenization
    def remove_stopwords(self, texts):
        
        print("removing stopwords...")
        
        filtered_texts = []
        for text in texts:
            filtered_words = [word for word in text.split() if word.lower() not in stop_words]
            filtered_texts.append(" ".join(filtered_words))
        return filtered_texts

    def custom_tokenizer(self, text):
        # Tokenize the text using the tokenizer with truncation
        return self.tokenizer.tokenize(text, max_length=TOKEN_TOKENIZER_MAX_LENGTH, truncation=True)
    
    
    def vectorize(self):
    
        """
        Build vector representation of data set using TF-IDF or CountVectorizer and constructing 
        the embeddings such that they align with pretrained embeddings tokenization method
        """

        print("building vector representation of dataset...")

        # initialize local variables
        self.model = None
        self.embedding_dim = 0
        self.embedding_vocab_matrix = None
        self.vocab = None

        print("self.vectorization_type:", self.vectorization_type)
        print("self.embedding_type:", self.embedding_type)
        
        # Choose the vectorization and tokenization strategy based on embedding type
        if self.embedding_type == 'word':

            print("Using word-level vectorization...")
            
            if self.vectorization_type == 'tfidf':
                print("using TFIDS vectorization...")
                self.vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE)
            elif self.vectorization_type == 'count':
                print("using Count vectorization...")
                self.vectorizer = CountVectorizer(max_features=MAX_VOCAB_SIZE)
            else:
                raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")
            
            # Fit and transform the text data to obtain tokenized features
            # NB we need to vectorize training and test data, both of which
            # are loaded when the dataset is initialized 
            self.Xtr_vectorized = self.vectorizer.fit_transform(self.Xtr)                       
            self.Xte_vectorized = self.vectorizer.transform(self.Xte)                           
        elif self.embedding_type == 'token':
            
            print(f"Using token-level vectorization with {self.pretrained.upper()} embeddings...")

            if self.pretrained == 'bert': 
                self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT')
                self.model = BertModel.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT').to(self.device)
            elif self.pretrained == 'roberta':
                self.tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_MODEL, cache_dir=VECTOR_CACHE+'/RoBERTa')
                self.model = RobertaModel.from_pretrained(ROBERTA_MODEL, cache_dir=VECTOR_CACHE+'/RoBERTa').to(self.device)
            elif self.pretrained == 'llama':
                self.tokenizer = LlamaTokenizerFast.from_pretrained(LLAMA_MODEL, cache_dir=VECTOR_CACHE+'/LLaMa')
                self.model = LlamaModel.from_pretrained(LLAMA_MODEL, cache_dir=VECTOR_CACHE+'/LLaMa').to(self.device)
            else:
                raise ValueError("Invalid embedding type. Use 'bert', 'roberta', or 'llama' for token embeddings.")
 
            print("model:\n", self.model)
            print("tokenizer:\n", self.tokenizer)

            # Ensure padding token is available
            if self.tokenizer.pad_token is None:
                #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id                   # Reuse the end-of-sequence token for padding

            if self.vectorization_type == 'tfidf':
                self.vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE, tokenizer=(self.custom_tokenizer))
            elif self.vectorization_type == 'count':
                self.vectorizer = CountVectorizer(max_features=MAX_VOCAB_SIZE, tokenizer=self.custom_tokenizer)
            else:
                raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")

            print("fitting training data... X type and length:", type(self.X), len(self.X))

            # Fit and transform the text data to obtain tokenized features, 
            # NB we must vectorize both the training and test data which are 
            # loaded when the dataset is initialized  
            self.Xtr_vectorized = self.vectorizer.fit_transform(self.Xtr)
            self.Xte_vectorized = self.vectorizer.fit_transform(self.Xte)
        else:
            raise ValueError("Invalid embedding type. Use 'word' for word embeddings or 'token' for BERT/LLaMa embeddings.")

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
        #self.vocab_dict = {k.lower(): v for k, v in self.vectorizer.vocabulary_.items()}              # Ensure the vocabulary is all lowercased to avoid case mismatches
        #print("vocab_dict:", type(self.vocab_dict), len(self.vocab_dict))    
        self.vocab_ndarr = self.vectorizer.get_feature_names_out()
        print("vocab_ndarr:", type(self.vocab_ndarr), len(self.vocab_ndarr))
        #self.vocab = self.vocab_dict
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


    def generate_dataset_embeddings(self):
        """
        Generate embedding representation of dataset docs in three forms for both the training preprocessed data (Xtr)
        and the test preprocessed data (Xte). The three forms are:

        - weighted_embeddings: Weighted average of word embeddings using TF-IDF scores.
        - summary_embeddings: CLS token embeddings from BERT or RoBERTa models.
        - avg_embeddings: Average of token embeddings from BERT or RoBERTa models.
        
        Returns:
        - Xtr_weighted_embeddings: Weighted average embeddings for training data.
        - Xte_weighted_embeddings: Weighted average embeddings for test data.
        - Xtr_summary_embeddings: CLS token embeddings for training data.
        - Xte_summary_embeddings: CLS token embeddings for test data.
        - Xtr_avg_embeddings: Average embeddings for training data.
        - Xte_avg_embeddings: Average embeddings for test data.
        """
        
        print("generating dataset embedding representation forms...")
        
        print("self.pretrained:", self.pretrained)
        print("self.pretrained_path:", self.pretrained_path)

        if (self.pretrained in ['bert', 'llama', 'roberta']):

            # BERT or RoBERTa embeddings        
            if (self.pretrained in ['bert', 'roberta']): 

                print("generating BERT or RoBERTa embeddings...")

                self.Xtr_weighted_embeddings = self.get_weighted_transformer_embeddings(
                    texts=self.Xtr, 
                    batch_size=self.batch_size, 
                    max_len=TOKEN_TOKENIZER_MAX_LENGTH
                    )

                self.Xte_weighted_embeddings = self.get_weighted_transformer_embeddings(
                    texts=self.Xte, 
                    batch_size=self.batch_size, 
                    max_len=TOKEN_TOKENIZER_MAX_LENGTH
                    )

                print("Xtr_weighted_embeddings:", type(self.Xtr_weighted_embeddings), self.Xtr_weighted_embeddings.shape)
                print("Xte_weighted_embeddings:", type(self.Xte_weighted_embeddings), self.Xte_weighted_embeddings.shape)

                self.Xtr_avg_embeddings = self.get_avg_transformer_embeddings(
                    texts=self.Xtr, 
                    batch_size=self.batch_size, 
                    max_len=TOKEN_TOKENIZER_MAX_LENGTH
                    )

                self.Xte_avg_embeddings = self.get_avg_transformer_embeddings(
                    texts=self.Xte, 
                    batch_size=self.batch_size, 
                    max_len=TOKEN_TOKENIZER_MAX_LENGTH
                    )

                print("Xtr_avg_embeddings:", type(self.Xtr_avg_embeddings), self.Xtr_avg_embeddings.shape)
                print("Xte_avg_embeddings:", type(self.Xte_avg_embeddings), self.Xte_avg_embeddings.shape)

                self.Xtr_summary_embeddings = self.get_transformer_embedding_cls(
                    texts=self.Xtr, 
                    batch_size=self.batch_size, 
                    max_len=TOKEN_TOKENIZER_MAX_LENGTH
                    )

                self.Xte_summary_embeddings = self.get_transformer_embedding_cls(
                    texts=self.Xte, 
                    batch_size=self.batch_size, 
                    max_len=TOKEN_TOKENIZER_MAX_LENGTH
                    )

                print("Xtr_summary_embeddings (cls):", type(self.Xtr_summary_embeddings), self.Xtr_summary_embeddings.shape)
                print("Xte_summary_embeddings (cls):", type(self.Xte_summary_embeddings), self.Xte_summary_embeddings.shape)


            # LLaMa embeddings
            elif (self.pretrained == 'llama'):
                
                print("generating LlaMa embeddings...")

                # Generate the weighted average embeddings for the dataset
                self.Xtr_weighted_embeddings_new = self.get_weighted_llama_embeddings(
                    self.Xtr.toarray(),
                    self.llama_vocab_embeddings,
                    self.vocab_ndarr
                    )

                # Generate the weighted average embeddings for the dataset
                self.Xte_weighted_embeddings_new = self.get_weighted_llama_embeddings(
                    self.Xte.toarray(),
                    self.llama_vocab_embeddings,
                    self.vocab_ndarr
                    )

                print("Xtr_weighted_embeddings_new:", type(self.Xtr_weighted_embeddings_new), self.Xtr_weighted_embeddings_new.shape)
                print("Xte_weighted_embeddings_new:", type(self.Xte_weighted_embeddings_new), self.Xte_weighted_embeddings_new.shape)
                
                self.Xtr_weighted_embeddings = self.Xtr_weighted_embeddings_new
                self.Xte_weighted_embeddings = self.Xte_weighted_embeddings_new

                print("Xtr_weighted_embeddings:", type(self.Xtr_weighted_embeddings), self.Xtr_weighted_embeddings.shape)
                print("Xte_weighted_embeddings:", type(self.Xte_weighted_embeddings), self.Xte_weighted_embeddings.shape)

                self.Xtr_avg_embeddings = self.get_avg_llama_embeddings(
                    texts=self.Xtr, 
                    batch_size=self.batch_size, 
                    max_len=TOKEN_TOKENIZER_MAX_LENGTH
                    )  
                
                self.Xte_avg_embeddings = self.get_avg_llama_embeddings(
                    texts=self.Xte, 
                    batch_size=self.batch_size, 
                    max_len=TOKEN_TOKENIZER_MAX_LENGTH
                    )  

                print("Xtr_avg_embeddings:", type(self.Xtr_avg_embeddings), self.Xtr_avg_embeddings.shape)
                print("Xte_avg_embeddings:", type(self.Xte_avg_embeddings), self.Xte_avg_embeddings.shape)

                self.Xtr_summary_embeddings = self.Xtr_avg_embeddings
                self.Xte_summary_embeddings = self.Xte_avg_embeddings

                print("Xtr_summary_embeddings set to avg_embeddings:", type(self.Xtr_summary_embeddings), self.Xtr_summary_embeddings.shape)
                print("Xte_summary_embeddings set to avg_embeddings:", type(self.Xte_summary_embeddings), self.Xte_summary_embeddings.shape)

        # word based embeddins
        else:
            
            print("generating word embeddings...")

            self.Xtr_weighted_embeddings = self.get_weighted_word_embeddings(
                self.Xtr, 
                self.vectorizer, 
                self.embedding_vocab_matrix
                )

            self.Xte_weighted_embeddings = self.get_weighted_word_embeddings(
                self.Xte, 
                self.vectorizer, 
                self.embedding_vocab_matrix
                )

            print("Xtr_weighted_embeddings:", type(self.Xtr_weighted_embeddings), self.Xtr_weighted_embeddings.shape)
            print("Xte_weighted_embeddings:", type(self.Xte_weighted_embeddings), self.Xte_weighted_embeddings.shape)

            # Word-based embeddings
            self.Xtr_avg_embeddings = self.get_avg_word_embeddings(
                self.Xtr, 
                self.vectorizer, 
                self.embedding_vocab_matrix
                )

            self.Xte_avg_embeddings = self.get_avg_word_embeddings(
                self.Xte, 
                self.vectorizer, 
                self.embedding_vocab_matrix
                )

            print("Xtr_avg_embeddings:", type(self.Xtr_avg_embeddings), self.Xtr_avg_embeddings.shape)
            print("Xte_avg_embeddings:", type(self.Xte_avg_embeddings), self.Xte_avg_embeddings.shape)

            # CLS token summary embeddings not supported in pretrained 
            # word embedding models like word2vec, GloVe or fasTtext
            self.Xtr_summary_embeddings = self.Xtr_avg_embeddings
            self.Xte_summary_embeddings = self.Xte_avg_embeddings

            print("Xtr_summary_embeddings set to Xtr_avg_embeddings:", type(self.Xtr_summary_embeddings), self.Xtr_summary_embeddings.shape)
            print("Xte_summary_embeddings set to Xte_avg_embeddings:", type(self.Xte_summary_embeddings), self.Xte_summary_embeddings.shape)

        return self.Xtr_weighted_embeddings, self.Xte_weighted_embeddings, self.Xtr_summary_embeddings, self.Xte_summary_embeddings, self.Xtr_avg_embeddings, self.Xte_avg_embeddings
        



    def get_avg_word_embeddings(self, texts, vectorizer, embedding_vocab_matrix):
        """
        Compute document embeddings by averaging the word embeddings for each token in the document.

        Args:
        - texts: List of input documents (as raw text).
        - vectorizer: Fitted vectorizer (e.g., TF-IDF) that contains the vocabulary.
        - embedding_vocab_matrix: Matrix of pre-trained word embeddings where each row corresponds to a word in the vocabulary.

        Returns:
        - document_embeddings: Numpy array of averaged document embeddings for each document.
        """
        
        print("Calculating averaged word embeddings...")

        document_embeddings = []

        for doc in texts:
            # Tokenize the document
            tokens = doc.split()
            token_ids = [vectorizer.vocabulary_.get(token.lower(), None) for token in tokens]

            # Initialize the weighted sum and count of valid tokens
            valid_embeddings = []
            
            for token, token_id in zip(tokens, token_ids):
                if token_id is not None and 0 <= token_id < embedding_vocab_matrix.shape[0]:
                    valid_embeddings.append(embedding_vocab_matrix[token_id])

            # Compute the average of embeddings if there are valid tokens
            if valid_embeddings:
                avg_embedding = np.mean(valid_embeddings, axis=0)
            else:
                avg_embedding = np.zeros(embedding_vocab_matrix.shape[1])  # Handle empty or OOV cases

            document_embeddings.append(avg_embedding)

        return np.array(document_embeddings)


    
    def get_weighted_word_embeddings(self, texts, vectorizer, embedding_vocab_matrix):
        """
        Calculate document embeddings by weighting word embeddings using TF-IDF scores for each word in the document.
        For each document, it tokenizes the text, retrieves the corresponding embeddings from (pretrained) 
        embedding_matrix, and weights them by their TF-IDF scores.

        Args:
        - texts: List of input documents (as raw text).
        - vectorizer: TF-IDF vectorizer trained on the text corpus.
        - embedding_vocab_matrix: Pre-trained word embedding matrix where rows correspond to words and columns to embedding dimensions.

        Returns:
        - document_embeddings: Numpy array where each row is a document's (weighted) embedding.
        """

        print("get_weighted_word_embeddings...")
        
        document_embeddings = []
        
        for doc in texts:
            # Tokenize the document and get token IDs
            tokens = doc.split()
            token_ids = [vectorizer.vocabulary_.get(token.lower(), None) for token in tokens]

            # Calculate TF-IDF weights for the tokens
            tfidf_vector = vectorizer.transform([doc]).toarray()[0]

            # Aggregate the embeddings weighted by TF-IDF
            weighted_sum = np.zeros(embedding_vocab_matrix.shape[1])
            total_weight = 0.0

            for token, token_id in zip(tokens, token_ids):
                if token_id is not None and 0 <= token_id < embedding_vocab_matrix.shape[0]:
                    weight = tfidf_vector[token_id]
                    weighted_sum += embedding_vocab_matrix[token_id] * weight
                    total_weight += weight

            if total_weight > 0:
                doc_embedding = weighted_sum / total_weight
            else:
                doc_embedding = np.zeros(embedding_vocab_matrix.shape[1])

            document_embeddings.append(doc_embedding)

        return np.array(document_embeddings)


    def get_bert_embedding_cls(self, texts, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
        """
        Gets the aasociated BERT embeddings for a given input text(s) using just the CLS 
        Token Embeddings as a summmary of the related doc (text), which is the first token 
        in the BERT input sequence. NB: the BERT model is trained so that the [CLS] token 
        contains a summary of the entire sequence, making it suitable for tasks like classification.
        
        Args:
        - texts: List of input text documents.
        - batch_size: Number of samples to process in a single batch during BERT inference.
        - max_len: Maximum sequence length for tokenizing the documents.

        Returns:
        - A 2D numpy array where each row corresponds to a document's averaged embedding.
        """
         
        print("getting CLS BERT embeddings...")
        
        dataset = TextDataset(texts, self.tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        embeddings = []

        self.model.to(self.device)
        self.model.eval()

        with torch.cuda.amp.autocast(), torch.no_grad():
            for batch in tqdm(dataloader, desc="building BERT embeddings for dataset..."):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                
                # take only the embedding for the first token ([CLS]), 
                # which is expected to encode the entire document's context.
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

                embeddings.append(cls_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)

        return embeddings


    def get_avg_bert_embeddings(self, texts, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
        """
        Compute document embeddings using a pretrained BERT model by averaging token embeddings.

        Args:
        - texts: List of input text documents.
        - batch_size: Number of samples to process in a single batch during BERT inference.
        - max_len: Maximum sequence length for tokenizing the documents.

        Returns:
        - A 2D numpy array where each row corresponds to a document's averaged embedding.
        """

        print("getting averaged bert embeddings (without TF-IDF weights)...")

        # Create a dataset from the input texts and prepare it for batch processing.
        dataset = TextDataset(texts, self.tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Initialize a list to store the embeddings for each document.
        embeddings = []

        # Move the model to the appropriate device (CPU, GPU, etc.) and set it to evaluation mode.
        self.model.to(self.device)
        self.model.eval()

        # Disable gradient computation and enable mixed precision for inference (if available on GPU).
        with torch.cuda.amp.autocast(), torch.no_grad():
            # Iterate over the dataset in batches.
            for batch in tqdm(dataloader, desc="building BERT embeddings for dataset..."):
                # Move the batch data to the device (input IDs, attention mask, and token type IDs).
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)

                # Pass the batch through the BERT model to get the token embeddings.
                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

                # Iterate over each document in the batch.
                for i in range(token_embeddings.size(0)):
                    # Average the token embeddings across the sequence length dimension.
                    # This produces a single vector per document by averaging all token embeddings.
                    doc_embedding = token_embeddings[i].mean(dim=0).cpu().numpy()

                    # Append the document's embedding to the list of embeddings.
                    embeddings.append(doc_embedding)

        # Return a 2D numpy array where each row is a document's final averaged embedding.
        return np.vstack(embeddings)


    def get_weighted_bert_embeddings(self, texts, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
        """
        Compute weighted document embeddings using a pretrained BERT model and TF-IDF weights.

        Args:
        - texts: List of input text documents.
        - batch_size: Number of samples to process in a single batch during BERT inference.
        - max_len: Maximum sequence length for tokenizing the documents.

        Returns:
        - A 2D numpy array where each row corresponds to a document's weighted embedding.
        """
        print("getting weighted bert embeddings...")
        
        dataset = TextDataset(texts, self.tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        embeddings = []

        self.model.to(self.device)
        self.model.eval()

        with torch.cuda.amp.autocast(), torch.no_grad():
            for batch in tqdm(dataloader, desc="building weighted BERT embeddings for dataset..."):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

                for i in range(token_embeddings.size(0)):  # Iterate over each document in the batch
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                    tfidf_vector = self.vectorizer.transform([texts[i]]).toarray()[0]

                    weighted_sum = np.zeros(token_embeddings.size(2))  # Initialize weighted sum for this document
                    total_weight = 0.0

                    for j, token in enumerate(tokens):
                        if token in self.vectorizer.vocabulary_:  # Only consider tokens in the vectorizer's vocab
                            token_weight = tfidf_vector[self.vectorizer.vocabulary_[token]]
                            weighted_sum += token_embeddings[i, j].cpu().numpy() * token_weight
                            total_weight += token_weight

                    if total_weight > 0:
                        doc_embedding = weighted_sum / total_weight  # Normalize by the sum of weights
                    else:
                        doc_embedding = np.zeros(token_embeddings.size(2))  # Handle cases with no valid tokens

                    embeddings.append(doc_embedding)

        return np.vstack(embeddings)


    def get_weighted_transformer_embeddings(self, texts, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
        """
        Compute weighted document embeddings using a pretrained BERT or RoBERTa model and TF-IDF weights.

        Args:
        - texts: List of input text documents.
        - batch_size: Number of samples to process in a single batch during BERT/Roberta inference.
        - max_len: Maximum sequence length for tokenizing the documents.

        Returns:
        - A 2D numpy array where each row corresponds to a document's weighted embedding.
        """
        print(f"Getting weighted {'BERT' if self.pretrained == 'bert' else 'RoBERTa'} embeddings...")

        dataset = TextDataset(texts, self.tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        embeddings = []

        self.model.to(self.device)  # Ensure the model is on the correct device
        self.model.eval()

        with torch.no_grad():  # Disable gradient computation
            for batch in tqdm(dataloader, desc=f"Building weighted {'BERT' if self.pretrained == 'bert' else 'RoBERTa'} embeddings..."):
                input_ids = batch['input_ids'].to(self.device)  # Move input IDs to device
                attention_mask = batch['attention_mask'].to(self.device)  # Move attention mask to device
                token_type_ids = batch['token_type_ids'].to(self.device)

                #outputs = self.model(input_ids, attention_mask=attention_mask)
                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

                for i in range(token_embeddings.size(0)):  # Iterate over each document in the batch
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                    tfidf_vector = self.vectorizer.transform([texts[i]]).toarray()[0]

                    weighted_sum = np.zeros(token_embeddings.size(2))  # Initialize weighted sum for this document
                    total_weight = 0.0

                    for j, token in enumerate(tokens):
                        if token in self.vectorizer.vocabulary_:  # Only consider tokens in the vectorizer's vocab
                            #token_weight = tfidf_vector[self.vectorizer.vocabulary_[token.lower()]]
                            token_weight = tfidf_vector[self.vectorizer.vocabulary_[token]]
                            weighted_sum += token_embeddings[i, j].cpu().numpy() * token_weight
                            total_weight += token_weight

                    if total_weight > 0:
                        doc_embedding = weighted_sum / total_weight  # Normalize by the sum of weights
                    else:
                        doc_embedding = np.zeros(token_embeddings.size(2))  # Handle cases with no valid tokens

                    embeddings.append(doc_embedding)

        return np.vstack(embeddings)


    def get_avg_transformer_embeddings(self, texts, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
        """
        Compute document embeddings using a pretrained BERT or RoBERTa model by averaging token embeddings.

        Args:
        - texts: List of input text documents.
        - batch_size: Number of samples to process in a single batch during BERT/Roberta inference.
        - max_len: Maximum sequence length for tokenizing the documents.

        Returns:
        - A 2D numpy array where each row corresponds to a document's averaged embedding.
        """

        print(f"Getting averaged {'BERT' if self.pretrained == 'bert' else 'RoBERTa'} embeddings...")

        dataset = TextDataset(texts, self.tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        embeddings = []

        self.model.to(self.device)  # Move model to correct device
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Building averaged {'BERT' if self.pretrained == 'bert' else 'RoBERTa'} embeddings..."):
                input_ids = batch['input_ids'].to(self.device)  # Move input IDs to device
                attention_mask = batch['attention_mask'].to(self.device)  # Move attention mask to device

                outputs = self.model(input_ids, attention_mask=attention_mask)
                token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

                for i in range(token_embeddings.size(0)):
                    doc_embedding = token_embeddings[i].mean(dim=0).cpu().numpy()  # Average over sequence length
                    embeddings.append(doc_embedding)

        return np.vstack(embeddings)


    def get_transformer_embedding_cls(self, texts, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
        """
        Gets the associated BERT or RoBERTa embeddings for a given input text(s) using just the CLS 
        Token Embeddings as a summary of the related doc (text), which is the first token 
        in the BERT/Roberta input sequence.

        Args:
        - texts: List of input text documents.
        - batch_size: Number of samples to process in a single batch during BERT/Roberta inference.
        - max_len: Maximum sequence length for tokenizing the documents.

        Returns:
        - A 2D numpy array where each row corresponds to a document's CLS token embedding.
        """

        print(f"Getting CLS {'BERT' if self.pretrained == 'bert' else 'RoBERTa'} embeddings...")

        dataset = TextDataset(texts, self.tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        embeddings = []

        self.model.to(self.device)  # Move model to correct device
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Building CLS {'BERT' if self.pretrained == 'bert' else 'RoBERTa'} embeddings..."):
                input_ids = batch['input_ids'].to(self.device)  # Move input IDs to device
                attention_mask = batch['attention_mask'].to(self.device)  # Move attention mask to device

                outputs = self.model(input_ids, attention_mask=attention_mask)

                # Take only the embedding for the first token ([CLS]), which encodes the entire document's context.
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

                embeddings.append(cls_embeddings.cpu().numpy())  # Move to CPU before appending

        return np.vstack(embeddings)
        


    def get_avg_llama_embeddings(self, texts, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
        """
        Compute LLaMA document embeddings by averaging token embeddings for each document.
        """

        print("getting average LLaMa embeddings...")

        # Convert texts into dataset and dataloader
        dataset = TextDataset(texts, self.tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=2)

        embeddings = []

        self.model.to(self.device)
        self.model.eval()

        # Disable gradient computation
        with torch.no_grad():
            # Use autocast for CUDA devices to improve performance
            autocast_context = torch.cuda.amp.autocast if self.device.type == 'cuda' else nullcontext()

            with autocast_context():
                for batch in tqdm(dataloader, desc="building LLaMa average embeddings for dataset..."):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    # Get the model outputs (token embeddings)
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

                    # Average the token embeddings across the sequence length axis
                    avg_embeddings = token_embeddings.mean(dim=1)  # Shape: (batch_size, hidden_dim)

                    # Convert to numpy and append to list
                    embeddings.append(avg_embeddings.cpu().numpy())

        # Stack the individual embeddings into a single numpy array
        embeddings = np.vstack(embeddings)

        return embeddings


    def get_weighted_llama_embeddings(self, texts, vocab_embeddings, vocab):
        """
        Projects TF-IDF vectors into LLaMA embedding space using tfidf vectors and pretrained embeddings
        thereby generating dense LLaMA embeddings for the input text data using weighted TF-IDF scores.

        Parameters:
        - texts: TF-IDF vectorized text (sparse matrix).
        - vocab_embeddings: Dictionary mapping vocab terms to LLaMA embeddings.
        - vocab: Vocabulary list corresponding to TF-IDF vectors.

        Returns:
        - embedded_vectors: Dense LLaMA embeddings projected from TF-IDF vectors.
        """

        print("getting weighted llama embeddings...")
            
        print("tfidf_vectors:", type(texts), texts.shape)
        print("vocab_embeddings:", type(vocab_embeddings), len(vocab_embeddings))
        print("vocab:", type(vocab), vocab.shape)
        
        # Initialize an empty matrix for the final projected embeddings.
        embedded_vectors = np.zeros((texts.shape[0], list(vocab_embeddings.values())[0].shape[1]))
        print("embedded_vectors:", type(embedded_vectors), embedded_vectors.shape)
        
        # Iterate through each document.
        # Add tqdm progress bar for iterating over documents
        for i, doc in tqdm(enumerate(texts), total=texts.shape[0], desc="converting vectorized text into LlaMa embedding (vocabulary) space..."):
            # For each word in the vocabulary, project into LLaMa embedding space using the weighted TF-IDF score.
            for j, token in enumerate(vocab):
                if token in vocab_embeddings:
                    # Multiply TF-IDF value by the corresponding LLaMa embedding and sum for each document.
                    embedded_vectors[i] += doc[j] * vocab_embeddings[token].squeeze()
        
        return embedded_vectors



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
        
        # Load the pre-trained embeddings based on the specified model
        if self.pretrained in ['word2vec', 'fasttext', 'glove']:

            if (self.pretrained == 'word2vec'):
                print("Using Word2Vec pretrained embeddings...")
                self.model = KeyedVectors.load_word2vec_format(self.pretrained_path, binary=True)
            elif self.pretrained == 'glove':
                print("Using GloVe pretrained embeddings...")
                from gensim.scripts.glove2word2vec import glove2word2vec
                glove_input_file = self.pretrained_path
                word2vec_output_file = glove_input_file + '.word2vec'
                glove2word2vec(glove_input_file, word2vec_output_file)
                self.model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
            elif self.pretrained == 'fasttext':
                print("Using fastText pretrained embeddings...")
                self.model = KeyedVectors.load_word2vec_format(self.pretrained_path)
            
            self.embedding_dim = self.model.vector_size
            #print("embedding_dim:", embedding_dim)

            print("creating (pretrained) embedding matrix which aligns with dataset vocabulary...")
            
            # Create the embedding matrix that aligns with the TF-IDF vocabulary
            #self.vocab_size = self.X_vectorized.shape[1]
            self.vocab_size = len(self.vectorizer.vocabulary_)
            self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))
            
            print("embedding_dim:", self.embedding_dim)
            print("vocab_size:", self.vocab_size)
            print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
            
            # Extract the pretrained embeddings for words in the TF-IDF vocabulary
            for word, idx in self.vocab.items():
                if word in self.model:
                    self.embedding_vocab_matrix[idx] = self.model[word]
                else:
                    # If the word is not found in the pretrained model, use a random vector or zeros
                    self.embedding_vocab_matrix[idx] = np.random.normal(size=(self.embedding_dim,))
        
        elif self.pretrained in ['bert', 'roberta', 'llama']:
            
            # NB: tokenizer and model should be initialized in the embedding_type == 'token' block
            
            print("creating (pretrained) embedding vocab matrix which aligns with dataset vocabulary...")
            
            print("Tokenizer vocab size:", self.tokenizer.vocab_size)
            print("Model vocab size:", self.model.config.vocab_size)

            """
            # Ensure padding token is available
            if tokenizer.pad_token is None:
                #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                tokenizer.pad_token_id = tokenizer.eos_token_id  # Reuse the end-of-sequence token for padding
            """
            
            self.model.eval()  # Set model to evaluation mode

            self.embedding_dim = self.model.config.hidden_size  # Get the embedding dimension size
            self.vocab_size = len(self.vectorizer.vocabulary_)

            """
            if self.pretrained in ['bert', 'roberta']:
                self.vocab_size = len(self.vocab_dict)
            elif self.pretrained == 'llama':
                self.vocab_size = len(self.vocab_ndarr)
            """

            self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))
            
            print("embedding_dim:", self.embedding_dim)
            print("dataset vocab size:", self.vocab_size)
            #print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)         

            self.embedding_vocab_matrix_orig = self.build_embedding_vocab_matrix_core(self.vocab, batch_size=MPS_BATCH_SIZE)
            print("embedding_vocab_matrix_orig:", type(self.embedding_vocab_matrix_orig), self.embedding_vocab_matrix_orig.shape)
                
            #
            # NB we use different embedding vocab matrices here depending upon the pretrained model
            #
            if (self.pretrained in ['bert', 'roberta']):
                self.embedding_vocab_matrix = self.embedding_vocab_matrix_orig
            elif (self.pretrained == 'llama'):
                print("creating vocabulary list of LLaMA encoded tokens based on the vectorizer vocabulary...")
                
                self.model = self.model.to(self.device)
                print(f"Using device: {self.device}")  # To confirm which device is being used

                self.llama_vocab_embeddings = {}
                for token in tqdm(self.vocab_ndarr, desc="encoding Vocabulary using LlaMa pretrained embeddings..."):
                    input_ids = self.tokenizer.encode(token, return_tensors='pt').to(self.device)
                    with torch.no_grad():
                        output = self.model(input_ids)
                    self.llama_vocab_embeddings[token] = output.last_hidden_state.mean(dim=1).cpu().numpy()
            
                print("llama_vocab_embeddings:", type(self.llama_vocab_embeddings), len(self.llama_vocab_embeddings))
            
                # Function to convert llama_vocab_embeddings (dict) to a numpy matrix
                def convert_dict_to_matrix(vocab_embeddings, vocab, embedding_dim):
                    
                    print("converting dict to matrix...")
                    
                    # Assuming all embeddings have the same dimension and it's correctly 4096 as per the LLaMA model dimension
                    embedding_dim = embedding_dim
                    embedding_matrix = np.zeros((len(vocab), embedding_dim))  # Shape (vocab_size, embedding_dim)

                    print("embedding_dim:", embedding_dim)
                    print("embedding_matrix:", type(embedding_matrix), embedding_matrix.shape)
                    
                    for i, token in enumerate(vocab):
                        if token in vocab_embeddings:
                            # Direct assignment of the embedding which is already in the correct shape (4096,)
                            embedding_matrix[i, :] = vocab_embeddings[token]
                        else:
                            # Initialize missing tokens with zeros or a small random value
                            embedding_matrix[i, :] = np.zeros(embedding_dim)

                    return embedding_matrix

                self.embedding_vocab_matrix_new = convert_dict_to_matrix(self.llama_vocab_embeddings, self.vocab_ndarr, self.embedding_dim)
                print("embedding_vocab_matrix_new:", type(self.embedding_vocab_matrix_new), self.embedding_vocab_matrix_new.shape)
                
                self.embedding_vocab_matrix = self.embedding_vocab_matrix_new
        else:
            raise ValueError("Invalid pretrained type.")
        
        # should be a numpy array 
        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)

        return self.embedding_vocab_matrix
            
            

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

        # preprocess the text (stopword removal, lemmatization, etc)
        self.Xtr = self._preprocess(self.X_train_raw)
        print("Xtr:", type(self.Xtr), self.Xtr.shape)
        #print("Xtr[0]:\n", self.Xtr[0])

        self.Xte = self._preprocess(self.X_test_raw)
        print("Xte:", type(self.Xte), self.Xte.shape)
        #print("Xte[0]:\n", self.Xte[0])       

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

        #self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))

        self.target_names = train_set['Category'].unique()       
        self.label_names = self.target_names           # set self.labels to the class label names        
        self.labels = self.label_names
        self.num_label_names = len(self.label_names)
        self.num_labels = self.num_label_names
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        # Encode the labels
        label_encoder = LabelEncoder()
        self.y_train = label_encoder.fit_transform(self.devel_target)
        self.y_test = label_encoder.fit_transform(self.test_target)
        
        # Convert Y to a sparse matrix
        self.y_train_sparse = csr_matrix(self.y_train).T         # Transpose to match the expected shape
        self.y_test_sparse = csr_matrix(self.y_test).T         # Transpose to match the expected shape

        return self.target_names


    def split(self):
        
        return train_test_split(
            self.X,
            self.y, 
            test_size = TEST_SIZE, 
            random_state = 60,
            shuffle=True
            #stratify=self.y
            )


    def _load_20news(self):
        
        print("\n\tloading 20newsgroups dataset...")
        
        metadata = ('headers', 'footers', 'quotes')
        
        self.devel = fetch_20newsgroups(subset='train', remove=metadata)
        self.test = fetch_20newsgroups(subset='test', remove=metadata)

        self.classification_type = 'singlelabel'
        self.class_type = 'singlelabel'
        
        #self.devel_raw, self.test_raw = mask_numbers(self.devel.data), mask_numbers(self.test.data)
        self.devel_target, self.test_target = self.devel.target, self.test.target
        
        print("devel_target:", type(self.devel_target), len(self.devel_target))
        print("test_target:", type(self.test_target), len(self.test_target))

        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))

        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)

        print("self.labels:", type(self.labels), len(self.labels))

        self.label_names = self.devel.target_names           # set self.labels to the class label names

        self.X_raw = pd.Series(self.devel.data)              # convert to Series object (from list)
        print("X_raw:", type(self.X_raw), len(self.X_raw))
        print("X_raw[0]:\n", self.X_raw[0])

        self.X = self._preprocess(self.X_raw)
        print("self.X:", type(self.X), self.X.shape)
        print("self.X[0]:\n", self.X[0])

        self.target_names = self.label_names
        print("target_names:", type(self.target_names), len(self.target_names))

        self.num_labels = len(self.labels)
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        # Encode the labels
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(self.devel_target)
        
        # Convert Y to a sparse matrix
        self.y_sparse = csr_matrix(self.y).T                   # Transpose to match the expected shape

        return self.label_names



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
        
        #self.devel_raw, self.test_raw = mask_numbers(self.devel.data), mask_numbers(self.test.data)
        
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
        
        self.X_raw = self.devel.data
        print("self.X_raw:", type(self.X_raw), len(self.X_raw))

        self.X_raw = pd.Series(self.X_raw)                      # convert to Series object (from tuple)
        print("self.X_raw:", type(self.X_raw), len(self.X_raw))
        print("self.X_raw[0]:\n", self.X_raw[0])

        self.X = self._preprocess(pd.Series(self.X_raw))
        print("self.X:", type(self.X), self.X.shape)
        print("self.X[0]:\n", self.X[0])

        self.target_names = self.label_names
        print("target_names:", type(self.target_names), len(self.target_names))

        self.num_labels = len(self.labels)
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        # Now self.devel_target is already a dense NumPy array with shape (9603, 115), so no need for MultiLabelBinarizer.
        self.y = self.devel_target
        print("y:", type(self.y), self.y.shape)

        # Convert Y to a sparse matrix
        self.y_sparse = csr_matrix(self.y)  # No need for transpose
        print("y_sparse:", type(self.y_sparse), self.y_sparse.shape)


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
        
        #self.devel_raw, self.test_raw = mask_numbers(self.devel.data), mask_numbers(self.test.data)

        self.devel_target, self.test_target = self.devel.target, self.test.target
        print("devel_target:", type(self.devel_target), len(self.devel_target))
        print("test_target:", type(self.test_target), len(self.test_target))

        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel.target, self.test.target)
        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        print("labels:\n", self.labels)

        self.label_names = self.devel.target_names                                  # set self.labels to the class label names
        print("self.label_names:\n", self.label_names)

        self.X_raw = self.devel.data
        print("self.X_raw:", type(self.X_raw), len(self.X_raw))
        print("self.X_raw[0]:\n", self.X_raw[0])

        self.X = self._preprocess(pd.Series(self.X_raw))
        print("self.X:", type(self.X), self.X.shape)
        print("self.X[0]:\n", self.X[0])

        self.target_names = self.label_names
        print("target_names:", type(self.target_names), len(self.target_names))
        
        self.num_labels = len(self.labels)
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        # Encode the labels using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        self.y = mlb.fit_transform(self.devel_target)   # Transform multi-label targets into a binary matrix
        print("y (after MultiLabelBinarizer):", type(self.y), self.y.shape)
        
        # Convert Y to a sparse matrix
        #self.y_sparse = csr_matrix(self.y).T       # Transpose to match the expected shape
        self.y_sparse = csr_matrix(self.y)          # without Transpose to match the expected shape
        print("y_sparse:", type(self.y_sparse), self.y_sparse.shape)

        return self.label_names




    def _load_rcv1(self):

        data_path = '../datasets/RCV1-v2/rcv1/'               

        print("\n\tloading rcv1 LCDataset (_load_rcv1) from path:", data_path)

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

        print("fetching training and test data...")
        devel = fetch_RCV1(subset='train', data_path=data_path, debug=False)
        test = fetch_RCV1(subset='test', data_path=data_path, debug=False)

        print("training data:", type(devel))
        print("training data:", type(devel.data), len(devel.data))
        print("training targets:", type(devel.target), len(devel.target))

        print("testing data:", type(test))
        print("testing data:", type(test.data), len(test.data))
        print("testing targets:", type(test.target), len(test.target))

        self.classification_type = 'multilabel'

        print("masking numbers...")
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(devel.target, test.target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix



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

    def _preprocess(self, text_series):
        """
        Preprocess a pandas Series of texts, tokenizing, removing punctuation, stopwords, 
        and applying stemming and lemmatization.

        Parameters:
        - text_series: A pandas Series containing text data (strings).

        Returns:
        - processed_texts: A NumPy array containing processed text strings with the shape property.
        """
        print("preprocessing text...")
        print("text_series:", type(text_series), text_series.shape)

        processed_texts = []

        for train_text in text_series:
            # Remove punctuation
            train_text = self._remove_punctuation(train_text)

            # Word tokenization
            tokenized_train_set = word_tokenize(train_text.lower())

            # Stop word removal
            stop_words = set(stopwords.words('english'))
            stopwordremove = [i for i in tokenized_train_set if i not in stop_words]

            # Join words into sentence
            stopwordremove_text = ' '.join(stopwordremove)

            # Remove numbers
            numberremove_text = ''.join(c for c in stopwordremove_text if not c.isdigit())

            # Stemming
            stemmer = PorterStemmer()
            stem_input = word_tokenize(numberremove_text)
            stem_text = ' '.join([stemmer.stem(word) for word in stem_input])

            # Lemmatization
            lemmatizer = WordNetLemmatizer()

            def get_wordnet_pos(word):
                """Map POS tag to first character lemmatize() accepts"""
                tag = nltk.pos_tag([word])[0][1][0].upper()
                tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
                return tag_dict.get(tag, wordnet.NOUN)

            lem_input = word_tokenize(stem_text)
            lem_text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in lem_input])

            processed_texts.append(lem_text)

        return np.array(processed_texts)  # Convert processed texts to NumPy array to support .shape





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

    """
    @classmethod
    def loadpt_data(cls, dataset, vtype='tfidf', pretrained=None, embedding_path=VECTOR_CACHE, emb_type='word'):

        print("LCDataset::loadpt():", dataset, PICKLE_DIR)

        #
        # load the dataset using appropriate tokenization method as dictated by pretrained embeddings
        #
        pickle_file_name=f'{dataset}_{vtype}_{pretrained}_{MAX_VOCAB_SIZE}_tokenized.pickle'

        print(f"Loading data set {dataset}...")

        pickle_file = PICKLE_DIR + pickle_file_name                                     
            
        #
        # we pick up the vectorized dataset along with the associated pretrained 
        # embedding matrices when e load the data - either from data files directly
        # if the first time parsing the dataset or from the pickled file if it exists
        # and the data has been cached for faster loading
        #
        if os.path.exists(pickle_file):                                                 # if the pickle file exists
            
            print(f"Loading tokenized data from '{pickle_file}'...")
            
            X_vectorized, y_sparse, target_names, class_type, embedding_vocab_matrix, weighted_embeddings, \
                avg_embeddings, summary_embeddings = cls.load_from_pickle(cls, pickle_file)

            return X_vectorized, y_sparse, target_names, class_type, embedding_vocab_matrix, weighted_embeddings, avg_embeddings, summary_embeddings

        else:
            print(f"'{pickle_file}' not found, loading {dataset}...")
            
            cls._initialize(
                cls,                                        # class
                vectorization_type=vtype,                   # vectorization type
                embedding_type=emb_type,                    # embedding type
                pretrained=pretrained,                      # pretrained embeddings
                pretrained_path=embedding_path              # path to embeddings
                )

            # Save the tokenized matrices to a pickle file
            cls.save_to_pickle(
                cls,                            # class
                cls.X_vectorized,               # vectorized data
                cls.y_sparse,                   # labels
                cls.target_names,               # target names
                cls.class_type,                 # class type (single-label or multi-label):
                cls.embedding_vocab_matrix,     # vector representation of the dataset vocabulary
                cls.weighted_embeddings,        # weighted avg embedding representation of dataset
                cls.avg_embeddings,             # avg embedding representation of dataset
                cls.summary_embeddings,         # summary embedding representation of dataset
                pickle_file)         
    
            return cls.X_vectorized, cls.y_sparse, cls.target_names, cls.class_type, cls.embedding_vocab_matrix, cls.weighted_embeddings, cls.avg_embeddings, cls.summary_embeddings
    """
    
    
            
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
                    print(f'Exception raised, failed to pickle data: {e}')

        else:
            print(f'loading dataset {dataset_name}')
            dataset = LCDataset(name=dataset_name, vectorization_type=vectorization_type, embedding_type=embedding_type)

        return dataset

#
# end class
#

def loadpt_data(dataset, vtype='tfidf', pretrained=None, embedding_path=VECTOR_CACHE, emb_type='word'):

    print("loadpt_data():", dataset, PICKLE_DIR)

    #
    # load the dataset using appropriate tokenization method as dictated by pretrained embeddings
    #
    pickle_file_name=f'{dataset}_{vtype}_{pretrained}_{MAX_VOCAB_SIZE}_tokenized.pickle'

    print(f"Loading data set {dataset}...")

    pickle_file = PICKLE_DIR + pickle_file_name                                     
        
    #
    # we pick up the vectorized dataset along with the associated pretrained 
    # embedding matrices when e load the data - either from data files directly
    # if the first time parsing the dataset or from the pickled file if it exists
    # and the data has been cached for faster loading
    #
    if os.path.exists(pickle_file):                                                 # if the pickle file exists
        
        print(f"Loading tokenized data from '{pickle_file}'...")
        
        Xtr_vectorized, Xte_vectorized, y_train_sparse, y_test_sparse, target_names, class_type, embedding_vocab_matrix, Xtr_weighted_embeddings, \
            Xte_weighted_embeddings, Xtr_avg_embeddings, Xte_avg_embeddings, Xtr_summary_embeddings, Xte_summary_embeddings = load_from_pickle(pickle_file)

        return Xtr_vectorized, Xte_vectorized, y_train_sparse, y_test_sparse, target_names, class_type, embedding_vocab_matrix, Xtr_weighted_embeddings, \
            Xte_weighted_embeddings, Xtr_avg_embeddings, Xte_avg_embeddings, Xtr_summary_embeddings, Xte_summary_embeddings

    else:
        print(f"'{pickle_file}' not found, loading {dataset}...")
        
        lcd = LCDataset(
            name=dataset,                               # dataset name 
            vectorization_type=vtype,                   # vectorization type (one of 'tfidf', 'count')
            embedding_type=emb_type                     # embedding type (one of 'word', 'token')
        )    

        lcd.init_embedding_matrices(
            pretrained=pretrained,                      # pretrained embeddings
            pretrained_path=embedding_path              # path to embeddings
            )

        # Save the tokenized matrices to a pickle file
        save_to_pickle(
            lcd.Xtr_vectorized,                         # vectorized training data
            lcd.Xte_vectorized,                         # vectorized test data
            lcd.y_train_sparse,                         # training data labels
            lcd.y_test_sparse,                          # test data labels
            lcd.target_names,                           # target names
            lcd.class_type,                             # class type (single-label or multi-label):
            lcd.embedding_vocab_matrix,                 # vector representation of the dataset vocabulary
            lcd.Xtr_weighted_embeddings,                # weighted avg embedding representation of dataset training data
            lcd.Xte_weighted_embeddings,                # weighted avg embedding representation of dataset test data
            lcd.Xtr_avg_embeddings,                     # avg embedding representation of dataset training data
            lcd.Xte_avg_embeddings,                     # avg embedding representation of dataset test data
            lcd.Xtr_summary_embeddings,                 # summary embedding representation of dataset training data
            lcd.Xte_summary_embeddings,                 # summary embedding representation of dataset test data
            pickle_file)         

        return lcd.Xtr_vectorized, lcd.Xte_vectorized, lcd.y_train_sparse, lcd.y_test_sparse, lcd.target_names, lcd.class_type, lcd.embedding_vocab_matrix, \
            lcd.Xtr_weighted_embeddings, lcd.Xte_weighted_embeddings, lcd.Xtr_avg_embeddings, lcd.Xte_avg_embeddings, lcd.Xtr_summary_embeddings, lcd.Xte_summary_embeddings


def save_to_pickle(Xtr, Xte, y_train, y_test, target_names, class_type, embedding_matrix, Xtr_weighted_embeddings, Xte_weighted_embeddings, Xtr_avg_embeddings, 
        Xte_avg_embeddings, Xtr_summary_embeddings, Xte_summary_embeddings, pickle_file):
    
    print(f'saving {pickle_file} to pickle...')

    #print("embedding_matrix:", type(embedding_matrix), embedding_matrix.shape)
    #print("embedding_matrix[0]:\n", embedding_matrix[0])

    # Open the file for writing and write the pickle data
    try:

        with open(pickle_file, 'wb', pickle.HIGHEST_PROTOCOL) as f:
        
            pickle.dump(
                Xtr,                                # vectorized dataset training data 
                Xte,                                # vectorized dataset test data
                y_train,                            # training dataset labels
                y_test,                             # test dataset labels
                target_names,                       # target names
                class_type,                         # class type (single-label or multi-label):
                embedding_matrix,                   # vector representation of the dataset vocabulary
                Xtr_weighted_embeddings,            # weighted avg embedding representation of dataset training data
                Xte_weighted_embeddings,            # weighted avg embedding representation of dataset test data
                Xtr_avg_embeddings,                 # avg embedding representation of dataset training data
                Xte_avg_embeddings,                 # avg embedding representation of dataset test data
                Xtr_summary_embeddings,             # summary embedding representation of dataset training data
                Xte_summary_embeddings,             # summary embedding representation of dataset test data
                f                                   # file
            )

        print("data successfully pickled at:", pickle_file)

    except Exception as e:
        print(f'Exception raised, failed to pickle data to {pickle_file}: {e}')
        
   

def load_from_pickle(pickle_file):
    
    print(f"Loading pickle file: {pickle_file}...")
    
    with open(pickle_file, 'rb') as f:
        Xtr, Xte, y_train, y_test, target_names, class_type, embedding_matrix, Xtr_weighted_embeddings, Xte_weighted_embeddings, Xtr_avg_embeddings, Xte_avg_embeddings, Xtr_summary_embeddings, Xte_summary_embeddings = pickle.load(f)

    print("embedding_matrix:", type(embedding_matrix), embedding_matrix.shape)
    #print("embedding_matrix[0]:\n", embedding_matrix[0])

    return Xtr, Xte, y_train, y_test, target_names, class_type, embedding_matrix, Xtr_weighted_embeddings, Xte_weighted_embeddings, Xtr_avg_embeddings, Xte_avg_embeddings, Xtr_summary_embeddings, Xte_summary_embeddings



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


def mask_numbers(data, number_mask='numbermask'):
    """
    Masks numbers in the given text data with a placeholder.
    """
    #print("masking numbers...")

    mask = re.compile(r'\b[0-9][0-9.,-]*\b')
    masked = []
    for text in tqdm(data, desc='masking numbers'):
        masked.append(mask.sub(number_mask, text))
    return masked


    
