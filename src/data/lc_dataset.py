import os
from os.path import join
import pickle
import numpy as np
from tqdm import tqdm
import re
import pandas as pd
import string

from sklearn.datasets import get_data_home, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder

from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1

from pathlib import Path
from urllib import request
import tarfile
import gzip
import shutil

from scipy.sparse import csr_matrix, csc_matrix

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


stop_words = set(stopwords.words('english'))

import torch
from transformers import BertTokenizer, BertTokenizerFast, LlamaTokenizer, LlamaTokenizerFast
from transformers import BertModel, LlamaModel
from gensim.models import KeyedVectors

from torch.utils.data import DataLoader, Dataset



VECTOR_CACHE = '../.vector_cache'
DATASET_DIR = '../datasets/'
PICKLE_DIR = '../pickles/'

MAX_VOCAB_SIZE = 10000                                      # max feature size for TF-IDF vectorization

BERT_MODEL = 'bert-base-uncased'                            # dimension = 768
LLAMA_MODEL = 'meta-llama/Llama-2-7b-hf'                    # dimension = 4096
#LLAMA_MODEL = 'meta-llama/Llama-2-13b-hf'

TOKEN_TOKENIZER_MAX_LENGTH = 512


# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 16
DEFAULT_GPU_BATCH_SIZE = 64
MPS_BATCH_SIZE = 16

#
# tokens for LLAMA model access, must be requested from huggingface
#
from huggingface_hub import login

HF_TOKEN = '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'
HF_TOKEN2 = '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'



# ------------------------------------------------------------------------------------------------------------------------
# Utility functions for preprocessing data
# ------------------------------------------------------------------------------------------------------------------------

def init_tfidf_vectorizer():
    """
    Initializes and returns a sklearn TFIDFVectorizer with specific configuration.
    """
    print("init_tfidf_vectorizer()")
    return TfidfVectorizer(stop_words='english', min_df=3, sublinear_tf=True)



def init_count_vectorizer():
    """
    Initializes and returns a sklearn CountVectorizer with specific configuration.
    """
    print("init_count_vectorizer()")
    return CountVectorizer(stop_words='english', min_df=3)


def missing_values(df):
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


def remove_punctuation(x):
    punctuationfree="".join([i for i in x if i not in string.punctuation])
    return punctuationfree


# Function to lemmatize text with memory optimization
def lemmatization(texts, chunk_size=1000):
    lmtzr = WordNetLemmatizer()
    
    num_chunks = len(texts) // chunk_size + 1
    #print(f"Number of chunks: {num_chunks}")
    for i in range(num_chunks):
        chunk = texts[i*chunk_size:(i+1)*chunk_size]
        texts[i*chunk_size:(i+1)*chunk_size] = [' '.join([lmtzr.lemmatize(word) for word in text.split()]) for text in chunk]
    
    return texts


# ------------------------------------------------------------------------------------------------------------------------

def preprocessDataset(train_text):
    
    #print("preprocessing...")
    
    # Ensure input is string
    train_text = str(train_text)
    
    # Word tokenization using NLTK's word_tokenize
    tokenized_train_set = word_tokenize(train_text.lower())
    
    # Stop word removal
    stop_words = set(stopwords.words('english'))
    stopwordremove = [i for i in tokenized_train_set if i not in stop_words]
    
    # Join words into sentence
    stopwordremove_text = ' '.join(stopwordremove)
    
    # Remove numbers
    numberremove_text = ''.join(c for c in stopwordremove_text if not c.isdigit())
    
    # Stemming using NLTK's PorterStemmer
    stemmer = PorterStemmer()
    stem_input = word_tokenize(numberremove_text)
    stem_text = ' '.join([stemmer.stem(word) for word in stem_input])
    
    # Lemmatization using NLTK's WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    lem_input = word_tokenize(stem_text)
    lem_text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in lem_input])
    
    return lem_text

# ------------------------------------------------------------------------------------------------------------------------





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



# ------------------------------------------------------------------------------------------------------------------------
# LCDataset class
# ------------------------------------------------------------------------------------------------------------------------
class LCDataset:
    """
    A class to handle loading and preparing datasets for text classification.
    Supports multiple datasets including Reuters, 20 Newsgroups, Ohsumed, RCV1, and WIPO.
    """

    dataset_available = {'reuters21578', '20newsgroups', 'ohsumed', 'rcv1', 'bbc-news'}
    
    def _initialize(self, name=None, vectorization_type='tfidf', embedding_type='word', pretrained=None, pretrained_path=None):
        """
        Initializes the LCDataset object by loading the appropriate dataset.
        
        Parameters:
        - name: dataset name, must be one of supprted datasets 
        - vectorizer_type: 'tfidf' or 'count', determines which vectorizer to use for tokenization.
        - embedding_type: 'word' for word-based embeddings (GloVe, Word2Vec, fastText) or 'token' for token-based models (BERT, LLaMa).
        - pretrained: 'word2vec', 'glove', 'fasttext', 'bert', or 'llama' for the pretrained embeddings to use.
        - pretrained_path: Path to the pretrained embeddings file.
        """

        print("initializing dataset with name and vectorization_type:", name, vectorization_type)
        
        assert name in LCDataset.dataset_available, f'dataset {name} is not available'

        self.name = name
        self.loaded = False
        self.vectorization_type = vectorization_type
        self.emebdding_type = embedding_type
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.embedding_type = embedding_type
        
        print("name:", self.name)
        print("vectorization_type:", self.vectorization_type)
        print("embedding_type:", self.embedding_type)
        print("pretrained:", self.pretrained)
        print("pretrained_path:", self.pretrained_path)


        # Setup device prioritizing CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print("device:", self.device)
        

        if name=='bbc-news':
            self._load_bbc_news(self)

        elif name == '20newsgroups':
            self._load_20news(self)

        elif name == 'ohsumed':
            self._load_ohsumed(self)

        elif name == 'rcv1':
            self._load_rcv1(self)

        elif name == 'reuters21578':
            self._load_reuters(self)
            
        self.nC = self.num_labels
        print("nC:", self.nC)

        if (vectorization_type=='tfidf'):
            print("initializing tfidf vectors...")
            self._vectorizer = init_tfidf_vectorizer()
        elif (vectorization_type=='count'):
            print("initializing count vectors...")
            self._vectorizer = init_count_vectorizer()
        else:
            print("WARNING: unknown vectorization_type, initializing tf-idf vectorizer...")
            self._vectorizer = init_tfidf_vectorizer()

        print("fitting training data... devel_raw type and length:", type(self.devel_raw), len(self.devel_raw))
        #self._vectorizer.fit(self.devel_raw)

        print("setting vocabulary...")
        #self.vocabulary = self._vectorizer.vocabulary_

        # vectorize dataset and build vectorizer vocabulary structures        
        self.vectorize(self)                                    
           
        # build the embedding vocabulary matrix to align 
        # with the dataset vocabulary and embedding type
        self.build_embedding_vocab_matrix(self)         
        
        # generate pretrained embedding representation of dataset 
        self.generate_dataset_embeddings(self)
        
        # Ensure X_vectorized is a sparse matrix (in case of word-based embeddings)
        if not isinstance(self.X_vectorized, csr_matrix):
            self.X_vectorized = csr_matrix(self.X_vectorized)
        
        self.loaded = True
        self.name = name


    def show(self):
        nTr_docs = len(self.devel_raw)
        nTe_docs = len(self.test_raw)
        nfeats = len(self._vectorizer.vocabulary_)
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
        # Tokenize with truncation
        return self.tokenizer.tokenize(text, max_length=TOKEN_TOKENIZER_MAX_LENGTH, truncation=True)
        
    # -----------------------------------------------------------------------------------------------------
    # vectorize()
    # 
    # Build vector representation of data set using TF-IDF or CountVectorizer and constructing 
    # the embeddings such that they align with pretrained embeddings tokenization method
    # -----------------------------------------------------------------------------------------------------
    def vectorize(self):
    
        print("building vector representation of dataset...")

        # initialize local variables
        self.model = None
        self.embedding_dim = 0
        self.embedding_vocab_matrix = None
        self.vocab = None

        # Choose the vectorization and tokenization strategy based on embedding type
        if self.embedding_type == 'word':
            print("Using word-level vectorization...")
            
            if self.vectorization_type == 'tfidf':
                self.vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE)  
            elif self.vectorization_type == 'count':
                self.vectorizer = CountVectorizer(max_features=MAX_VOCAB_SIZE)  
            else:
                raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")
            
            self.X_vectorized = self.vectorizer.fit_transform(self.X_raw)                   # Fit and transform the text data to obtain tokenized features
            
        elif self.embedding_type == 'token':
            
            print(f"Using token-level vectorization with {self.pretrained.upper()} embeddings...")

            from functools import partial

            if self.pretrained == 'bert': 
                self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT')
                self.model = BertModel.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT').to(self.device)
            elif self.pretrained == 'llama':
                self.tokenizer = LlamaTokenizerFast.from_pretrained(LLAMA_MODEL, cache_dir=VECTOR_CACHE+'/LLaMa')
                self.model = LlamaModel.from_pretrained(LLAMA_MODEL, cache_dir=VECTOR_CACHE+'/LLaMa').to(self.device)
            else:
                raise ValueError("Invalid embedding type. Use 'bert' or 'llama' for token embeddings.")
            
            print("model:\n", self.model)

            # Ensure padding token is available
            if self.tokenizer.pad_token is None:
                #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id                   # Reuse the end-of-sequence token for padding
    
            # NB: Our custom_tokenizer() method needs access to the object instance (self), so 
            # we need to use functools.partial to bind the self argument. With functools.partial, the 
            # self argument is automatically passed when calling the custom_tokenizer method, making it 
            # compatible with TfidfVectorizer.

            if self.vectorization_type == 'tfidf':
                self.vectorizer = TfidfVectorizer(
                    max_features=MAX_VOCAB_SIZE, 
                    tokenizer=partial(self.custom_tokenizer, self)
                )
            elif self.vectorization_type == 'count':
                self.vectorizer = CountVectorizer(
                    max_features=MAX_VOCAB_SIZE, 
                    tokenizer=partial(self.custom_tokenizer, self)
                )
            else:
                raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")

            print("fitting training data... devel_raw type and length:", type(self.devel_raw), len(self.devel_raw))

            # Fit and transform the text data to obtain tokenized features, 
            # note the requirement for the partial() method to bind the self 
            # argument when we call fit_transform() here 
            self.X_vectorized = self.vectorizer.fit_transform(self.X_raw)

        else:
            raise ValueError("Invalid embedding type. Use 'word' for word embeddings or 'token' for BERT/LLaMa embeddings.")

        print("X_vectorized:", type(self.X_vectorized), self.X_vectorized.shape)

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
        
        self.vocab_dict = {k.lower(): v for k, v in self.vectorizer.vocabulary_.items()}              # Ensure the vocabulary is all lowercased to avoid case mismatches
        print("vocab_dict:", type(self.vocab_dict), len(self.vocab_dict))
        """
        for i, (word, index) in enumerate(vocab_dict.items()):
            if i >= 10:
                break
            print(f"{i+1:3}: {word} -> {index}")
        """
        
        self.vocab_ndarr = self.vectorizer.get_feature_names_out()
        print("vocab_ndarr:", type(self.vocab_ndarr), len(self.vocab_ndarr))
        """
        count = 0
        for x in vocab_ndarr:
            print(f'vocab_ndarr[{count}: {x}')
            count+=1
            if (count > 10):
                break
        """
        
        self.vocab = self.vocab_dict
        print("vocab:", type(self.vocab), len(self.vocab))
        
        """    
        if not hasattr(self, 'Xtr') or not hasattr(self, 'Xte'):

            print("self does not have Xtr or Xte attributes, transforming and sorting...")

            self.Xtr = self._vectorizer.transform(self.devel_raw)
            self.Xte = self._vectorizer.transform(self.test_raw)
            self.Xtr.sort_indices()
            self.Xte.sort_indices()
        
        print("self.Xtr:", type(self.Xtr), self.Xtr.shape)
        print("self.Xte:", type(self.Xte), self.Xte.shape)
        
        return self.Xtr, self.Xte
        """

        return self.X_vectorized



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
                    embeddings = self.process_batch(self, batch_words)
                    for i, embedding in zip(word_indices, embeddings):
                        if i < len(embedding_vocab_matrix):
                            embedding_vocab_matrix[i] = embedding
                        else:
                            print(f"IndexError: Skipping index {i} as it's out of bounds for embedding_vocab_matrix.")
                    
                    batch_words = []
                    word_indices = []
                    pbar.update(batch_size)

            if batch_words:
                embeddings = self.process_batch(self, batch_words)
                for i, embedding in zip(word_indices, embeddings):
                    if i < len(embedding_vocab_matrix):
                        embedding_vocab_matrix[i] = embedding
                    else:
                        print(f"IndexError: Skipping index {i} as it's out of bounds for the embedding_vocab_matrix.")
                
                pbar.update(len(batch_words))

        return embedding_vocab_matrix


    def get_bert_embedding_cls(self, texts, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=256):
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
         
        print("getting BERT embeddings...")
        
        dataset = TextDataset(texts, self.tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        embeddings = []

        self.model.to(self.device)
        self.model.eval()

        with torch.cuda.amp.autocast(), torch.no_grad():
            for batch in tqdm(dataloader, desc="building BERT embeddings for dataset..."):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                
                # take only the embedding for the first token ([CLS]), 
                # which is expected to encode the entire document's context.
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

                embeddings.append(cls_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)

        return embeddings



    def get_bert_embeddings(self, texts, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
        """
        Compute document embeddings using a pretrained BERT model by averaging token embeddings.

        Args:
        - texts: List of input text documents.
        - batch_size: Number of samples to process in a single batch during BERT inference.
        - max_len: Maximum sequence length for tokenizing the documents.

        Returns:
        - A 2D numpy array where each row corresponds to a document's averaged embedding.
        """

        print("getting bert embeddings (without TF-IDF weights)...")

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
                            token_weight = tfidf_vector[self.vectorizer.vocabulary_[token.lower()]]
                            weighted_sum += token_embeddings[i, j].cpu().numpy() * token_weight
                            total_weight += token_weight

                    if total_weight > 0:
                        doc_embedding = weighted_sum / total_weight  # Normalize by the sum of weights
                    else:
                        doc_embedding = np.zeros(token_embeddings.size(2))  # Handle cases with no valid tokens

                    embeddings.append(doc_embedding)

        return np.vstack(embeddings)


    def get_llama_embeddings(self, texts, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
        
        print("getting llama embeddings...")
        
        dataset = TextDataset(texts, self.tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        embeddings = []

        self.model.to(self.device)
        self.model.eval()

        with torch.cuda.amp.autocast(), torch.no_grad():
            for batch in tqdm(dataloader, desc="building LLaMa embeddings for dataset..."):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token or equivalent
                embeddings.append(cls_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)

        return embeddings


    
    # -------------------------------------------------------------------------------------------------------------
    # build_embedding_vocab_matrix()
    # 
    # build the vector representation of the dataset vocabulary, ie the representation of the features that 
    # we can use to add information to the embeddings that we feed into the model (depending upon 'mode')
    # -------------------------------------------------------------------------------------------------------------

    def build_embedding_vocab_matrix(self):

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
            self.vocab_size = self.X_vectorized.shape[1]
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
        
        elif self.pretrained in ['bert', 'llama']:
            
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
            
            if self.pretrained == 'bert':
                self.vocab_size = len(self.vocab_dict)
            elif self.pretrained == 'llama':
                self.vocab_size = len(self.vocab_ndarr)
                
            self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))
            
            print("embedding_dim:", self.embedding_dim)
            print("dataset vocab size:", self.vocab_size)
            #print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)         

            self.embedding_vocab_matrix_orig = self.build_embedding_vocab_matrix_core(self, self.vocab, batch_size=MPS_BATCH_SIZE)
            print("embedding_vocab_matrix_orig:", type(self.embedding_vocab_matrix_orig), self.embedding_vocab_matrix_orig.shape)
                
            #
            # NB we use different embedding vocab matrices here depending upon the pretrained model
            #
            if (self.pretrained == 'bert'):
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
        
            
        """
        # Check if embedding_vocab_matrix is a dictionary or an ndarray and print accordingly
        if isinstance(self.embedding_vocab_matrix, dict):
            print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), len(self.embedding_vocab_matrix))
        elif isinstance(self.embedding_vocab_matrix, np.ndarray):
            print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        else:
            print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), "Unsupported type")
        """
        
        # should be a numpy array 
        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
            
            

    # ------------------------------------------------------------------------------------------------------------------------
    # load_bbc_news()
    #
    # Load BBC NEWS dataset
    # ------------------------------------------------------------------------------------------------------------------------
    def _load_bbc_news(self):
        """
        Load and preprocess the BBC News dataset and return X, Y sparse arrays along with the vocabulary.

        Returns:
        - self.target_names: list iof categories 
        """    
     
        print(f'\n\tloading BBC News dataset from {DATASET_DIR}...')

        for dirname, _, filenames in os.walk(DATASET_DIR + 'bbc-news'):
            for filename in filenames:
                print(os.path.join(dirname, filename))

        # Load datasets
        train_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Train.csv')
        test_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Test.csv')

        print("train_set:", train_set.shape)
        print("train_set columns:\n", train_set.columns)
        # print("train_set:\n", train_set.head())

        print("test_set:", test_set.shape)            
        print("test_set columns:\n", test_set.columns)
        #print("test_set:\n", test_set.head())
        
        #train_set['Category'].value_counts().plot(kind='bar', title='Category distribution in training set')
        #train_set['Category'].value_counts()
        
        print("Unique Categories:\n", train_set['Category'].unique())
        numCats = len(train_set['Category'].unique())
        print("# of categories:", numCats)

        self.X_raw = train_set['Text'].tolist()
        self.y = np.array(train_set['Category'])

        self.target_names = train_set['Category'].unique()       
        self.class_type = 'singlelabel'
        
        print("removing stopwords...")

        # Remove stopwords from the raw text
        self.X_raw = self.remove_stopwords(self, self.X_raw)
        print("X_raw:", type(self.X_raw), len(self.X_raw))

        """
        print("X_raw[0]:\n", self.X_raw[0])
        print("y[0]:", self.y[0])
        """
        
        self.classification_type = 'singlelabel'
        
        self.devel_raw = self.X_raw
        self.test_raw = test_set['Text'].tolist()
        
        #self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_target = train_set['Category']
        self.test_target = None
        
        #self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))

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
        self.y = label_encoder.fit_transform(self.devel_target)
        
        # Convert Y to a sparse matrix
        self.y_sparse = csr_matrix(self.y).T         # Transpose to match the expected shape

        return self.target_names
        


    # --------------------------------------------------------------------------------------------------------------------
    # generate_dataset_embeddings()
    #
    # generate embedding representation of dataset docs
    # --------------------------------------------------------------------------------------------------------------------
    def generate_dataset_embeddings(self):
        
        print("generating weighted average embeddings...")
        
        if (self.pretrained in ['bert', 'llama']):
            
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.batch_size = DEFAULT_GPU_BATCH_SIZE
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.batch_size = MPS_BATCH_SIZE
            else:
                self.device = torch.device("cpu")
                self.batch_size = DEFAULT_CPU_BATCH_SIZE

            # BERT embeddings        
            if (self.pretrained == 'bert'): 

                self.weighted_embeddings = self.get_weighted_bert_embeddings(
                    self,
                    texts=self.X_raw, 
                    batch_size=self.batch_size, 
                    max_len=TOKEN_TOKENIZER_MAX_LENGTH
                    )  
                
            # LLaMa embeddings
            elif (self.pretrained == 'llama'):
                

                # Project the TF-IDF vectors into the LLaMA embedding space
                def llama_weighted_average_vectorization(tfidf_vectors, vocab_embeddings, vocab):
                    
                    print("projecting tfidf vectorized data to llama embeddings...")
                        
                    print("tfidf_vectors:", type(tfidf_vectors), tfidf_vectors.shape)
                    print("vocab_embeddings:", type(vocab_embeddings), len(vocab_embeddings))
                    print("vocab:", type(vocab), vocab.shape)
                    
                    embedded_vectors = np.zeros((tfidf_vectors.shape[0], list(vocab_embeddings.values())[0].shape[1]))
                    print("embedded_vectors:", type(embedded_vectors), embedded_vectors.shape)
                    
                    # Add tqdm progress bar for iterating over documents
                    for i, doc in tqdm(enumerate(tfidf_vectors), total=tfidf_vectors.shape[0], desc="converting vectorized text into LlaMa embedding (vocabulary) space..."):
                        for j, token in enumerate(vocab):
                            if token in vocab_embeddings:
                                embedded_vectors[i] += doc[j] * vocab_embeddings[token].squeeze()
                    
                    return embedded_vectors

                # Generate the weighted average embeddings for the dataset
                self.weighted_embeddings_new = llama_weighted_average_vectorization(
                    self.X_vectorized.toarray(),
                    self.llama_vocab_embeddings,
                    self.vocab_ndarr
                    )

                print("weighted_embeddings_new:", type(self.weighted_embeddings_new), self.weighted_embeddings_new.shape)
                
                self.weighted_embeddings = self.weighted_embeddings_new
        else:

            # ---------------------------------------------------------------------------------------
            # get_word_based_weighted_embeddings()
            #
            # calculate the weighted average of the pretrained embeddings for each document. 
            # For each document, it tokenizes the text, retrieves the corresponding embeddings from (pretrained) 
            # embedding_matrix, and weights them by their TF-IDF scores.
            # 
            # This method returns a NumPy array where each row is a document's embedding.  
            # ---------------------------------------------------------------------------------------    
            def get_word_based_weighted_embeddings(text_data, vectorizer, embedding_vocab_matrix):
                
                print("get_word_based_weighted_embeddings...")
                
                document_embeddings = []
                
                for doc in text_data:
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
            
            # Word-based embeddings
            self.weighted_embeddings = get_word_based_weighted_embeddings(
                self.X_raw, 
                self.vectorizer, 
                self.embedding_vocab_matrix
                )
            
        print("weighted_embeddings:", type(self.weighted_embeddings), self.weighted_embeddings.shape)
        
        

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
        
        self.devel_raw, self.test_raw = mask_numbers(self.devel.data), mask_numbers(self.test.data)
        
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
        self.X_raw = self.remove_stopwords(self, self.X_raw)                        # Remove stopwords from the raw text
        print("self.X_raw:", type(self.X_raw), len(self.X_raw))
        
        self.target_names = self.label_names
        
        self.num_labels = len(self.labels)
        self.num_label_names = len(self.label_names)
        print("# labels, # label_names:", self.num_labels, self.num_label_names)
        if (self.num_labels != self.num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        """
        # Encode the labels using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        self.y = mlb.fit_transform(self.devel_target)  # Transform multi-label targets into a binary matrix
        """

        # Now self.devel_target is already a dense NumPy array with shape (9603, 115), so no need for MultiLabelBinarizer.
        self.y = self.devel_target
        print("y:", type(self.y), self.y.shape)

        # Convert Y to a sparse matrix
        self.y_sparse = csr_matrix(self.y)  # No need for transpose
        print("y_sparse:", type(self.y_sparse), self.y_sparse.shape)


        return self.label_names




    def _load_20news(self):
        
        print("\n\tloading 20newsgroups dataset...")
        
        metadata = ('headers', 'footers', 'quotes')
        
        self.devel = fetch_20newsgroups(subset='train', remove=metadata)
        self.test = fetch_20newsgroups(subset='test', remove=metadata)

        self.classification_type = 'singlelabel'
        
        self.class_type = 'singlelabel'
        
        self.devel_raw, self.test_raw = mask_numbers(self.devel.data), mask_numbers(self.test.data)
        self.devel_target, self.test_target = self.devel.target, self.test.target
        
        print("devel_target:", type(self.devel_target), len(self.devel_target))
        print("test_target:", type(self.test_target), len(self.test_target))

        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))

        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)

        print("self.labels:", type(self.labels), len(self.labels))

        self.label_names = self.devel.target_names           # set self.labels to the class label names

        self.X_raw = self.devel.data
        
        print("removing stopwords...")

        # Remove stopwords from the raw text
        #texts = self.X_raw
        self.X_raw = self.remove_stopwords(self, self.X_raw)
        print("self.X_raw:", type(self.X_raw), len(self.X_raw))
        
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


    def _load_ohsumed_orig(self):
        data_path = os.path.join(get_data_home(), 'ohsumed50k')
        devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        test = fetch_ohsumed50k(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(devel.target, test.target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix



    def _load_ohsumed(self):

        print("\n\tloading ohsumed dataset...")

        #data_path = os.path.join(get_data_home(), 'ohsumed50k')
        data_path = os.path.join(DATASET_DIR, 'ohsumed50k')

        print("data_path:", data_path)  

        self.devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        self.test = fetch_ohsumed50k(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.class_type = 'multilabel'
        
        self.devel_raw, self.test_raw = mask_numbers(self.devel.data), mask_numbers(self.test.data)

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
        self.X_raw = self.remove_stopwords(self, self.X_raw)                        # Remove stopwords from the raw text
        print("self.X_raw:", type(self.X_raw), len(self.X_raw))
        
        self.target_names = self.label_names
        
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
        return self._vectorizer.build_analyzer()


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
    def loadpt(cls, dataset, vtype='tfidf', pretrained=None, embedding_path=VECTOR_CACHE, emb_type='word'):

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
            
            X_vectorized, y_sparse, target_names, class_type, embedding_vocab_matrix, weighted_embeddings = load_from_pickle(pickle_file)

            return X_vectorized, y_sparse, target_names, class_type, embedding_vocab_matrix, weighted_embeddings
        else:
            print(f"'{pickle_file}' not found, loading {dataset}...")
            
            cls._initialize(
                cls,                                        # LCDataset object
                name=dataset,                               # dataset
                vectorization_type=vtype,                   # vectorization type
                embedding_type=emb_type,                    # embedding type
                pretrained=pretrained,                      # pretrained embeddings
                pretrained_path=embedding_path              # path to embeddings
                )

            # Save the tokenized matrices to a pickle file
            save_to_pickle(
                cls.X_vectorized,               # vectorized data
                cls.y_sparse,                   # labels
                cls.target_names,               # target names
                cls.class_type,                 # class type (single-label or multi-label):
                cls.embedding_vocab_matrix,     # vector representation of the dataset vocabulary
                cls.weighted_embeddings,        # weighted avg embedding representation of dataset
                pickle_file)         
    
            return cls.X_vectorized, cls.y_sparse, cls.target_names, cls.class_type, cls.embedding_vocab_matrix, cls.weighted_embeddings
    
            
            
    @classmethod
    def load(cls, dataset_name, vectorization_type='tfidf', base_pickle_path=None):

        print("Dataset::load():", dataset_name, base_pickle_path)

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
                dataset = LCDataset(name=dataset_name, vectorization_type=vectorization_type)
                print('vectorizing for faster processing')
                dataset.vectorize()
                
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
            dataset = LCDataset(name=dataset_name, vectorization_type=vectorization_type)

        return dataset


# ------------------------------------------------------------------------------------------------------------------------
# Save X, y sparse matrices, and vocabulary to pickle
# ------------------------------------------------------------------------------------------------------------------------
def save_to_pickle(X, y, target_names, class_type, embedding_matrix, weighted_embeddings, pickle_file):
    
    print(f"Saving pickle file: {pickle_file}...")
    
    print("embedding_matrix:", type(embedding_matrix), embedding_matrix.shape)
    #print("embedding_matrix[0]:\n", embedding_matrix[0])
    
    with open(pickle_file, 'wb') as f:
        # Save the sparse matrices and vocabulary as a tuple
        pickle.dump((X, y, target_names, class_type, embedding_matrix, weighted_embeddings), f)

# ------------------------------------------------------------------------------------------------------------------------
# Load X, y sparse matrices, and vocabulary from pickle
# ------------------------------------------------------------------------------------------------------------------------
def load_from_pickle(pickle_file):
    
    print(f"Loading pickle file: {pickle_file}...")
    
    with open(pickle_file, 'rb') as f:
        X, y, target_names, class_type, embedding_matrix, weighted_embeddings = pickle.load(f)

    print("embedding_matrix:", type(embedding_matrix), embedding_matrix.shape)
    #print("embedding_matrix[0]:\n", embedding_matrix[0])

    return X, y, target_names, class_type, embedding_matrix, weighted_embeddings


def _label_matrix(tr_target, te_target):
    
    print("_label_matrix...")

    mlb = MultiLabelBinarizer(sparse_output=True)
    ytr = mlb.fit_transform(tr_target)
    yte = mlb.transform(te_target)

    #print("MultiLabelBinarizer.classes_:", mlb.classes_)

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

