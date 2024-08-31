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

from scipy.sparse import csr_matrix

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import torch
from transformers import BertTokenizer, BertTokenizerFast, LlamaTokenizer, LlamaTokenizerFast
from transformers import BertModel, LlamaModel
from gensim.models import KeyedVectors

from torch.utils.data import DataLoader, Dataset
from fontTools.misc.textTools import num2binary
#from torch.testing._internal.distributed.rpc.examples.parameter_server_test import batch_size


VECTOR_CACHE = '../.vector_cache'
DATASET_DIR = '../datasets/'

MAX_VOCAB_SIZE = 50000                                      # max feature size for TF-IDF vectorization

BERT_MODEL = 'bert-base-uncased'                            # dimension = 768
LLAMA_MODEL = 'meta-llama/Llama-2-7b-hf'                    # dimension = 4096
#LLAMA_MODEL = 'meta-llama/Llama-2-13b-hf'

TOKEN_TOKENIZER_MAX_LENGTH = 512


# batch sizes for pytorch encoding routines
DEFAULT_GPU_BATCH_SIZE = 16
DEFAULT_GPU_BATCH_SIZE = 32
MPS_BATCH_SIZE = 32

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

def get_bert_embeddings(texts, model, tokenizer, device, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=256):
    
    print("getting BERT embeddings...")
    
    dataset = TextDataset(texts, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []

    model.to(device)
    model.eval()

    with torch.cuda.amp.autocast(), torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing BERT Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings

def get_weighted_bert_embeddings(texts, model, tokenizer, vectorizer, device, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
    
    print("getting weighted bert embeddings...")
    
    dataset = TextDataset(texts, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []

    model.to(device)
    model.eval()

    with torch.cuda.amp.autocast(), torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Weighted BERT Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

            for i in range(token_embeddings.size(0)):  # Iterate over each document in the batch
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                tfidf_vector = vectorizer.transform([texts[i]]).toarray()[0]

                weighted_sum = np.zeros(token_embeddings.size(2))  # Initialize weighted sum for this document
                total_weight = 0.0

                for j, token in enumerate(tokens):
                    if token in vectorizer.vocabulary_:  # Only consider tokens in the vectorizer's vocab
                        token_weight = tfidf_vector[vectorizer.vocabulary_[token.lower()]]
                        weighted_sum += token_embeddings[i, j].cpu().numpy() * token_weight
                        total_weight += token_weight

                if total_weight > 0:
                    doc_embedding = weighted_sum / total_weight  # Normalize by the sum of weights
                else:
                    doc_embedding = np.zeros(token_embeddings.size(2))  # Handle cases with no valid tokens

                embeddings.append(doc_embedding)

    return np.vstack(embeddings)

def get_llama_embeddings(texts, model, tokenizer, device, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
    
    print("getting llama embeddings...")
    
    dataset = TextDataset(texts, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []

    model.to(device)
    model.eval()

    with torch.cuda.amp.autocast(), torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing LLaMa Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token or equivalent
            embeddings.append(cls_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings

def get_weighted_llama_embeddings(texts, model, tokenizer, vectorizer, device, batch_size=DEFAULT_GPU_BATCH_SIZE, max_len=TOKEN_TOKENIZER_MAX_LENGTH):
    
    print("getting weighted llama embeddings...")
    
    dataset = TextDataset(texts, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []

    model = model.to(device)                # move model to device
    model.eval()

    with torch.cuda.amp.autocast(), torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Weighted LLaMa Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

            for i in range(token_embeddings.size(0)):  # Iterate over each document in the batch
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                tfidf_vector = vectorizer.transform([texts[i]]).toarray()[0]

                weighted_sum = np.zeros(token_embeddings.size(2))  # Initialize weighted sum for this document
                total_weight = 0.0

                for j, token in enumerate(tokens):
                    if token in vectorizer.vocabulary_:  # Only consider tokens in the vectorizer's vocab
                        token_weight = tfidf_vector[vectorizer.vocabulary_[token.lower()]]
                        weighted_sum += token_embeddings[i, j].cpu().numpy() * token_weight
                        total_weight += token_weight

                if total_weight > 0:
                    doc_embedding = weighted_sum / total_weight  # Normalize by the sum of weights
                else:
                    doc_embedding = np.zeros(token_embeddings.size(2))  # Handle cases with no valid tokens

                embeddings.append(doc_embedding)

    return np.vstack(embeddings)


# -------------------------------------------------------------------------------------------------------------------
def cache_embeddings(train_embeddings, test_embeddings, cache_path):
    print("caching embeddings to:", cache_path)
    
    np.savez(cache_path, train=train_embeddings, test=test_embeddings)

def load_cached_embeddings(cache_path, debug=False):
    if (debug):
        print("looking for cached embeddings at ", cache_path)
    
    if os.path.exists(cache_path):
        
        if (debug):
            print("found cached embeddings, loading...")
        data = np.load(cache_path)
        return data['train'], data['test']
    
    if (debug):
        print("did not find cached embeddings, returning None...")
    return None, None
# -------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------
# Function to create embeddings for a given text
# -------------------------------------------------------------------------------------------------------------------
def create_embedding(texts, model, embedding_dim):
    embeddings = np.zeros((len(texts), embedding_dim))
    for i, text in enumerate(texts):
        words = text.split()
        word_embeddings = [model[word] for word in words if word in model]
        if word_embeddings:
            embeddings[i] = np.mean(word_embeddings, axis=0)            # Average the word embeddings
        else:
            embeddings[i] = np.zeros(embedding_dim)                     # Use a zero vector if no words are in the model
    return embeddings
# -------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------
# Tokenize text using BERT tokenizer and vectorize accordingly
# -------------------------------------------------------------------------------------------------------------------
def tokenize_and_vectorize(texts, tokenizer):
    tokenized_texts = texts.apply(lambda x: tokenizer.tokenize(x))
    token_indices = tokenized_texts.apply(lambda tokens: tokenizer.convert_tokens_to_ids(tokens))
    
    # Initialize a zero matrix to store token counts (csr_matrix for efficiency)
    num_texts = len(texts)
    vocab_size = tokenizer.vocab_size
    token_matrix = csr_matrix((num_texts, vocab_size), dtype=np.float32)
    
    for i, tokens in enumerate(token_indices):
        for token in tokens:
            token_matrix[i, token] += 1  # Increment the count for each token
    
    return token_matrix
# -------------------------------------------------------------------------------------------------------------------






# ------------------------------------------------------------------------------------------------------------------------
# load_20newsgroups()
#
# Define the load_20newsgroups function
# ------------------------------------------------------------------------------------------------------------------------
def load_20newsgroups(vectorizer_type='tfidf', embedding_type='word', pretrained=None, pretrained_path=None):
    """
    Load and preprocess the 20 Newsgroups dataset, returning X and y sparse matrices.

    Parameters:
    - vectorizer_type: 'tfidf' or 'count', determines which vectorizer to use for tokenization.
    - embedding_type: 'word' for word-based embeddings (GloVe, Word2Vec, fastText) or 'token' for token-based models (LLaMA, BERT).

    Returns:
    - X: Sparse matrix of features (tokenized text aligned with the chosen vectorizer and embedding type).
    - y: Sparse array of target labels.
    - dataset_vocab: Vocabulary generated by the vectorizer (for word-level embeddings).
    - embeddings_vocab: Vocabulary generated by the tokenizer (for token-level embeddings).
    """

    print("Loading 20 newsgroups dataset...")

    # Fetch the 20 newsgroups dataset
    X_raw, y = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), return_X_y=True)

    print("preprocessing text...")

    # initualize local variables
    model = None
    embedding_dim = 0
    embedding_matrix = None
    vocab = None

    # Initialize the vectorizer based on the type
    if embedding_type == 'word':
        print("Using word-level vectorization...")
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE, stop_words='english')  # Adjust max_features as needed
        elif vectorizer_type == 'count':
            vectorizer = CountVectorizer(max_features=MAX_VOCAB_SIZE, stop_words='english')  # Adjust max_features as needed
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")
        
        # Fit and transform the text data to obtain tokenized features
        X_vectorized = vectorizer.fit_transform(X_raw)

        vocab = vectorizer.vocabulary_                                  # Set vocabulary for the dataset

    elif embedding_type == 'token':
        print("Using token-level vectorization...")
        if pretrained == 'bert':
            print("Using token-level vectorization with BERT embeddings...")
            tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)                               # BERT tokenizer
        elif pretrained == 'llama': 
            print("Using token-level vectorization with BERT or LLaMa embeddings...")
            tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL)                             # LLaMa tokenizer
        else:
            raise ValueError("Invalid embedding type. Use pretrained = 'bert' or pretrained = 'llama' for token embeddings.")

        # Ensure padding token is available
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def tokenize(text):
            tokens = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=TOKEN_TOKENIZER_MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            return tokens['input_ids']
        
        X_vectorized = [tokenize(text) for text in X_raw]
        X_vectorized = csr_matrix(np.vstack(X_vectorized))              # Convert list of arrays to a sparse matrix

        vocab = tokenizer.get_vocab()                                                             # Get vocabulary from the tokenizer

    else:
        raise ValueError("Invalid embedding type. Use 'word' for word embeddings or 'token' for BERT/LLaMA embeddings.")
    
    print("\n\tbuilding pretrained embeddings for dataset vocabulary...")

    print("pretrained:", pretrained)
    print("pretrained_path:", pretrained_path)
    print("vocab:", type(vocab), len(vocab))

    # Load the pre-trained embeddings based on the specified model
    if pretrained in ['word2vec', 'fastetxt', 'glove']:

        if (pretrained == 'word2vec'):
            print("Using Word2Vec pretrained embeddings...")
            model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
        elif pretrained == 'glove':
            print("Using GloVe pretrained embeddings...")
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove_input_file = pretrained_path
            word2vec_output_file = glove_input_file + '.word2vec'
            glove2word2vec(glove_input_file, word2vec_output_file)
            model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        elif pretrained == 'fasttext':
            print("Using fastText pretrained embeddings...")
            model = KeyedVectors.load_word2vec_format(pretrained_path)
        
        embedding_dim = model.vector_size
        print("embedding_dim:", embedding_dim)

        print("building embedding matrix for dataset...")
        # Create the embedding matrix that aligns with the TF-IDF vocabulary
        vocab_size = X_vectorized.shape[1]
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        # Extract the pretrained embeddings for words in the TF-IDF vocabulary
        for word, idx in vocab.items():
            if word in model:
                embedding_matrix[idx] = model[word]
            else:
                # If the word is not found in the pretrained model, use a random vector or zeros
                embedding_matrix[idx] = np.random.normal(size=(embedding_dim,))

        print("embedding_matrix:", type(embedding_matrix), embedding_matrix.shape)

    # Ensure X_vectorized is a sparse matrix
    if not isinstance(X_vectorized, csr_matrix):
        X_vectorized = csr_matrix(X_vectorized)
    
    label_encoder = LabelEncoder()                                                                  # Encode the labels
    y_sparse = csr_matrix(label_encoder.fit_transform(y)).T                                         # Transpose to match the expected shape

    print("Labels encoded and converted to sparse format.")

    return X_vectorized, y_sparse, embedding_matrix, vocab                                              # Return X (features) and Y (target labels) as sparse arrays                               






# ------------------------------------------------------------------------------------------------------------------------
# load_bbc_news()
# ------------------------------------------------------------------------------------------------------------------------
def load_bbc_news(vectorizer_type='tfidf', embedding_type='word', pretrained=None, pretrained_path=None):
    """
    Load and preprocess the BBC News dataset and return X, Y sparse arrays along with the vocabulary.
    
    Parameters:
    - vectorizer_type: 'tfidf' or 'count', determines which vectorizer to use for tokenization.
    - embedding_type: 'word' for word-based embeddings (GloVe, Word2Vec, fastText) or 'token' for token-based models (BERT, LLaMa).
    - pretrained: 'word2vec', 'glove', 'fasttext', 'bert', or 'llama' for the pretrained embeddings to use.
    - pretrained_path: Path to the pretrained embeddings file.

    Returns:
    - X: Sparse array of features (tokenized text aligned with the chosen vectorizer and embedding type).
    - Y: Sparse array of target labels.
    - embedding_vocab_matrix: Pretrained embeddings aligned with the dataset vocabulary.
    - weighted_embeddings: Weighted average embeddings for each document in the dataset.
    - vocab: Vocabulary generated by the vectorizer or tokenizer.
    """
    
    print("embedding_type:", embedding_type)
    print("pretrained:", pretrained)
    print("pretrained_path:", pretrained_path)

    # ----------------------------------------------------------------------
    # I: Load the dataset
    # ----------------------------------------------------------------------

    print(f'\n\tloading BBC News dataset from {DATASET_DIR}...')

    for dirname, _, filenames in os.walk(DATASET_DIR + 'bbc-news'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    # Load datasets
    train_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Train.csv')
    test_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Test.csv')

    print("train_set:", train_set.shape)
    print("test_set:", test_set.shape)    
    
    print("train_set columns:", train_set.columns)
    #print("train_set:\n", train_set.head())
    
    #train_set['Category'].value_counts().plot(kind='bar', title='Category distribution in training set')
    #train_set['Category'].value_counts()
    print("Unique Categories:\n", train_set['Category'].unique())
    numCats = len(train_set['Category'].unique())
    print("# of categories:", numCats)
    
    # ----------------------------------------------------------------------
    # II: Build vector representation of data set using TF-IDF 
    # or CountVectorizer and constructing the embeddings such that they 
    # align with pretrained embeddings tokenization method
    # ----------------------------------------------------------------------

    print("building vector representation of dataset...")

    # initialize local variables
    model = None
    embedding_dim = 0
    embedding_vocab_matrix = None
    vocab = None

    # Choose the vectorization and tokenization strategy based on embedding type
    if embedding_type == 'word':

        print("Using word-level vectorization...")
        
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE, stop_words=stopwords.words("english"))  
        elif vectorizer_type == 'count':
            vectorizer = CountVectorizer(max_features=MAX_VOCAB_SIZE, stop_words=stopwords.words("english"))  
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")
        
        # Fit and transform the text data to obtain tokenized features
        X_vectorized = vectorizer.fit_transform(train_set['Text'])
        
        # Ensure the vocabulary is all lowercased to avoid case mismatches
        vocab = {k.lower(): v for k, v in vectorizer.vocabulary_.items()}

    elif embedding_type == 'token':
        print(f"Using token-level vectorization with {pretrained.upper()} embeddings...")

        if pretrained == 'bert': 
            tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT')
            model = BertModel.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT')
        elif pretrained == 'llama':
            tokenizer = LlamaTokenizerFast.from_pretrained(LLAMA_MODEL, cache_dir=VECTOR_CACHE+'/LLaMa')
            model = LlamaModel.from_pretrained(LLAMA_MODEL, cache_dir=VECTOR_CACHE+'/LLaMa')
        else:
            raise ValueError("Invalid embedding type. Use 'bert' or 'llama' for token embeddings.")
        
        def custom_tokenizer(text):
            # Tokenize with truncation
            tokens = tokenizer.tokenize(text, max_length=TOKEN_TOKENIZER_MAX_LENGTH, truncation=True)
            return tokens

        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE, stop_words=stopwords.words("english"), tokenizer=custom_tokenizer) 
        elif vectorizer_type == 'count':
            vectorizer = CountVectorizer(max_features=MAX_VOCAB_SIZE, stop_words=stopwords.words("english"), tokenizer=custom_tokenizer)  
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")

        # Fit and transform the text data to obtain tokenized features
        X_vectorized = vectorizer.fit_transform(train_set['Text'])

        # Use the tokenizer's vocabulary directly, lowercased for consistency
        vocab = {k.lower(): v for k, v in vectorizer.vocabulary_.items()}
    
    else:
        raise ValueError("Invalid embedding type. Use 'word' for word embeddings or 'token' for BERT/LLaMa embeddings.")

    print("vectorizer vocabulary:", type(vocab), len(vocab))
    
    # Print the first 100 entries in the vocabulary
    """
    print("First 100 entries in the vectorizer vocabulary:")
    for i, (word, index) in enumerate(vocab.items()):
        if i >= 100:
            break
        print(f"{i+1:3}: {word} -> {index}")
    """
    
    # ----------------------------------------------------------------------
    # III: build the vector representation of the dataset vocabulary, 
    # ie the representation of the features that we can use to add information
    # to the embeddings that we feed into the model (depending upon 'mode')
    # ----------------------------------------------------------------------

    print("\n\tconstructing (pretrained) embeddings dataset vocabulary matrix...")    
    
    # Load the pre-trained embeddings based on the specified model
    if pretrained in ['word2vec', 'fasttext', 'glove']:

        if (pretrained == 'word2vec'):
            print("Using Word2Vec pretrained embeddings...")
            model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
        elif pretrained == 'glove':
            print("Using GloVe pretrained embeddings...")
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove_input_file = pretrained_path
            word2vec_output_file = glove_input_file + '.word2vec'
            glove2word2vec(glove_input_file, word2vec_output_file)
            model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        elif pretrained == 'fasttext':
            print("Using fastText pretrained embeddings...")
            model = KeyedVectors.load_word2vec_format(pretrained_path)
        
        embedding_dim = model.vector_size
        #print("embedding_dim:", embedding_dim)

        print("creating (pretrained) embedding matrix which aligns with dataset vocabulary...")
        
        # Create the embedding matrix that aligns with the TF-IDF vocabulary
        vocab_size = X_vectorized.shape[1]
        embedding_vocab_matrix = np.zeros((vocab_size, embedding_dim))
        
        print("embedding_dim:", embedding_dim)
        print("vocab_size:", vocab_size)
        print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)
        
        # Extract the pretrained embeddings for words in the TF-IDF vocabulary
        for word, idx in vocab.items():
            if word in model:
                embedding_vocab_matrix[idx] = model[word]
            else:
                # If the word is not found in the pretrained model, use a random vector or zeros
                embedding_vocab_matrix[idx] = np.random.normal(size=(embedding_dim,))
    
    elif pretrained in ['bert', 'llama']:
        
        # NB: tokenizer and model should be initialized in the embedding_type == 'token' block
        
        print("creating (pretrained) embedding vocab matrix which aligns with dataset vocabulary...")
        
        print("Tokenizer vocab size:", tokenizer.vocab_size)
        print("Model vocab size:", model.config.vocab_size)

        # Ensure padding token is available
        if tokenizer.pad_token is None:
            #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token_id = tokenizer.eos_token_id  # Reuse the end-of-sequence token for padding
        
        model.eval()  # Set model to evaluation mode

        embedding_dim = model.config.hidden_size  # Get the embedding dimension size
        vocab_size = len(vocab)
        embedding_vocab_matrix = np.zeros((vocab_size, embedding_dim))
        
        print("embedding_dim:", embedding_dim)
        print("dataset vocab size:", vocab_size)
        #print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)         
        
        def process_batch(batch_words):
            inputs = tokenizer(batch_words, return_tensors='pt', padding=True, truncation=True, max_length=TOKEN_TOKENIZER_MAX_LENGTH)
            input_ids = inputs['input_ids']
            max_vocab_size = model.config.vocab_size

            out_of_range_ids = input_ids[input_ids >= max_vocab_size]
            if len(out_of_range_ids) > 0:
                print("Warning: The following input IDs are out of range for the model's vocabulary:")
                for out_id in out_of_range_ids.unique():
                    token = tokenizer.decode(out_id.item())
                    print(f"Token '{token}' has ID {out_id.item()} which is out of range (vocab size: {max_vocab_size}).")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

        def build_embedding_vocab_matrix(vocab, batch_size=DEFAULT_GPU_BATCH_SIZE):
            
            embedding_vocab_matrix = np.zeros((len(vocab), model.config.hidden_size))
            batch_words = []
            word_indices = []

            with tqdm(total=len(vocab), desc="Processing Batches") as pbar:
                for word, idx in vocab.items():
                    batch_words.append(word)
                    word_indices.append(idx)

                    if len(batch_words) == batch_size:
                        embeddings = process_batch(batch_words)
                        for i, embedding in zip(word_indices, embeddings):
                            if i < len(embedding_vocab_matrix):
                                embedding_vocab_matrix[i] = embedding
                            else:
                                print(f"IndexError: Skipping index {i} as it's out of bounds for embedding_vocab_matrix.")
                        
                        batch_words = []
                        word_indices = []
                        pbar.update(batch_size)

                if batch_words:
                    embeddings = process_batch(batch_words)
                    for i, embedding in zip(word_indices, embeddings):
                        if i < len(embedding_vocab_matrix):
                            embedding_vocab_matrix[i] = embedding
                        else:
                            print(f"IndexError: Skipping index {i} as it's out of bounds for the embedding_vocab_matrix.")
                    
                    pbar.update(len(batch_words))

            return embedding_vocab_matrix

        embedding_vocab_matrix = build_embedding_vocab_matrix(vocab, batch_size=1024)
        print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)
            
    else:
        raise ValueError("Invalid pretrained type.")
        
    print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

    # ----------------------------------------------------------------------
    # IV: Generate Weighted Average Embedding Representations
    # ----------------------------------------------------------------------
    
    print("generating weighted average embeddings...")
    
    # ---------------------------------------------------------------------------------------
    # generate_weighted_embeddings()
    #
    # calculate the weighted average of the pretrained embeddings for each document. 
    # For each document, it tokenizes the text, retrieves the corresponding embeddings from 
    # embedding_matrix, and weights them by their TF-IDF scores.
    # This method returns a NumPy array where each row is a document's embedding.
    # ---------------------------------------------------------------------------------------    
    
    def get_word_based_weighted_embeddings(text_data, vectorizer, embedding_vocab_matrix):
        
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

    #print("vectorizer:", type(vectorizer), vectorizer)
    #print("tokenizer:", type(tokenizer), tokenizer)
    
    if (pretrained in ['bert', 'llama']):
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            batch_size = DEFAULT_GPU_BATCH_SIZE
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            batch_size = MPS_BATCH_SIZE
        else:
            device = torch.device("cpu")
            batch_size = DEFAULT_CPU_BATCH_SIZE

        # BERT embeddings        
        if (pretrained == 'bert'): 
            weighted_embeddings = get_weighted_bert_embeddings(
                texts=train_set['Text'].tolist(), 
                model=model, 
                tokenizer=tokenizer, 
                vectorizer=vectorizer, 
                device=device, 
                batch_size=batch_size, 
                max_len=TOKEN_TOKENIZER_MAX_LENGTH
            )  
            
        # LLaMa embeddings
        elif (pretrained == 'llama'):
        
            weighted_embeddings = get_weighted_llama_embeddings(
                texts=train_set['Text'].tolist(), 
                model=model, 
                tokenizer=tokenizer, 
                vectorizer=vectorizer, 
                device=device, 
                batch_size=batch_size, 
                max_len=TOKEN_TOKENIZER_MAX_LENGTH
            )        
            
    else:
        
        # Word-based embeddings
        weighted_embeddings = get_word_based_weighted_embeddings(
            train_set['Text'], 
            vectorizer, 
            embedding_vocab_matrix
            )
        
    print("weighted_embeddings:", type(weighted_embeddings), weighted_embeddings.shape)
    
    # Ensure X_vectorized is a sparse matrix (in case of word-based embeddings)
    if not isinstance(X_vectorized, csr_matrix):
        X_vectorized = csr_matrix(X_vectorized)
    
    # Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_set['Category'])
    
    # Convert Y to a sparse matrix
    y_sparse = csr_matrix(y).T         # Transpose to match the expected shape

    # NB we return X (features) and y (target labels) as sparse arrays
    return X_vectorized, y_sparse, embedding_vocab_matrix, weighted_embeddings, vocab         

# ------------------------------------------------------------------------------------------------------------------------




# ------------------------------------------------------------------------------------------------------------------------
# load_data()
# ------------------------------------------------------------------------------------------------------------------------
def load_data(dataset='20newsgroups', pretrained=None, embedding_path=None):

    print(f"Loading data set: {dataset}, pretrained: {pretrained}")

    if (pretrained == 'llama' or pretrained == 'bert'):
        embedding_type = 'token'
    else:
        embedding_type = 'word'


    if (dataset == '20newsgroups'): 
        
        X, y, embedding_vocab_matrix, weighted_embeddings, vocab = load_20newsgroups(
            embedding_type=embedding_type, 
            pretrained=pretrained, 
            pretrained_path=embedding_path
            )
        
        return X, y, embedding_matrix, weighted_embeddings, vocab

    elif (dataset == 'bbc-news'):
        
        X, y, embedding_vocab_matrix, weighted_embeddings, vocab = load_bbc_news(
            embedding_type=embedding_type, 
            pretrained=pretrained, 
            pretrained_path=embedding_path
            )

        return X, y, embedding_vocab_matrix, weighted_embeddings, vocab

    else:
        print(f"Dataset '{dataset}' not available, returning None.")
        return None
# ------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------
# Save X, y sparse matrices, and vocabulary to pickle
# ------------------------------------------------------------------------------------------------------------------------
def save_to_pickle(X, y, embedding_matrix, weighted_embeddings, vocab, pickle_file):
    
    print(f"Saving X, y, embedding_matrix, weighted_embeddings, and vocab to pickle file: {pickle_file}")
    
    with open(pickle_file, 'wb') as f:
        # Save the sparse matrices and vocabulary as a tuple
        pickle.dump((X, y, embedding_matrix, weighted_embeddings, vocab), f)

# ------------------------------------------------------------------------------------------------------------------------
# Load X, y sparse matrices, and vocabulary from pickle
# ------------------------------------------------------------------------------------------------------------------------
def load_from_pickle(pickle_file):
    
    print(f"Loading X, y, embedding_matrix, weighted_embeddings and vocab from pickle file: {pickle_file}")
    
    with open(pickle_file, 'rb') as f:
        X, y, embedding_matrix, weighted_embeddings, vocab = pickle.load(f)
    
    return X, y, embedding_matrix, weighted_embeddings, vocab













# ------------------------------------------------------------------------------------------------------------------------
# LCDataset class
# ------------------------------------------------------------------------------------------------------------------------
class LCDataset:
    """
    A class to handle loading and preparing datasets for text classification.
    Supports multiple datasets including Reuters, 20 Newsgroups, Ohsumed, RCV1, and WIPO.
    """

    dataset_available = {'reuters21578', '20newsgroups', 'ohsumed', 'rcv1', 'bbc-news'}
    
    def __init__(self, name=None, vectorization_type='tfidf'):
        """
        Initializes the LCDataset object by loading the appropriate dataset.
        """

        print("initializing dataset with name and vectorization_type:", name, vectorization_type)
        
        assert name in LCDataset.dataset_available, f'dataset {name} is not available'

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

        self.nC = self.devel_labelmatrix.shape[1]
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
        self._vectorizer.fit(self.devel_raw)

        print("setting vocabulary...")
        self.vocabulary = self._vectorizer.vocabulary_

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



    def _load_reuters(self):

        print("-- LCDataset::_load_reuters() --")
        data_path = os.path.join(get_data_home(), 'reuters21578')
        
        devel = fetch_reuters21578(subset='train', data_path=data_path)
        test = fetch_reuters21578(subset='test', data_path=data_path)

        #print("dev target names:", type(devel), devel.target_names)
        #print("test target names:", type(test), test.target_names)

        self.classification_type = 'multilabel'

        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(devel.target, test.target)

        self.label_names = devel.target_names           # set self.labels to the class label names

        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix

        num_labels = len(self.labels)
        num_label_names = len(self.label_names)

        print("# labels, # label_names:", num_labels, num_label_names)
        
        if (num_labels != num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        return self.label_names


    def _load_20news(self):
        metadata = ('headers', 'footers', 'quotes')
        
        devel = fetch_20newsgroups(subset='train', remove=metadata)
        test = fetch_20newsgroups(subset='test', remove=metadata)

        self.classification_type = 'singlelabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_target, self.test_target = devel.target, test.target
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))

        self.label_names = devel.target_names           # set self.labels to the class label names

        num_labels = len(self.labels)
        num_label_names = len(self.label_names)

        print("# labels, # label_names:", num_labels, num_label_names)
        
        if (num_labels != num_label_names):
            print("Warning, number of labels does not match number of label names.")
            return None

        return self.label_names


    def _load_rcv1(self):

        data_path = '../datasets/RCV1-v2/rcv1/'               

        print("loading rcv1 LCDataset (_load_rcv1) from path:", data_path)

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


    def _load_ohsumed(self):
        data_path = os.path.join(get_data_home(), 'ohsumed50k')
        devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        test = fetch_ohsumed50k(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(devel.target, test.target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix


    def _load_fasttext_data(self,name):
        data_path='../datasets/fastText'
        self.classification_type = 'singlelabel'
        name=name.replace('-','_')
        train_file = join(data_path,f'{name}.train')
        assert os.path.exists(train_file), f'file {name} not found, please place the fasttext data in {data_path}' #' or specify the path' #todo
        self.devel_raw, self.devel_target = load_fasttext_format(train_file)
        self.test_raw, self.test_target = load_fasttext_format(join(data_path, f'{name}.test'))
        self.devel_raw = mask_numbers(self.devel_raw)
        self.test_raw = mask_numbers(self.test_raw)
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1, 1), self.test_target.reshape(-1, 1))



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
        

    def vectorize(self):

        print("vectorizing dataset...")
        
        if not hasattr(self, 'Xtr') or not hasattr(self, 'Xte'):

            print("self does not have Xtr or Xte attributes, transforming and sorting...")

            self.Xtr = self._vectorizer.transform(self.devel_raw)
            self.Xte = self._vectorizer.transform(self.test_raw)
            self.Xtr.sort_indices()
            self.Xte.sort_indices()
        
        print("self.Xtr:", type(self.Xtr), self.Xtr.shape)
        print("self.Xte:", type(self.Xte), self.Xte.shape)
        
        return self.Xtr, self.Xte



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

