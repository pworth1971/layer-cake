import numpy as np

import torch, torchtext
from torchtext import vocab


from tqdm import tqdm
import os
import requests

from abc import ABC, abstractmethod

from simpletransformers.language_representation import RepresentationModel

from transformers import BertModel, RobertaModel, GPT2Model, XLNetModel, DistilBertModel, LlamaModel, PreTrainedTokenizerFast
from transformers import BertTokenizerFast, RobertaTokenizerFast, GPT2TokenizerFast, XLNetTokenizerFast, DistilBertTokenizerFast
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, PreTrainedTokenizer, PreTrainedModel

from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_vectors

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from joblib import Parallel, delayed
import re

import fasttext
from concurrent.futures import ThreadPoolExecutor

from typing import Optional, Union, Dict, Any, Tuple

import logging

# Set the logging level for gensim's FastText model to suppress specific warnings
logging.getLogger('gensim.models.fasttext').setLevel(logging.ERROR)

import gensim

# Set the logging level for gensim to suppress specific warnings
logging.getLogger('gensim.models.keyedvectors').setLevel(logging.ERROR)



# --------------------------------------------------------------------------------------------------------------------------
# tokens for LLAMA model access, must be requested from huggingface
#
from huggingface_hub import login

HF_TOKEN = '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'
HF_TOKEN2 = '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'
#
# ---------------------------------------------------------------------------------------------------------------------------

MAX_WORKER_THREADS = 10

NUM_JOBS = -1           # number of jobs for parallel processing

# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 8
DEFAULT_GPU_BATCH_SIZE = 8
DEFAULT_MPS_BATCH_SIZE = 8

VECTOR_CACHE = '../.vector_cache'                   # embedding cache directory
PICKLE_DIR = '../pickles/'                          # pickled data directory

# ---------------------------------------------------------------------------------------------------------------------------
# default pretrained language models.
# ---------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------
# Static Models
#
# NB: these models are all case sensitive, ie no need to lowercase the input text (see _preprocess)
#
#GLOVE_MODEL = 'glove.6B.300d.txt'                          # dimension 300, case insensensitve
#GLOVE_MODEL = 'glove.42B.300d.txt'                          # dimensiomn 300, case sensitive
GLOVE_MODEL = 'glove.840B.300d.txt'                          # dimensiomn 300, case sensitive

WORD2VEC_MODEL = 'GoogleNews-vectors-negative300.bin'       # dimension 300, case sensitive

#FASTTEXT_MODEL = 'crawl-300d-2M-subword.vec'                # dimension 300, case sensitive, subword based model
#FASTTEXT_MODEL = 'crawl-300d-2M-subword.bin'                # dimension 300, case sensitive, subwiord based model
FASTTEXT_MODEL = 'crawl-300d-2M.vec.bin'                    # dimensions 300, case sensitive, word based model

HYPERBOLIC_MODEL = 'best_model_dict_hyperbolic_300_cpae.vec'                # dimensions 300, word based model
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
# HuggingFace transformer models

#
# google-bert models
#
#BERT_MODEL = 'bert-base-uncased'                              # dimension = 768, case insensitive
#BERT_MODEL = 'bert-large-cased'                             # dimension = 1024, case sensitive
#BERT_MODEL = 'bert-large-uncased'                             # dimension = 1024, case insensitive
BERT_MODEL = 'bert-base-cased'                              # dimension = 768, case sensitive

#
# FacebookAI models
# NB also supports XLM and XLM-RoBERTa models as well as RoBERTa)
#
#ROBERTA_MODEL = 'roberta-large'                             # dimension = 1024, case sensitive
ROBERTA_MODEL = 'roberta-base'                             # dimension = 768, case sensitive

#
# distilbert models
#
#DISTILBERT_MODEL = 'distilbert-base-uncased'                 # dimension = 768, case insensitive
DISTILBERT_MODEL = 'distilbert-base-cased'                  # dimension 768, case sensisitve

#
# XLNet models
#
XLNET_MODEL = 'xlnet-base-cased'                                            # dimension = 768, case sensitive
#XLNET_MODEL = 'xlnet-large-cased'                           # dimension = 1024, case sensitive

#
# open-ai community huggingface models
#
GPT2_MODEL = 'gpt2'                                          # dimension = 768, case sensitive
#GPT2_MODEL = 'gpt2-medium'                                   # dimension = 1024, case sensitive
#GPT2_MODEL = 'gpt2-large'                                    # dimension = 1280, case sensitive

#
# meta-llama models
#
LLAMA_MODEL = 'meta-llama/Llama-3.2-1B'                                  # dimension = 2048, case sensitive
#LLAMA_MODEL = 'meta-llama/Llama-3.2-3B'                                  # dimension = 3072, case sensitive

#
# deepseek-ai models
# NB: limited huggingface support, requires python310 
#
#DEEPSEEK_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
#DEEPSEEK_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
#DEEPSEEK_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'             # dimensions 3584, case sensitive
DEEPSEEK_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'            # dimensions 1536, case sensitive
# ---------------------------------------------------------------------------------------------------------------------------


#
# Hugging Face Login info for gated models (eg LlaMa)
# needed for startup script which set this up
#
from huggingface_hub import login

HF_TOKEN = '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'
HF_TOKEN2 = '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'



#
# ---------------------------------------------------------------------------------------------------------------------------
MAX_VOCAB_SIZE = 20000                                      # max feature size for TF-IDF vectorization
MIN_DF_COUNT = 5                                            # min document frequency for TF-IDF vectorization
# ---------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------
# 
# Model Map for transformer based models (trans_layer_cake)
#
MODEL_MAP = {
    "glove": GLOVE_MODEL,
    "word2vec": WORD2VEC_MODEL,
    "fasttext": FASTTEXT_MODEL,
    "hyperbolic": HYPERBOLIC_MODEL,
    "bert": BERT_MODEL,
    "roberta": ROBERTA_MODEL,
    "distilbert": DISTILBERT_MODEL,
    "xlnet": XLNET_MODEL,
    "gpt2": GPT2_MODEL,
    "llama": LLAMA_MODEL,
    "deepseek": DEEPSEEK_MODEL,
}

MODEL_DIR = {
    "glove": 'GloVe',
    "word2vec": 'Word2Vec',
    "fasttext": 'fastText',
    "hyperbolic": 'hyperbolic',
    "bert": 'BERT',
    "roberta": 'RoBERTa',
    "distilbert": 'DistilBERT',
    "xlnet": 'XLNet',
    "gpt2": 'GPT2',
    "llama": 'Llama',
    "deepseek": 'DeepSeek',
}

MAX_LENGTH = 512  # default max sequence length for the transformer models
#
# ----------------------------------------------------------------------------------------------------------------------------





# Setup device prioritizing CUDA, then MPS, then CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    BATCH_SIZE = DEFAULT_GPU_BATCH_SIZE
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    BATCH_SIZE = DEFAULT_MPS_BATCH_SIZE
else:
    DEVICE = torch.device("cpu")
    BATCH_SIZE = DEFAULT_CPU_BATCH_SIZE




class LCRepresentationModel(RepresentationModel, ABC):
    """
    class LCRepresentationModel(RepresentationModel): 
    inherits from the transformer RepresentationModel class and acts as an abstract base class which
    computes representations of dataset text for different pretrained language models. This version 
    implements an encoding method that returns a summary vector for the different models as well, with 
    some performance optimizations.
    """
    
    def __init__(self, model_name=None, model_dir='../.vector_cache'):
        """
        Initialize the representation model.
        
        Args:
        - model_type (str): Type of the model ('bert', 'roberta', etc.).
        - model_name (str): Hugging Face Model Hub ID or local path.
        - model_dir (str, optional): Directory to save and load model files.
        - device (str, optional): Device to use for encoding ('cuda', 'mps', or 'cpu').
        """
        
        print("initializing LCRepresentationModel...")

        self.device = DEVICE
        self.batch_size = BATCH_SIZE
        print("self.device:", self.device)
        print("self.batch_size:", self.batch_size)

        self.model_name = model_name
        print("self.model_name:", self.model_name)
        
        self.model_dir = model_dir
        print("self.model_dir:", model_dir)

        self.combine_strategy = 'mean'          # default combine strategy  

        self.path_to_embeddings = model_dir + '/' + model_name
        print("self.path_to_embeddings:", self.path_to_embeddings)

        self.model = None
        self.tokenizer = None

        self.initialized = True
        
        self.max_length = 512


    def _download_file(self, url, destination):
        """
        Helper function to download a file from a URL.
        """
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(1024):
                file.write(data)
                bar.update(len(data))


    def _unzip_embeddings(self, zip_file_path):
        """
        Unzip embeddings.
        """
        from zipfile import ZipFile, BadZipFile

        print(f'unzipping embeddings... zip_file: {zip_file_path}')

        try:        
            with ZipFile(zip_file_path, 'r') as zip_ref:
                print(f"Extracting embeddings from {zip_file_path}...")
                zip_ref.extractall(os.path.dirname(zip_file_path))

            # Delete the zip file after extraction
            os.remove(zip_file_path)
            print(f"Deleted zip file {zip_file_path} after extraction.")

        except BadZipFile:
            print(f"Error: {zip_file_path} is not a valid ZIP file or is corrupted.")
            raise


    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer

    def show(self):
        """
        Display information about the LCRepresentationModel instance, including class type,
        model details, and vectorizer information.
        """
        print("--- LCRepresentationModel Information ---")
        print(f"Class type: {self.__class__.__name__}")
        
        # Display model details
        if self.model:
            if hasattr(self.model, 'name'):
                print(f"Model Name: {self.model.name}")
            if hasattr(self.model, 'config'):
                print(f"Model Configuration: {self.model.config}")
            if hasattr(self.model, 'dim'):
                print(f"Model Dimension: {self.model.dim}")
            else:
                print("Model Information: A model is loaded.")
        else:
            print("Model Information: No model loaded.")

        # Display vectorizer details
        if self.vectorizer:
            print(f"Vectorizer Type: {self.vectorizer.__class__.__name__}")
            if hasattr(self.vectorizer, 'min_df'):
                print(f"Minimum Document Frequency (min_df): {self.vectorizer.min_df}")
            if hasattr(self.vectorizer, 'sublinear_tf'):
                print(f"Sublinear TF Scaling: {self.vectorizer.sublinear_tf}")
            if hasattr(self.vectorizer, 'lowercase'):
                print(f"Lowercase: {self.vectorizer.lowercase}")
            if hasattr(self.vectorizer, 'vocabulary_'):
                print(f"Vocabulary Size: {len(self.vectorizer.vocabulary_)}")
        else:
            print("No vectorizer configured.")
        
        # Additional information
        print(f"Device: {self.device}")
        print(f"Batch Size: {self.batch_size}")
        print("---------------------------------------")


    @classmethod
    def reindex(cls, words, word2index):
        
        print("reindexing...")
              
        source_idx, target_idx = [], []
        oov = 0
        
        for i, word in enumerate(words):
            if word not in word2index: 
                oov += 1
                continue
            j = word2index[word]
            source_idx.append(i)
            target_idx.append(j)
        
        source_idx = np.asarray(source_idx)
        target_idx = np.asarray(target_idx)
        
        #print("oov:", oov)

        return source_idx, target_idx, oov


    @abstractmethod
    def vocabulary(cls):
        """
        Abstract method to be implemented by all subclasses.
        """
        pass


    @abstractmethod
    def extract(cls, words):
        """
        Abstract method to be implemented by all subclasses.
        """
        pass




class HyperbolicLCRepresentationModel(LCRepresentationModel):
    """
    HyperbolicLCRepresentationModel handles pretrained word embeddings trained in hyperbolic space,
    stored in .vec format.
    """
    def __init__(self, model_name='multi-relational_hyperbolic.vec', model_dir='./vector_cache/Hyperbolic', vtype='tfidf'):
        print("Initializing HyperbolicLCRepresentationModel...")

        super().__init__(model_name, model_dir=model_dir)

        # Automatically download embeddings if not present
        if not os.path.exists(self.path_to_embeddings):
            print(f"Embedding file {self.path_to_embeddings} not found. Downloading...")
            self._download_embeddings(model_name, model_dir)

        self.model = KeyedVectors.load_word2vec_format(self.path_to_embeddings, binary=False)    
    
        self.word2index = {w: i for i,w in enumerate(self.model.index_to_key)}

        self.vtype = vtype
        print(f"Vectorization type: {vtype}")

        self.type = 'hyperbolic'

        # Get embedding size (dimensionality)
        self.embedding_dim = self.model.vector_size
        print(f"self.embedding_dim: {self.embedding_dim}")

        if vtype == 'tfidf':
            print("using TF-IDF vectorization...")
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT, 
                sublinear_tf=True, 
        #        lowercase=False
        )
        elif vtype == 'count':
            print("using Count vectorization...")
            self.vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT, 
                #lowercase=False
            )
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")

    def vocabulary(self):
        return set(self.word2index.keys())

    def dim(self):
        return self.embedding_dim

    def extract(self, words):
        
        print("extracting words from Hyperbolic model...")

        source_idx, target_idx, oov = LCRepresentationModel.reindex(words, self.word2index)

        print("OOV:", oov)
        
        extraction = np.zeros((len(words), self.dim()))
        extraction[source_idx] = self.model.vectors[target_idx]
        extraction = torch.from_numpy(extraction).float()

        return extraction
    

    def build_embedding_vocab_matrix(self):
    
        print('Building embedding vocab matrix for hyperbolic embeddings...')
    
        vocabulary = np.asarray(list(zip(*sorted(self.vectorizer.vocabulary_.items(), key=lambda x: x[1])))[0])
        self.embedding_vocab_matrix = self.extract(vocabulary).numpy()
        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
    
        return self.embedding_vocab_matrix


    def encode_docs(self, texts, embedding_vocab_matrix):

        """
        Compute both weighted document embeddings (using TF-IDF) and average document embeddings for each document.

        Args:
        - texts: List of input documents (as raw text).
        - embedding_vocab_matrix: Matrix of pre-trained word embeddings (e.g., Word2Vec, GloVe).

        Returns:
        - weighted_document_embeddings: Numpy array of weighted document embeddings for each document.
        - avg_document_embeddings: Numpy array of average document embeddings for each document.
        """

        print(f"\n\tEncoding docs using Hyperbolic embeddings...")

        print("texts:", type(texts), len(texts))
        print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

        weighted_document_embeddings = []
        avg_document_embeddings = []

        # Compute the mean embedding for the entire embedding matrix as a fallback for OOV tokens
        self.mean_embedding = np.mean(embedding_vocab_matrix, axis=0)
        #print(f"Mean embedding vector (for OOV tokens): {self.mean_embedding.shape}")
        #print("mean_embedding:", type(self.mean_embedding), self.mean_embedding)

        oov_tokens = 0

        for doc in texts:
            # Tokenize the document using the vectorizer (ensures consistency in tokenization)
            tokens = self.vectorizer.build_analyzer()(doc)

            # Calculate TF-IDF weights for the tokens
            tfidf_vector = self.vectorizer.transform([doc]).toarray()[0]

            weighted_sum = np.zeros(embedding_vocab_matrix.shape[1])
            total_weight = 0.0
            valid_embeddings = []    

            for token in tokens:
                # Get the token's index in the vocabulary (case-sensitive lookup first)
                token_id = self.vectorizer.vocabulary_.get(token, None)

                # If the token is not found, try the lowercase version
                if token_id is None:
                    token_id = self.vectorizer.vocabulary_.get(token.lower(), None)

                if token_id is not None and 0 <= token_id < embedding_vocab_matrix.shape[0]:
                    # Get the embedding for the token from the embedding matrix
                    embedding = embedding_vocab_matrix[token_id]
                    # Get the TF-IDF weight for the token
                    weight = tfidf_vector[token_id]

                    # Accumulate the weighted embedding
                    weighted_sum += embedding * weight
                    total_weight += weight
                    valid_embeddings.append(embedding)
                else:
                    # Use the mean embedding for OOV tokens
                    embedding = self.mean_embedding
                    oov_tokens += 1
                    valid_embeddings.append(embedding)

            # Compute the weighted embedding for the document
            if total_weight > 0:
                weighted_document_embedding = weighted_sum / total_weight
            else:
                # Handle empty or OOV cases
                weighted_document_embedding = self.mean_embedding

            # Compute the average embedding for the document
            if valid_embeddings:
                avg_document_embedding = np.mean(valid_embeddings, axis=0)
            else:
                # Handle empty or OOV cases
                avg_document_embedding = self.mean_embedding

            weighted_document_embeddings.append(weighted_document_embedding)
            avg_document_embeddings.append(avg_document_embedding)

        weighted_document_embeddings = np.array(weighted_document_embeddings)
        avg_document_embeddings = np.array(avg_document_embeddings)

        print("weighted_document_embeddings:", type(weighted_document_embeddings), weighted_document_embeddings.shape)
        print("avg_document_embeddings:", type(avg_document_embeddings), avg_document_embeddings.shape)
        print("oov_tokens:", oov_tokens)

        return weighted_document_embeddings, avg_document_embeddings




class GloVeLCRepresentationModel(LCRepresentationModel):
    """
    GloVeLCRepresentationModel handles GloVe, word-based, embeddings.
    
    It computes sentence embeddings by averaging, summing, or computing TF-IDF weighted embeddings.
    """

    def __init__(self, model_name, model_dir, vtype='tfidf'):
        """
        Initialize the GloVe, word-based, representation model 

        Parameters:
        ----------
        model_name : str
            Name of the pre-trained word embedding model (e.g., 'word2vec', 'glove').
        embedding_path : str
            Path to the pre-trained embedding file (e.g., 'GoogleNews-vectors-negative300.bin').
        """
        
        print("Initializing GloveLCRepresentationModel...")

        super().__init__(model_name, model_dir=model_dir)  # parent constructor

        # Automatically download embeddings if not present
        if not os.path.exists(self.path_to_embeddings):
            print(f"Embedding file {self.path_to_embeddings} not found. Downloading...")
            self._download_embeddings(model_name, model_dir)

        if (model_name == 'glove.6B.300d.txt'):
            self.model = torchtext.vocab.GloVe(name='6B', cache=model_dir)
        elif (model_name == 'glove.42B.300d.txt'):
            self.model = torchtext.vocab.GloVe(name='42B', cache=model_dir)
        elif (model_name == 'glove.840B.300d.txt'):
            self.model = torchtext.vocab.GloVe(name='840B', cache=model_dir)
        else:
            raise ValueError(f"Unsupported GloVe model {model_name}.")
        
        print("self.model:\n", self.model)
        
        self.vtype = vtype
        print(f"vectorization type: {vtype}")

        self.type = 'glove'

        # Get embedding size (dimensionality)
        self.embedding_dim = self.model.dim
        print(f"self.embedding_dim: {self.embedding_dim}")
        
        #
        # vectorize the text, note that the Word2Vec and GloVe models we use are case sensitive
        #
        if vtype == 'tfidf':
            print("using TF-IDF vectorization...")
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                sublinear_tf=True,                              # use sublinear TF scaling
#                lowercase=False                                 # dont lowercase the tokens
            )              
        elif vtype == 'count':
            print("using Count vectorization...")
            self.vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
#                lowercase=False                                 # dont lowercase the tokens
            )
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")

    
    def _download_embeddings(self, model_name, model_dir):
        """
        Download pre-trained GloVe embeddings from a URL and save them to the specified path.
        """

        print(f'downloading embeddings... model:{model_name}, path:{model_dir}')

        # GloVe embeddings (Stanford)
        if (GLOVE_MODEL == 'glove.6B.300d.txt'):
            url = f"https://nlp.stanford.edu/data/glove.6B.zip"
            zipfile = 'glove.6B.zip'
        elif (GLOVE_MODEL == 'glove.42B.300d.txt'):    
            url = f"https://nlp.stanford.edu/data/glove.42B.300d.zip"
            zipfile = 'glove.42B.zip'
        elif (GLOVE_MODEL == 'glove.840B.300d.txt'):
            url = f"https://nlp.stanford.edu/data/glove.840B.300d.zip"
            zipfile = 'glove.840B.zip'
        else:
            raise ValueError(f"Unsupported model {GLOVE_MODEL} for download.")

        dest_zip_file = model_dir + '/' + zipfile
        print(f"Downloading embeddings from {url} to {dest_zip_file}...")
        self._download_file(url, dest_zip_file)

        # Unzip GloVe embeddings
        self._unzip_embeddings(dest_zip_file)


    def vocabulary(self):
        # Accessing the vocabulary from the torchtext GloVe model
        return set(self.model.stoi.keys())


    def dim(self):
        # getting the dimension of the embeddings
        return self.model.dim


    def extract(self, words):

        print("extracting words from GloVe model...")
        print("words:", type(words), len(words))

        source_idx, target_idx, oov = LCRepresentationModel.reindex(words, self.model.stoi)
        
        print("OOV:", oov)
        
        extraction = torch.zeros((len(words), self.model.dim))
        extraction[source_idx] = self.model.vectors[target_idx]
        
        return extraction
    

    def build_embedding_vocab_matrix(self):

        print('building embedding vocab matrix...')

        vocabulary = np.asarray(list(zip(*sorted(self.vectorizer.vocabulary_.items(), key=lambda x: x[1])))[0])

        self.embedding_vocab_matrix = self.extract(vocabulary).numpy()

        #print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)

        return self.embedding_vocab_matrix


    def encode_docs(self, texts, embedding_vocab_matrix):
        """
        Compute both weighted document embeddings (using TF-IDF) and average document embeddings for each document.

        Args:
        - texts: List of input documents (as raw text).
        - embedding_vocab_matrix: Matrix of pre-trained word embeddings (e.g., Word2Vec, GloVe).

        Returns:
        - weighted_document_embeddings: Numpy array of weighted document embeddings for each document.
        - avg_document_embeddings: Numpy array of average document embeddings for each document.
        """
        
        print(f"\n\tencoding docs using GloVe embeddings...")

        print("texts:", type(texts), len(texts))
        print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

        # Initialize as lists, not numpy arrays
        weighted_document_embeddings = []
        avg_document_embeddings = []
        
        # Calculate the mean embedding from the embedding vocab matrix
        mean_embedding = np.mean(embedding_vocab_matrix, axis=0)
        print(f"Mean embedding vector calculated: {mean_embedding.shape}")

        oov_tokens = 0

        for doc in texts:
            # Tokenize the document using the vectorizer (ensures consistency in tokenization)
            tokens = self.vectorizer.build_analyzer()(doc)

            # Calculate TF-IDF weights for the tokens
            tfidf_vector = self.vectorizer.transform([doc]).toarray()[0]

            weighted_sum = np.zeros(embedding_vocab_matrix.shape[1])
            total_weight = 0.0
            valid_embeddings = []    

            for token in tokens:
                # Get the embedding from GloVe
                if token in self.model.stoi:
                    embedding = self.model.vectors[self.model.stoi[token]].numpy()
                elif token.lower() in self.model.stoi:
                    # Check lowercase version if not found
                    embedding = self.model.vectors[self.model.stoi[token.lower()]].numpy()
                else:
                    # Handle OOV tokens by using the mean embedding from embedding_vocab_matrix
                    embedding = mean_embedding
                    oov_tokens += 1

                # Get the token's TF-IDF weight
                token_id = self.vectorizer.vocabulary_.get(token, None)
                weight = tfidf_vector[token_id] if token_id is not None else 1.0  # Default weight to 1.0 if not found

                # Accumulate the weighted embedding
                weighted_sum += embedding * weight
                total_weight += weight
                valid_embeddings.append(embedding)

            # Compute the weighted embedding for the document
            if total_weight > 0:
                weighted_doc_embedding = weighted_sum / total_weight
            else:
                weighted_doc_embedding = mean_embedding  # Use mean embedding for empty or OOV cases

            # Compute the average embedding for the document
            if valid_embeddings:
                avg_document_embedding = np.mean(valid_embeddings, axis=0)
            else:
                avg_document_embedding = mean_embedding  # Use mean embedding for empty or OOV cases
            
            weighted_document_embeddings.append(weighted_doc_embedding)
            avg_document_embeddings.append(avg_document_embedding)

        weighted_document_embeddings = np.array(weighted_document_embeddings)
        avg_document_embeddings = np.array(avg_document_embeddings)

        print("weighted_document_embeddings:", type(weighted_document_embeddings), weighted_document_embeddings.shape)
        print("avg_document_embeddings:", type(avg_document_embeddings), avg_document_embeddings.shape)

        print("oov_tokens:", oov_tokens)

        return weighted_document_embeddings, avg_document_embeddings




class Word2VecLCRepresentationModel(LCRepresentationModel):
    """
    Word2VecLCRepresentationModel handles word-based embeddings.
    It computes sentence embeddings by averaging, summing, or computing TF-IDF weighted embeddings.
    """

    def __init__(self, model_name, model_dir, vtype='tfidf'):
        """
        Initialize the word-based representation model (e.g., Word2Vec, GloVe).
        
        Parameters:
        ----------
        model_name : str
            Name of the pre-trained word embedding model (e.g., 'word2vec', 'glove').
        embedding_path : str
            Path to the pre-trained embedding file (e.g., 'GoogleNews-vectors-negative300.bin').
        """
        print("Initializing Word2VecLCRepresentationModel...")

        super().__init__(model_name, model_dir=model_dir)                   # parent constructor

        # Automatically download embeddings if not present
        if not os.path.exists(self.path_to_embeddings):
            print(f"Embedding file {self.path_to_embeddings} not found. Downloading...")
            self._download_embeddings(model_name, model_dir)

        self.model = KeyedVectors.load_word2vec_format(self.path_to_embeddings, binary=True)    
        
        self.word2index = {w: i for i,w in enumerate(self.model.index_to_key)}

        self.vtype = vtype
        print(f"Vectorization type: {vtype}")

        self.type = 'word2vec'

        # Get embedding size (dimensionality)
        self.embedding_dim = self.model.vector_size
        print(f"self.embedding_dim: {self.embedding_dim}")
        
        #
        # vectorize the text, note that the Word2Vec and GloVe models we use are case sensitive
        #
        if vtype == 'tfidf':
            print("using TF-IDF vectorization...")
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                sublinear_tf=True,                              # use sublinear TF scaling
                lowercase=False                                 # dont lowercase the tokens
            )              
        elif vtype == 'count':
            print("using Count vectorization...")
            self.vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                lowercase=False                                 # dont lowercase the tokens
            )
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")

    
    def _download_embeddings(self, model_name, model_dir):
        """
        Download pre-trained embeddings (Word2Vec or GloVe) from a URL and save them to the specified path.
        """

        print(f'downloading embeddings... model:{model_name}, path:{model_dir}')

        #
        # TODO: This URL is not correct, pls dowwnload these embeddings offline 
        # from kaggle here: https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300
        #
        
        # Word2Vec Google News embeddings (Commonly hosted on Google Drive)
        url = "https://drive.usercontent.google.com/download?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download&authuser=1"
    
        # Download the file
        print(f"Downloading embeddings from {url} to {self.path_to_embeddings}...")
    
        self._download_file(url, self.path_to_embeddings)


    def vocabulary(self):
        return set(self.word2index.keys())

    def dim(self):
        return self.model.vector_size

    def extract(self, words):
        
        print("extracting words from Word2Vec model...")

        source_idx, target_idx, oov = LCRepresentationModel.reindex(words, self.word2index)

        print("OOV:", oov)
        
        extraction = np.zeros((len(words), self.dim()))
        extraction[source_idx] = self.model.vectors[target_idx]
        extraction = torch.from_numpy(extraction).float()

        return extraction
    

    def build_embedding_vocab_matrix(self):

        print('building embedding vocab matrix...')

        vocabulary = np.asarray(list(zip(*sorted(self.vectorizer.vocabulary_.items(), key=lambda x: x[1])))[0])

        self.embedding_vocab_matrix = self.extract(vocabulary).numpy()

        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)

        return self.embedding_vocab_matrix
    

    def encode_docs(self, texts, embedding_vocab_matrix):
        """
        Compute both weighted document embeddings (using TF-IDF) and average document embeddings for each document.

        Args:
        - texts: List of input documents (as raw text).
        - embedding_vocab_matrix: Matrix of pre-trained word embeddings (e.g., Word2Vec, GloVe).

        Returns:
        - weighted_document_embeddings: Numpy array of weighted document embeddings for each document.
        - avg_document_embeddings: Numpy array of average document embeddings for each document.
        """

        print(f"\n\tencoding docs using Word2Vec embeddings...")

        print("texts:", type(texts), len(texts))
        print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

        weighted_document_embeddings = []
        avg_document_embeddings = []

        # Compute the mean embedding for the entire embedding matrix as a fallback for OOV tokens
        self.mean_embedding = np.mean(embedding_vocab_matrix, axis=0)
        #print(f"Mean embedding vector (f)or OOV tokens): {self.mean_embedding.shape}")
        #print("mean_embedding:", type(self.mean_embedding), self.mean_embedding)

        oov_tokens = 0

        for doc in texts:
            # Tokenize the document using the vectorizer (ensures consistency in tokenization)
            tokens = self.vectorizer.build_analyzer()(doc)

            # Calculate TF-IDF weights for the tokens
            tfidf_vector = self.vectorizer.transform([doc]).toarray()[0]

            weighted_sum = np.zeros(embedding_vocab_matrix.shape[1])
            total_weight = 0.0
            valid_embeddings = []    

            for token in tokens:
                # Get the token's index in the vocabulary (case-sensitive lookup first)
                token_id = self.vectorizer.vocabulary_.get(token, None)

                # If the token is not found, try the lowercase version
                if token_id is None:
                    token_id = self.vectorizer.vocabulary_.get(token.lower(), None)

                if token_id is not None and 0 <= token_id < embedding_vocab_matrix.shape[0]:
                    # Get the embedding for the token from the embedding matrix
                    embedding = embedding_vocab_matrix[token_id]
                    # Get the TF-IDF weight for the token
                    weight = tfidf_vector[token_id]

                    # Accumulate the weighted embedding
                    weighted_sum += embedding * weight
                    total_weight += weight
                    valid_embeddings.append(embedding)
                else:
                    # Use the mean embedding for OOV tokens
                    embedding = self.mean_embedding
                    oov_tokens += 1
                    valid_embeddings.append(embedding)

            # Compute the weighted embedding for the document
            if total_weight > 0:
                weighted_document_embedding = weighted_sum / total_weight
            else:
                # Handle empty or OOV cases
                weighted_document_embedding = self.mean_embedding

            # Compute the average embedding for the document
            if valid_embeddings:
                avg_document_embedding = np.mean(valid_embeddings, axis=0)
            else:
                # Handle empty or OOV cases
                avg_document_embedding = self.mean_embedding

            weighted_document_embeddings.append(weighted_document_embedding)
            avg_document_embeddings.append(avg_document_embedding)

        weighted_document_embeddings = np.array(weighted_document_embeddings)
        avg_document_embeddings = np.array(avg_document_embeddings)

        print("weighted_document_embeddings:", type(weighted_document_embeddings), weighted_document_embeddings.shape)
        print("avg_document_embeddings:", type(avg_document_embeddings), avg_document_embeddings.shape)
        print("oov_tokens:", oov_tokens)

        return weighted_document_embeddings, avg_document_embeddings

class FastTextGensimLCRepresentationModel(LCRepresentationModel):
    """
    FastTextLCRepresentationModel handles fastText (subword) based embeddings.
    It computes sentence embeddings by averaging, summing, or computing TF-IDF weighted embeddings.
    """

    def __init__(self, model_name=FASTTEXT_MODEL, model_dir=VECTOR_CACHE+'/fastText', vtype='tfidf'):
        """
        Initialize the word-based representation model (e.g., Word2Vec, GloVe).
        
        Parameters:
        ----------
        model_name : str
            Name of the pre-trained word embedding model (e.g. 'crawl-300d-2M-subword.bin').
        model_dir : str
            Path to the pre-trained embedding file (e.g., '../.vector_cache/fastText').
        vtype : str, optional
            vectorization type, either 'tfidf' or 'count'.
        """

        print("Initializing FastTextGensimLCRepresentationModel...")

        super().__init__(model_name, model_dir=model_dir)  # parent constructor

        # Automatically download embeddings if not present
        if not os.path.exists(self.path_to_embeddings):

            """
            print(f"Error: The FastText model file was not found at '{self.path_to_embeddings}'.")
            print("Please download the model from the official FastText repository:")
            print("https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip")
            print("Extract the zip file and ensure the .bin file is located at the specified path.")
            raise FileNotFoundError(f"Embedding file {self.path_to_embeddings} not found.")
            """
            vec_file, vec_path = self._download_embeddings(model_name, model_dir)

            # load .vec file 
            print("loading fastText embeddings from .vec file using Gensim (non-binary format)...")
            self.model = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=False)

            # extracts a .vec file if successful, must be stored as binary format for faster processing
            self.save_binary(vec_path+'.bin')

        print("loading fastText embeddings (binary format) from {}".format(self.path_to_embeddings))
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.path_to_embeddings, binary=True)
        
        # Load the FastText model using Gensim's load_facebook_vectors
        #self.model = load_facebook_model(self.path_to_embeddings)

        self.mean_embedding = np.mean(self.model.vectors, axis=0)  # Compute the mean vector

        self.vtype = vtype
        print(f"Vectorization type: {vtype}")

        self.type = 'fasttext'

        # Get embedding size (dimensionality) using get_dimension()
        self.embedding_dim = self.model.vector_size
        print(f"Embedding dimension: {self.embedding_dim}")
        
        #
        # vectorize the text, note that the Word2Vec and GloVe models we use are case sensitive
        #
        if vtype == 'tfidf':
            print("using TF-IDF vectorization...")
            
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                sublinear_tf=True,                              # use sublinear TF scaling
                lowercase=False                                 # dont lowercase the tokens
            )              
        elif vtype == 'count':
            print("using Count vectorization...")

            self.vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                lowercase=False                                 # dont lowercase the tokens
            )
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")


    def save_binary(self, path):

        print("saving fastText embeddings to {}".format(path))

        self.model.save_word2vec_format(path, binary=True)


    def _download_embeddings(self, model_name, model_dir):
        """
        Download pre-trained fastText embeddings from a URL, unzip, and locate the .vec file.

        Returns:
        - file_name: The name of the .vec file.
        - full_file_path: The complete path to the .vec file.
        """

        print(f'downloading embeddings... model: {model_name}, path: {model_dir}')

        # fastText Embeddings 
        if (FASTTEXT_MODEL == 'crawl-300d-2M.vec.bin'):
            url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
            zipfile = 'crawl-300d-2M.vec.zip'
        else:
            raise ValueError(f"Unsupported model {FASTTEXT_MODEL} for download.")

        dest_zip_file = model_dir + '/' + zipfile
        print(f"Downloading embeddings from {url} to {dest_zip_file}...")
        self._download_file(url, dest_zip_file)

        # Unzip embeddings
        print(f"Unzipping embeddings from {dest_zip_file}...")
        self._unzip_embeddings(dest_zip_file)

        # Look for the .vec file
        print("Looking for .vec file after extraction...")
        vec_file = None
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.vec'):
                    vec_file = file
                    full_file_path = os.path.join(root, file)
                    break

        # Raise an error if no .vec file is found
        if not vec_file:
            raise FileNotFoundError("No .vec file was found in the extracted contents.")

        print(f"Found .vec file: {vec_file} at {full_file_path}")

        return vec_file, full_file_path


    def vocabulary(self):
        return set(self.model.key_to_index.keys())

    def dim(self):
        return self.model.vector_size

    
    def extract(self, words):
        print("Extracting words from fastText model...")
        oov = 0
        extraction = np.zeros((len(words), self.dim()))

        for idx, word in enumerate(words):
            if word in self.model:
                extraction[idx] = self.model[word]
            else:
                extraction[idx] = self.model.get_vector(word, norm=True)
                oov += 1

        print(f"OOV count: {oov}")
        return torch.from_numpy(extraction).float()


    def extract_old(self, words):
        
        print("extracting words from fastText model...")

        oov = 0

        extraction = np.zeros((len(words), self.dim()))

        for idx, word in enumerate(words):
            if word in self.model.wv:
                extraction[idx] = self.model.wv[word]
            else:
                # Use subword information for OOV words
                extraction[idx] = self.model.get_vector(word, norm=True)
                oov += 1

        print("OOV:", oov)
        
        extraction = torch.from_numpy(extraction).float()
        
        return extraction
    

    def build_embedding_vocab_matrix(self):

        print("Building embedding vocab matrix using Gensim FastText model...")

        vocabulary = np.asarray(list(self.vectorizer.vocabulary_.keys()))
        embedding_matrix = np.zeros((len(vocabulary), self.embedding_dim))

        oov = 0

        for idx, word in enumerate(tqdm(vocabulary, desc="Embedding vocab matrix...")):
            try:
                # Check if the word exists directly in the vocabulary
                if word in self.model:
                    embedding_matrix[idx] = self.model[word]
                else:
                    # Handle OOV words using subword information
                    embedding_matrix[idx] = self.model.get_vector(word, norm=True)
                    oov += 1
            except KeyError:
                print(f"Subword vector not found for '{word}', falling back to mean embedding.")
                embedding_matrix[idx] = self.mean_embedding
                oov += 1

        print(f"OOV words: {oov} ({(oov / len(vocabulary)) * 100:.2f}%)")
        return embedding_matrix


    def encode_docs(self, texts):
        """
        Compute document embeddings using Gensim FastText API by averaging word and subword vectors.
        
        Args:
        - texts: List of input documents (as raw text).

        Returns:
        - avg_document_embeddings: Numpy array of average document embeddings for each document.
        """
        
        print("\n\tEncoding docs using Gensim FastText embeddings...")

        avg_document_embeddings = []
        oov = 0

        # Use tqdm to show progress while encoding documents
        for doc in tqdm(texts, desc="Encoding documents with Gensim FastText..."):
            tokens = self.vectorizer.build_analyzer()(doc)
            token_vectors = []

            for token in tokens:
                try:
                    # Check if the token exists directly in the FastText vocabulary
                    if token in self.model:
                        token_vectors.append(self.model[token])
                    else:
                        # Handle OOV words using subword information
                        token_vectors.append(self.model.get_vector(token, norm=True))
                        oov += 1
                except KeyError:
                    # If subword retrieval fails, fall back to mean embedding
                    print(f"Warning: '{token}' not found in vocabulary or subwords, using mean embedding.")
                    token_vectors.append(self.mean_embedding)
                    oov += 1

            if token_vectors:
                avg_document_embedding = np.mean(token_vectors, axis=0)
            else:
                avg_document_embedding = self.mean_embedding  # Handle empty documents by using the mean embedding

            avg_document_embeddings.append(avg_document_embedding)

        avg_document_embeddings = np.array(avg_document_embeddings)
        print("Average document embeddings shape:", avg_document_embeddings.shape)
        print(f"Total OOV words: {oov}")

        return avg_document_embeddings, avg_document_embeddings




    def build_embedding_vocab_matrix_old(self):

        print("building embedding vocab matrix for Gensim FastText model...")

        vocabulary = np.asarray(list(self.vectorizer.vocabulary_.keys()))
        print("vocabulary:", type(vocabulary), vocabulary.shape)

        # Preallocate embedding matrix with zeros for efficiency
        embedding_matrix = np.zeros((len(vocabulary), self.embedding_dim))

        oov = 0

        for idx, word in enumerate(tqdm(vocabulary, desc="building embedding vocab matrix using Gensim fastText...")):
            if word in self.model.wv:  # Access vectors through self.model.wv
                embedding = self.model.wv[word]
            else:
                oov += 1
                # Word is OOV, use subword information
                embedding = self.model.get_vector(word, norm=True)

            embedding_matrix[idx] = embedding

        self.embedding_vocab_matrix = embedding_matrix
        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        print(f"OOV words: {oov} ({(oov / len(vocabulary)) * 100:.2f}%)")

        return self.embedding_vocab_matrix


    def encode_docs_old(self, texts):
        """
        Compute document embeddings using Gensim FastText API by averaging word and subword vectors.
        
        Args:
        - texts: List of input documents (as raw text).

        Returns:
        - avg_document_embeddings: Numpy array of average document embeddings for each document.
        """
        
        print("\n\tencoding docs using Gensim FastText embeddings...")

        avg_document_embeddings = []

        oov = 0
        # Use tqdm to show progress while encoding documents
        for doc in tqdm(texts, desc="encoding docs with Gensim fastText..."):
            tokens = self.vectorizer.build_analyzer()(doc)
            token_vectors = []

            for token in tokens:
                if token in self.model.wv:  # Access word vectors through self.model.wv
                    # Use the word vector directly if available
                    token_vectors.append(self.model.wv[token])
                else:
                    # Otherwise, use the subword vector
                    token_vectors.append(self.model.get_vector(token, norm=True))
                    oov += 1

            if token_vectors:
                avg_document_embedding = np.mean(token_vectors, axis=0)
            else:
                avg_document_embedding = self.mean_embedding

            avg_document_embeddings.append(avg_document_embedding)

        avg_document_embeddings = np.array(avg_document_embeddings)
        print("avg_document_embeddings:", type(avg_document_embeddings), avg_document_embeddings.shape)
        print("OOV words:", oov)

        return avg_document_embeddings, avg_document_embeddings



# -----------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------


class TransformerLCRepresentationModel(LCRepresentationModel):
    """
    Base class for Transformer based architcteure langugae models such as BERT, RoBERTa, XLNet and GPT2
    """
    
    def __init__(self, model_name, model_dir, vtype='tfidf'):

        super().__init__(model_name, model_dir)
    
        self.device = DEVICE

        # Set dtype for half-precision or full precision
        self.torch_dtype = torch.float16                    # Mixed precision for memory efficiency

        # Load tokenizer (no dtype needed here)
        self.tokenizer = None  # Subclasses will define this


    def _custom_tokenizer(self, text):
        """
        Tokenize the text using the tokenizer, returning tokenized strings (not token IDs) for TF-IDF or CountVectorizer.
        This tokenizer works for BERT, RoBERTa, and LLaMA models.
        
        Parameters:
        - text: The input text to be tokenized.
        
        Returns:
        - tokens: A list of tokens with special tokens removed based on the model in use.
        """

        tokens = self.tokenizer.tokenize(text, max_length=self.max_length, truncation=True)
        
        # Retrieve special tokens from the tokenizer object
        special_tokens = self.tokenizer.all_special_tokens                  # Dynamically fetch special tokens like [CLS], [SEP], <s>, </s>, etc.
        
        # Optionally, remove special tokens
        tokens = [token for token in tokens if token not in special_tokens]

        return tokens
    

    def _tokenize(self, texts):
        """
        Tokenize a batch of texts using `encode_plus` to ensure truncation and padding.
        """
        input_ids = []
        attention_masks = []

        for text in texts:
            # Use encode_plus to handle truncation, padding, and return attention mask
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,                                # Add special tokens like [CLS] and [SEP]
                max_length=self.max_length,                             # Truncate sequences to this length
                padding='max_length',                                   # Pad sequences to the max_length
                return_attention_mask=True,                             # Generate attention mask
                return_tensors='pt',                                    # Return PyTorch tensors
                truncation=True                                         # Ensure truncation to max_length
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        # Convert lists of tensors to a single tensor
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks
         

    def _compute_mean_embedding(self):
        """
        Compute the mean embedding vector using all tokens in the model's vocabulary.
        This vector will be used for handling OOV tokens.
        """
        vocab = list(self.tokenizer.get_vocab().keys())
        batch_size = self.batch_size
        total_embeddings = []

        # Process the tokens in batches with a progress bar
        with tqdm(total=len(vocab), desc="Computing mean embedding for model vocabulary", unit="token") as pbar:
            for i in range(0, len(vocab), batch_size):
                batch_tokens = vocab[i:i + batch_size]
                
                # Tokenize the batch of tokens, ensuring attention_mask is created
                inputs = self.tokenizer(batch_tokens, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)  # Ensure attention_mask is passed

                # Pass through the model to get token embeddings
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)  # Pass attention_mask
                token_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                total_embeddings.append(token_embeddings)

                # Update the progress bar with the actual batch size
                pbar.update(len(batch_tokens))

        # Concatenate all embeddings and compute the mean embedding
        total_embeddings = np.concatenate(total_embeddings, axis=0)
        mean_embedding = total_embeddings.mean(axis=0)  # Compute the mean embedding across all tokens
        print(f"Mean embedding shape: {mean_embedding.shape}")

        return mean_embedding


    def dim(self):
        # Return the hidden size of the Transformer model
        return self.model.config.hidden_size

    
    def extract(self, words, pooling='mean'):
        """
        Extract embeddings for a list of words using the Transformer model.
        
        Parameters:
        ----------
        words : list of str
            List of words for which embeddings are to be extracted.
        pooling : str, optional
            Pooling strategy to aggregate subword embeddings. Choices: 'mean', 'max'. Defaults to 'mean'.
            
        Returns:
        -------
        torch.Tensor
            Tensor of word embeddings where each row corresponds to a word.
        """
        if pooling not in ['mean', 'max']:
            raise ValueError("Pooling must be 'mean' or 'max'.")

        print(f"Extracting embeddings for words using {self.model_name} model with {pooling} pooling...")

        # Initialize empty list to store embeddings
        embeddings = []

        # Process words in batches to avoid OOM
        for batch in tqdm([words[i:i + self.batch_size] for i in range(0, len(words), self.batch_size)]):
            # Tokenize each word in the batch and collect their input IDs and attention masks
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.max_length,
                is_split_into_words=False  # Ensure tokens are processed as whole words
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            with torch.no_grad():  # No gradients needed during extraction
                try:
                    # Optionally enable mixed precision for memory optimization
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                    # Extract token embeddings (last hidden state of the model)
                    token_embeddings = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]

                    # Aggregate subword embeddings for each word in the batch
                    if pooling == 'mean':
                        # Mean pooling: average over sequence length
                        batch_embeddings = token_embeddings.mean(dim=1)
                    elif pooling == 'max':
                        # Max pooling: max over sequence length
                        batch_embeddings = token_embeddings.max(dim=1).values
                    
                    embeddings.append(batch_embeddings.cpu().numpy())

                except RuntimeError as e:
                    print(f"RuntimeError during extraction: {e}")
                    if 'out of memory' in str(e):
                        print('Clearing CUDA cache to free memory.')
                        torch.cuda.empty_cache()  # Clear the cache in case of memory issues
                    else:
                        raise e  # Re-raise if it's not a memory-related issue

            # Free up memory after each batch
            torch.cuda.empty_cache()

        # Concatenate all embeddings together after processing batches
        embeddings = np.concatenate(embeddings, axis=0)

        # Convert to PyTorch tensor and return as a float tensor
        extraction = torch.from_numpy(embeddings).float()

        return extraction

    
    def vocabulary(self):
        # Get the tokenizer vocabulary from the model
        return set(self.tokenizer.get_vocab().keys())
    

    def build_embedding_vocab_matrix(self):
        """
        Build the embedding vocabulary matrix for Transformer model, as a function of the underlying
        tokenizer and model architecture.
        """
        print("Building embedding vocab matrix for Transformer model...")

        self.model.eval()
        self.model = self.model.to(self.device)

        embedding_dim = self.model.config.hidden_size
        vocabulary = list(self.vectorizer.vocabulary_.keys())  # Tokens from the vectorizer

        # Initialize an empty embedding matrix
        self.embedding_vocab_matrix = np.zeros((len(vocabulary), embedding_dim))

        # Precompute the mean embedding for OOV handling
        self.mean_embedding = self._compute_mean_embedding()

        oov_list = []  # Keep track of OOV tokens

        # Helper function to process a batch of tokens
        def process_batch(batch_tokens):
            inputs = self.tokenizer(batch_tokens, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            # Get token embeddings from model
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Aggregate embeddings (mean pooling as default)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        batch_size = self.batch_size

        # Add a progress bar to track the processing of the vocabulary
        with tqdm(total=len(vocabulary), desc="computing transformer embedding vocab matrix...", unit="token") as pbar:
            for i in range(0, len(vocabulary), batch_size):
                batch_tokens = vocabulary[i:i + batch_size]
                embeddings = []

                for token in batch_tokens:
                    # Re-tokenize using the model tokenizer
                    subwords = self.tokenizer.tokenize(token)

                    if subwords:
                        # Process subwords as a batch
                        subword_embeddings = process_batch(subwords)
                        # Pool subword embeddings (mean pooling as default)
                        token_embedding = np.mean(subword_embeddings, axis=0)
                        embeddings.append(token_embedding)
                    else:
                        # Handle OOV tokens
                        oov_list.append(token)
                        embeddings.append(self.mean_embedding)

                # Assign embeddings to the embedding matrix
                for j, embedding in enumerate(embeddings):
                    self.embedding_vocab_matrix[i + j] = embedding

                # Update the progress bar by the batch size
                pbar.update(batch_size)

        print("self.embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        print(f"OOV tokens: {len(oov_list)} encountered.")

        return self.embedding_vocab_matrix


    def encode_docs(self, text_list, pooling_strategy='mean'):
        """
        Encode documents using Transformer-based embeddings.

        For each document:
        - Use vectorizer tokens (already tokenized by `_custom_tokenizer`).
        - For each token, compute its embedding using subwords via pooling (mean or max or summ (CLS)).
        - Aggregate token embeddings to form the document embedding.

        Parameters:
        ----------
        text_list : list of str
            List of documents to encode.
        pooling_strategy : str, optional
            Pooling strategy for subwords ('mean' or 'max or 'summ'), defaults to 'mean'.

        Returns:
        -------
        doc_embeddings : np.ndarray
            Array of document embeddings.
        doc_embeddings : np.ndarray (same as above)
        """
        print(f"Encoding documents using Transformer-based embeddings with {pooling_strategy} pooling...")

        self.model.eval()
        self.model = self.model.to(self.device)

        doc_embeddings = []
        summ_embeddings = []
        oov_tokens = []

        def process_document(document):
    
            inputs = self.tokenizer(
                document, 
                return_tensors='pt', 
                truncation=True, 
                max_length=self.max_length, 
                padding=True
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                token_vectors = outputs.last_hidden_state          # Shape: [batch_size, seq_len, hidden_dim]

                # Extract the first token embedding (e.g., [CLS] or <s>)
                first_token_embedding = token_vectors[:, 0, :].cpu().numpy()

                # Aggregate all token embeddings for the document
                if pooling_strategy == 'mean':
                    pooled_embedding = token_vectors.mean(dim=1).cpu().numpy()
                elif pooling_strategy == 'max':
                    pooled_embedding = token_vectors.max(dim=1).values.cpu().numpy()
                else:
                    raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")

            return pooled_embedding[0], first_token_embedding[0]
        

        for doc in tqdm(text_list, desc="encoding documents..."):
            doc_embedding, summ_embedding = process_document(doc)
            doc_embeddings.append(doc_embedding)
            summ_embeddings.append(summ_embedding)
 
        doc_embeddings = np.stack(doc_embeddings)
        summ_embeddings = np.stack(summ_embeddings)

        print(f"Document embeddings shape: {doc_embeddings.shape}")
        print(f"Summary embeddings shape: {summ_embeddings.shape}")
        
        return doc_embeddings, summ_embeddings


class BERTLCRepresentationModel(TransformerLCRepresentationModel):
    """
    BERTRepresentation subclass implementing sentence encoding using BERT
    """
    
    def __init__(self, model_name=BERT_MODEL, model_dir=VECTOR_CACHE+'/BERT', vtype='tfidf'):

        print("initializing BERT representation model...")

        super().__init__(model_name, model_dir)                             # parent constructor

        # instantiate model and tokenizer
        self.model = BertModel.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            torch_dtype=self.torch_dtype                    # set in base class init()
        ).to(self.device)
        
        self.tokenizer = BertTokenizerFast.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            do_lower_case=False                             # keep tokenizer case sensitive
        )

        self.max_length = self.tokenizer.model_max_length
        print("self.max_length:", self.max_length)

        # NB BertTokenizerFast has a pad_token
            
        self.type = 'bert'

        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT, 
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")
    

class RoBERTaLCRepresentationModel(TransformerLCRepresentationModel):
    """
    RoBERTaRepresentationModel subclass implementing sentence encoding using RoBERTa
    """

    def __init__(self, model_name=ROBERTA_MODEL, model_dir=VECTOR_CACHE+'/RoBERTa', vtype='tfidf'):

        print("initializing RoBERTa representation model...")

        super().__init__(model_name, model_dir)                             # parent constructor

        self.model = RobertaModel.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            torch_dtype=self.torch_dtype                    # set in base class init()
        ).to(self.device)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            do_lower_case=False                             # keep tokenizer case sensitive
        )
    
        self.max_length = self.tokenizer.model_max_length
        print("self.max_length:", self.max_length)

        # NB RoBERTaTokenizerFast has a pad_token

        self.type = 'roberta'

        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT, 
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")


class DistilBERTLCRepresentationModel(TransformerLCRepresentationModel):
    """
    DistilBERT representation model implementing sentence encoding using DistilBERT.
    """

    def __init__(self, model_name='distilbert-base-cased', model_dir=VECTOR_CACHE+'/DistilBERT', vtype='tfidf'):

        print("Initializing DistilBERT representation model...")

        super().__init__(model_name, model_dir)  # parent constructor

        # Load the DistilBERT model and tokenizer
        self.model = DistilBertModel.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            torch_dtype=self.torch_dtype                    # set in base class init()
        ).to(self.device)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            do_lower_case=False                             # keep tokenizer case-sensitive
        )

        self.max_length = self.tokenizer.model_max_length
        print("self.max_length:", self.max_length)

        self.type = 'distilbert'

        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT,
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT,
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")



class XLNetLCRepresentationModel(TransformerLCRepresentationModel):
    """
    XLNet representation model implementing sentence encoding using XLNet.
    """

    def __init__(self, model_name='xlnet-base-cased', model_dir=VECTOR_CACHE+'/XLNet', vtype='tfidf'):
        print("Initializing XLNet representation model...")

        super().__init__(model_name, model_dir)  # parent constructor

        # Load the XLNet model and tokenizer
        self.model = XLNetModel.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            torch_dtype=self.torch_dtype                    # set in base class init()
        ).to(self.device)
 
        self.tokenizer = XLNetTokenizerFast.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            do_lower_case=False                             # keep the model case sensitive
        )

        # Ensure padding token is available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id                   # Reuse the end-of-sequence token for padding

        if (XLNET_MODEL == 'xlnet-base-cased'):
            self.max_length = 512                  
        elif (XLNET_MODEL == 'xlnet-large-cased'):
            self.max_length = 1024
        else:
            raise ValueError("Invalid XLNet model. Must be in [xlnet-base-cased, xlnet-large-cased].")
            
        self.type = 'xlnet'

        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT,
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT,
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")
        
    
    def _custom_tokenizer(self, text):
        """
        Tokenize the text using the tokenizer, returning tokenized strings (not token IDs) for TF-IDF or CountVectorizer.
        This tokenizer works for BERT, RoBERTa, and LLaMA models.
        
        Parameters:
        - text: The input text to be tokenized.
        
        Returns:
        - tokens: A list of tokens with special tokens removed based on the model in use.
        """

        #tokens = self.tokenizer.tokenize(text, max_length=self.max_length, truncation=True)
        tokens = self.tokenizer.tokenize(text)
        
        # Retrieve special tokens from the tokenizer object
        special_tokens = self.tokenizer.all_special_tokens  # Dynamically fetch special tokens like [CLS], [SEP], <s>, </s>, etc.
        
        # Optionally, remove special tokens
        tokens = [token for token in tokens if token not in special_tokens]

        return tokens


    def build_embedding_vocab_matrix(self):
        """
        Builds the embedding vocabulary matrix using XLNet embeddings and tracks OOV tokens.
        """
        print("Building XLNet embedding vocab matrix...")

        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        self.vocab_size = len(self.vectorizer.vocabulary_)

        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Vocab size: {self.vocab_size}")

        self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        mean_vector = np.zeros(self.embedding_dim)

        oov_tokens = 0
        oov_list = []
        batch_words = []
        word_indices = []

        def process_batch(batch_words):
            # Tokenize the batch of words into subwords
            inputs = self.tokenizer(batch_words, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
            input_ids = inputs['input_ids'].to(self.device)

            # Get token embeddings from XLNet
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)

            # Average the embeddings of all tokens (subwords)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


        # Tokenize the vocabulary and build embedding matrix
        with tqdm(total=len(self.vectorizer.vocabulary_), desc="Processing XLNet embedding vocab matrix") as pbar:
                    
            for word, idx in self.vectorizer.vocabulary_.items():
                if word in self.tokenizer.get_vocab():
                    batch_words.append(word)
                    word_indices.append(idx)
                # If the case-sensitive version is not found, try the lowercase version
                elif word.lower() in self.tokenizer.get_vocab():
                    lower_word = word.lower()
                    batch_words.append(lower_word)
                    word_indices.append(idx)
                else:
                    # Track OOV tokens
                    oov_tokens += 1
                    oov_list.append(word)
                    self.embedding_vocab_matrix[idx] = mean_vector
                    
                if len(batch_words) == self.batch_size:
                    embeddings = process_batch(batch_words)
                    for i, embedding in zip(word_indices, embeddings):
                        self.embedding_vocab_matrix[i] = embedding
                    batch_words = []
                    word_indices = []
                    pbar.update(self.batch_size)

            if batch_words:
                embeddings = process_batch(batch_words)
                for i, embedding in zip(word_indices, embeddings):
                    self.embedding_vocab_matrix[i] = embedding
                pbar.update(len(batch_words))

        print("self.embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        print(f"oov_tokens: {oov_tokens}")
        #print(f"List of OOV tokens: {oov_list}")
    
        #return self.embedding_vocab_matrix, self.vectorizer.vocabulary_
        return self.embedding_vocab_matrix
    

    def encode_docs(self, text_list):
        """
        Encode documents using XLNet and extract token embeddings.
        """
        print("\n\tencoding docs using XLNet embeddings...")

        self.model.eval()
        mean_embeddings = []
        summ_embeddings = []

        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)

                # Move inputs to the appropriate device (MPS, CUDA, or CPU)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                # Forward pass through the model
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Extract the summary token embeddings (last token in XLNet)
                token_vectors = outputs[0]                                                      # Shape: [batch_size, sequence_length, hidden_size]
                summ_embeddings_batch = token_vectors[:, -1, :].cpu().detach().numpy()          # Last token embedding
                summ_embeddings.append(summ_embeddings_batch)

                # Compute the mean of all token embeddings
                mean_embeddings_batch = token_vectors.mean(dim=1).cpu().detach().numpy()
                mean_embeddings.append(mean_embeddings_batch)

        # Concatenate results
        mean_embeddings = np.concatenate(mean_embeddings, axis=0)
        summ_embeddings = np.concatenate(summ_embeddings, axis=0)

        print("mean_embeddings:", type(mean_embeddings), mean_embeddings.shape)
        print("summ_embeddings:", type(summ_embeddings), summ_embeddings.shape)

        return mean_embeddings, summ_embeddings
    


class GPT2LCRepresentationModel(TransformerLCRepresentationModel):
    """
    GPT-2 representation model implementing sentence encoding using GPT-2.
    """

    def __init__(self, model_name=GPT2_MODEL, model_dir=VECTOR_CACHE+'/GPT2', vtype='tfidf'):

        print("Initializing GPT-2 representation model...")

        super().__init__(model_name, model_dir)  # parent constructor

        # Load the GPT-2 model and tokenizer
        self.model = GPT2Model.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            torch_dtype=self.torch_dtype                        # set in parent constructor
        ).to(self.device)           
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            do_lower_case=False                                 # keep tokenizer case-sensitive 
        )

        # Ensure padding token is available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id                   # Reuse the end-of-sequence token for padding

        self.max_length = self.tokenizer.model_max_length
        print("self.max_length:", self.max_length)

        self.type = 'gpt2'

        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT,
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT,
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")


    def _custom_tokenizer(self, text):
        """
        Tokenize the text using the GPT-2 tokenizer, ensuring truncation and padding are applied.
        This method returns tokenized strings (subwords) for use with TF-IDF/CountVectorizer.
        """

        # Tokenize the text with GPT-2, applying truncation to limit sequence length
        tokenized_output = self.tokenizer(
            text,
            max_length=self.max_length,                     # Limit to model's max length (usually 1024)
            truncation=True,                                # Ensure sequences longer than max_length are truncated
            padding='max_length',                           # Optional: can pad to max length (if needed)
            return_tensors=None,                            # Return token strings, not tensor IDs
            add_special_tokens=False                        # Do not add special tokens like <|endoftext|> for TF-IDF use
        )

        # The tokenizer will return a dictionary, so we extract the tokenized sequence
        tokens = tokenized_output['input_ids']

        # Convert token IDs back to token strings for use with TF-IDF/CountVectorizer
        tokens = self.tokenizer.convert_ids_to_tokens(tokens)

        return tokens
        

    def build_embedding_vocab_matrix(self):
        """
        Builds the embedding vocabulary matrix using GPT-2 embeddings and tracks OOV tokens.
        """
        print("Building GPT-2 embedding vocab matrix...")

        self.model.eval()
        self.model = self.model.to(self.device)

        self.embedding_dim = self.model.config.hidden_size
        self.vocab_size = len(self.vectorizer.vocabulary_)

        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Vocab size: {self.vocab_size}")

        self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        mean_vector = np.zeros(self.embedding_dim)

        oov_tokens = 0
        oov_list = []
        batch_words = []
        word_indices = []

        def process_batch(batch_words):
            inputs = self.tokenizer(batch_words, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
            input_ids = inputs['input_ids'].to(self.device)

            # Get token embeddings from GPT-2
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)

            # Average embeddings across all tokens (instead of just the first)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()          # Mean pooling across tokens

        # Tokenize the vocabulary and build embedding matrix
        with tqdm(total=len(self.vectorizer.vocabulary_), desc="computing GPT-2 embedding vocab matrix") as pbar:
            for word, idx in self.vectorizer.vocabulary_.items():
                # Tokenize the word into subwords (this will handle cases where words are split into subword units)
                subwords = self.tokenizer.tokenize(word)

                if len(subwords) > 0:  # If tokenization was successful (no empty subwords)
                    batch_words.append(word)
                    word_indices.append(idx)
                else:
                    # If completely unknown (unlikely with GPT-2 tokenizer), mark as OOV
                    oov_tokens += 1
                    oov_list.append(word)
                    self.embedding_vocab_matrix[idx] = mean_vector

                if len(batch_words) == self.batch_size:
                    embeddings = process_batch(batch_words)
                    for i, embedding in zip(word_indices, embeddings):
                        self.embedding_vocab_matrix[i] = embedding
                    batch_words = []
                    word_indices = []
                    pbar.update(self.batch_size)

            if batch_words:
                embeddings = process_batch(batch_words)
                for i, embedding in zip(word_indices, embeddings):
                    self.embedding_vocab_matrix[i] = embedding
                pbar.update(len(batch_words))

        print("self.embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        print(f"oov_tokens: {oov_tokens}")
        #print(f"List of OOV tokens: {oov_list}")

        #return self.embedding_vocab_matrix, self.vectorizer.vocabulary_
        return self.embedding_vocab_matrix


    def encode_docs(self, text_list):
        """
        Encode documents using GPT2 model and extract token embeddings.
        """
        print("\n\tencoding docs using GPT2 embeddings...")

        self.model.eval()
        mean_embeddings = []
        summ_embeddings = []

        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)

                # Move inputs to the appropriate device (MPS, CUDA, or CPU)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                # Forward pass through the model
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Extract the summary token embeddings (last token in GPT2)
                token_vectors = outputs[0]                                                          # Shape: [batch_size, sequence_length, hidden_size]
                summ_embeddings_batch = token_vectors[:, -1, :].cpu().detach().numpy()              # Last token embedding
                summ_embeddings.append(summ_embeddings_batch)

                # Compute the mean of all token embeddings
                mean_embeddings_batch = token_vectors.mean(dim=1).cpu().detach().numpy()
                mean_embeddings.append(mean_embeddings_batch)

        # Concatenate results
        mean_embeddings = np.concatenate(mean_embeddings, axis=0)
        summ_embeddings = np.concatenate(summ_embeddings, axis=0)

        print("mean_embeddings:", type(mean_embeddings), mean_embeddings.shape)
        print("summ_embeddings:", type(summ_embeddings), summ_embeddings.shape)

        return mean_embeddings, summ_embeddings



class LlamaLCRepresentationModel(TransformerLCRepresentationModel):
    """
    Llama representation model for text encoding and embeddings.
    """

    def __init__(self, model_name=LLAMA_MODEL, model_dir=VECTOR_CACHE+'/Llama', vtype='tfidf'):
        """
        Initialize the Llama representation model.
        
        Parameters:
        ----------
        model_name : str, optional
            Name of the pretrained Llama model. Defaults to 'llama-7b'.
        model_dir : str, optional
            Directory to cache or load the model.
        vtype : str, optional
            Type of vectorization ('tfidf' or 'count'). Defaults to 'tfidf'.
        """
        print("Initializing Llama representation model...")

        super().__init__(model_name, model_dir)  # Call parent constructor

        # Load the Llama model and tokenizer
        self.model = LlamaModel.from_pretrained(
            model_name,
            cache_dir=model_dir,
            torch_dtype=self.torch_dtype,                           # set in parent constructor
            #device_map="auto"                                      # Automatically distribute the model across GPUs/CPU
        ).to(self.device)

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_name,
            cache_dir=model_dir,
            do_lower_case=False                                     # keep tokenizer case sensitive
        )

        # Ensure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            # Find the ID for '<|finetune_right_pad_id|>' in the tokenizer
            finetune_right_pad_token = "<|finetune_right_pad_id|>"
            if finetune_right_pad_token in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = finetune_right_pad_token
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(finetune_right_pad_token)
                print(f"PAD token set to '{finetune_right_pad_token}' with ID {self.tokenizer.pad_token_id}.")
            else:
                raise ValueError(f"Token '{finetune_right_pad_token}' not found in the tokenizer vocabulary.")
        
        # Verify the PAD token
        print("PAD token:", self.tokenizer.pad_token)
        print("PAD token ID:", self.tokenizer.pad_token_id)

        #self.max_length = self.tokenizer.model_max_length
        self.max_length = MAX_LENGTH                                # need to set to default value
        print(f"Llama max sequence length: {self.max_length}")

        self.type = 'llama'

        # Initialize vectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT,
                sublinear_tf=True,
                lowercase=False,
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT,
                lowercase=False,
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be 'tfidf' or 'count'.")

        
    def _custom_tokenizer(self, text):
        """
        Tokenize the text using the GPT-2 tokenizer, ensuring truncation and padding are applied.
        This method returns tokenized strings (subwords) for use with TF-IDF/CountVectorizer.
        """

        # Tokenize the text with LlamaTokenizerFast, applying truncation to limit sequence length
        tokenized_output = self.tokenizer(
            text,
            max_length=self.max_length,             # Limit to model's max length - set to 512)
            truncation=True,                        # Ensure sequences longer than max_length are truncated
            padding='max_length',                   # Optional: can pad to max length (if needed)
            return_tensors=None,                    # Return token strings, not tensor IDs
            add_special_tokens=False                 # Do not add special tokens like <|endoftext|> for TF-IDF use
        )

        # The tokenizer will return a dictionary, so we extract the tokenized sequence
        tokens = tokenized_output['input_ids']

        # Convert token IDs back to token strings for use with TF-IDF/CountVectorizer
        tokens = self.tokenizer.convert_ids_to_tokens(tokens)

        return tokens


    def _compute_mean_embedding(self):
        """
        Compute the mean embedding vector using all tokens in the model's vocabulary.
        This vector will be used for handling OOV tokens.
        """
        print("Computing mean embedding for model vocabulary...")

        vocab = list(self.tokenizer.get_vocab().keys())
        batch_size = self.batch_size
        total_embeddings = []

        # Get the tokenizer's vocabulary size
        vocab_size = self.tokenizer.vocab_size
        print("vocab_size:", vocab_size)

        # Process the tokens in batches with a progress bar
        with tqdm(total=len(vocab), desc="Computing mean embedding for model vocabulary", unit="token") as pbar:
            for i in range(0, len(vocab), batch_size):
                batch_tokens = vocab[i:i + batch_size]

                # Tokenize the batch of tokens, ensuring attention_mask is created
                inputs = self.tokenizer(
                    batch_tokens, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length
                )
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)  # Ensure attention_mask is passed

                # Pass through the model to get token embeddings
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)  # Pass attention_mask
                token_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                total_embeddings.append(token_embeddings)

                # Update the progress bar with the actual batch size
                pbar.update(len(batch_tokens))

        # Concatenate all embeddings and compute the mean embedding
        total_embeddings = np.concatenate(total_embeddings, axis=0)
        mean_embedding = total_embeddings.mean(axis=0)  # Compute the mean embedding across all tokens

        print(f"Mean embedding shape: {mean_embedding.shape}")

        return mean_embedding



class DeepSeekLCRepresentationModel(TransformerLCRepresentationModel):
    """
    DeepSeek representation model for text encoding and embeddings.
    """

    def __init__(self, model_name=DEEPSEEK_MODEL, model_dir=VECTOR_CACHE + '/DeepSeek', vtype='tfidf'):
        """
        Initialize the DeepSeek representation model.
        
        Args:
        - model_name: Name of the pretrained DeepSeek model (default: 'deepseek-small').
        - model_dir: Directory to cache or load the model.
        - vtype: Type of vectorization ('tfidf' or 'count').
        """
        print("Initializing DeepSeek representation model...")

        super().__init__(model_name, model_dir)  # Call parent constructor

        # Load model and tokenizer with optimizations
        self._load(
            model_name=model_name,
            cache_dir=model_dir,
            device_map="auto",                                      # Automatically distribute layers across GPUs/CPU
            max_memory={                                            # Restrict memory usage
                0: "40GiB",                                         # GPU 0 memory limit
                "cpu": "20GiB"                                      # CPU memory limit
            },
            offload_folder="./offload"                              # Folder for CPU offloading
        )

        # Ensure padding token is available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id                   # Reuse the end-of-sequence token for padding

        self.max_length = self.tokenizer.model_max_length
        print("self.max_length:", self.max_length)

        self.type = 'deepseek'

        # Configure vectorizer with the custom tokenizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT,
                sublinear_tf=True,
                lowercase=False,
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT,
                lowercase=False,
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be 'tfidf' or 'count'.")

        # Enable gradient checkpointing for memory savings
        self.model.gradient_checkpointing_enable()

        # Enable mixed precision to save memory and increase speed
        self.mixed_precision = True                                 # Set to False to disable mixed precision
        if self.mixed_precision:
            self.model = self.model.half()                          # Convert model to half precision

        print("DeepSeek model initialized and loaded onto device.")


    def _load(
        self,
        model_name: str,
        cache_dir: str,
        trust_remote_code: bool = True,
        device_map: Optional[Union[str, Dict[str, Any]]] = "auto",
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        offload_folder: Optional[str] = None,
        **kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        
        try:
            # Load model with device map, max memory, and CPU offloading
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype= self.torch_dtype,                      # set torch dtype from parent constructor value
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                **kwargs
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                do_lower_case=False,                        # keep tokenizer case sensitive
                trust_remote_code=trust_remote_code
            )
            print("[INFO]: DeepSeek model loaded successfully with original configuration")

        except ValueError as e:
            if "Unknown quantization type" in str(e):
                print("[WARNING]: Quantization type not supported directly. Attempting to load without quantization...")

                # Load configuration and remove quantization if unsupported
                self.config = AutoConfig.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code,
                )
                if hasattr(self.config, "quantization_config"):
                    delattr(self.config, "quantization_config")

                # Attempt to reload model without quantization
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=self.torch_dtype,                  # set torch dtype from parent constructor value
                    config=self.config,
                    trust_remote_code=trust_remote_code,
                    device_map=device_map,
                    max_memory=max_memory,
                    offload_folder=offload_folder,
                    **kwargs
                )

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    do_lower_case=False,                        # keep tokenizer case sensitive
                    trust_remote_code=trust_remote_code,
                )
                print("[INFO]: Model loaded successfully without quantization")

            else:
                print(f"[ERROR]: Unexpected error during model loading: {str(e)}")
                raise


    def extract(self, words, pooling='mean'):
        """
        Extract embeddings for a list of words using the Transformer model.
        
        Parameters:
        ----------
        words : list of str
            List of words for which embeddings are to be extracted.
        pooling : str, optional
            Pooling strategy to aggregate subword embeddings. Choices: 'mean', 'max'. Defaults to 'mean'.
            
        Returns:
        -------
        torch.Tensor
            Tensor of word embeddings where each row corresponds to a word.
        """
        if pooling not in ['mean', 'max']:
            raise ValueError("Pooling must be 'mean' or 'max'.")

        print(f"Extracting embeddings for words using {self.model_name} model with {pooling} pooling...")

        embeddings = []
        self.model.eval()

        for batch in tqdm([words[i:i + self.batch_size] for i in range(0, len(words), self.batch_size)], desc="Extracting embeddings"):
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.max_length,
                is_split_into_words=False
            ).to(self.model.device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state

                if pooling == 'mean':
                    batch_embeddings = token_embeddings.mean(dim=1)
                elif pooling == 'max':
                    batch_embeddings = token_embeddings.max(dim=1).values

                embeddings.append(batch_embeddings.cpu().numpy())

            torch.cuda.empty_cache()

        embeddings = np.concatenate(embeddings, axis=0)
        return torch.from_numpy(embeddings).float()


    def build_embedding_vocab_matrix(self):
        """
        Build the embedding vocabulary matrix for the DeepSeek model.

        Returns:
        -------
        np.ndarray
            Embedding vocabulary matrix where each row corresponds to a token.
        """
        print("Building embedding vocab matrix for DeepSeek model...")

        self.model.eval()
        embedding_dim = self.model.config.hidden_size
        vocabulary = list(self.vectorizer.vocabulary_.keys())

        self.embedding_vocab_matrix = np.zeros((len(vocabulary), embedding_dim))
        self.mean_embedding = self._compute_mean_embedding()
        oov_list = []

        with tqdm(total=len(vocabulary), desc="Processing vocabulary tokens", unit="token") as pbar:
            for i, token in enumerate(vocabulary):
                subwords = self.tokenizer.tokenize(token)
                if subwords:
                    inputs = self.tokenizer(
                        subwords,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=self.max_length
                    ).to(self.model.device)

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        token_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                    self.embedding_vocab_matrix[i] = token_embeddings.mean(axis=0)
                else:
                    oov_list.append(token)
                    self.embedding_vocab_matrix[i] = self.mean_embedding

                pbar.update(1)

        print(f"Finished building embedding matrix. OOV tokens: {len(oov_list)}")
        return self.embedding_vocab_matrix


    def encode_docs(self, text_list, pooling_strategy='mean'):
        """
        Encode documents using Transformer-based embeddings.

        Parameters:
        ----------
        text_list : list of str
            List of documents to encode.
        pooling_strategy : str, optional
            Pooling strategy for subwords ('mean', 'max'). Defaults to 'mean'.

        Returns:
        -------
        np.ndarray
            Array of document embeddings.
        np.ndarray
            Array of summary embeddings.
        """
        print(f"Encoding documents using DeepSeek Transformer-based embeddings with {pooling_strategy} pooling...")

        self.model.eval()
        doc_embeddings = []
        summ_embeddings = []

        def process_document(document):
            inputs = self.tokenizer(
                document,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                token_vectors = outputs.last_hidden_state

                first_token_embedding = token_vectors[:, 0, :].cpu().numpy()
                if pooling_strategy == 'mean':
                    pooled_embedding = token_vectors.mean(dim=1).cpu().numpy()
                elif pooling_strategy == 'max':
                    pooled_embedding = token_vectors.max(dim=1).values.cpu().numpy()
                else:
                    raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")

            return pooled_embedding[0], first_token_embedding[0]

        for doc in tqdm(text_list, desc="Encoding documents"):
            doc_embedding, summ_embedding = process_document(doc)
            doc_embeddings.append(doc_embedding)
            summ_embeddings.append(summ_embedding)

        return np.stack(doc_embeddings), np.stack(summ_embeddings)
