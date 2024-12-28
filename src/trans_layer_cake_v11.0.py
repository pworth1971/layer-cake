import argparse
import os
import numpy as np
import pandas as pd
from time import time
import pickle

import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.sparse import csr_matrix

from typing import Dict, Union, Any

import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords

# sklearn
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from collections import defaultdict


# PyTorch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Adam
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR

# HuggingFace Transformers library
import transformers     
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, DistilBertForSequenceClassification
from transformers import XLNetForSequenceClassification, GPT2ForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoConfig, PreTrainedModel

# Custom
from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1

#from data.lc_dataset import trans_lc_load_dataset, SUPPORTED_DATASETS
from data.lc_trans_dataset import SUPPORTED_DATASETS, RANDOM_SEED, PICKLE_DIR
from data.lc_trans_dataset import get_dataset_data, show_class_distribution, check_empty_docs, spot_check_documents

from util.metrics import evaluation_nn
from util.common import initialize_testing, get_embedding_type

#from embedding.supervised import get_supervised_embeddings, compute_supervised_embeddings, compute_tces, embedding_matrices, embedding_matrix
from embedding.supervised import compute_tces
from embedding.pretrained import MODEL_MAP

import unicodedata



VECTOR_CACHE = "../.vector_cache"



#
# hyper parameters
#
VAL_SIZE = 0.20                     # percentage of data to be set aside for model validation
MC_THRESHOLD = 0.5                  # Multi-class threshold
PATIENCE = 5                        # Early stopping patience
LEARNING_RATE = 1e-6                # Learning rate
EPOCHS = 33

MAX_TOKEN_LENGTH = 512              # Maximum token length for transformer models models

#
# batch sizes for training and testing
# set depending upon dataset size
#
DEFAULT_MIN_CPU_BATCH_SIZE = 8
DEFAULT_MIN_MPS_BATCH_SIZE = 8
DEFAULT_MIN_CUDA_BATCH_SIZE = 8

DEFAULT_MAX_CPU_BATCH_SIZE = 32
DEFAULT_MAX_MPS_BATCH_SIZE = 32
DEFAULT_MAX_CUDA_BATCH_SIZE = 32

#
# supported operations for transformer classifier
# combination method with TCEs
#
SUPPORTED_OPS = ["cat", "add", "dot"]


class LCTokenizer:

    def __init__(self, model_name, model_path, lowercase=False, remove_special_tokens=False, padding='max_length', truncation=True):
        """
        Wrapper around Hugging Face tokenizer for custom tokenization.

        Args:
            tokenizer: Hugging Face tokenizer object.
            max_length: Maximum token length for truncation.
            lowercase: Whether to convert text to lowercase.
            remove_special_tokens: Whether to remove special tokens from tokenized output.
            padding: Padding strategy ('max_length', True, False). Defaults to 'max_lenth'.
            truncation: Truncation strategy. Defaults to True.
        """

        #print(f"LCTokenizer:__init__()... model_name: {model_name}, model_path: {model_path}, max_length: {max_length}, lowercase: {lowercase}, remove_special_tokens: {remove_special_tokens}, padding: {padding}, truncation: {truncation}")

        self.model_name = model_name
        self.model_path = model_path

        self.lowercase = lowercase
        self.remove_special_tokens = remove_special_tokens
        self.padding = padding
        self.truncation = truncation
        
        # Debugging information
        print("LCTokenizer initialized with the following parameters:")
        print(f"  Model name: {self.model_name}")
        print(f"  Model path: {self.model_path}")
        print(f"  Lowercase: {self.lowercase}")
        print(f"  Remove special tokens: {self.remove_special_tokens}")
        print(f"  Padding: {self.padding}")
        print(f"  Truncation: {self.truncation}")
        
        print("creating tokenizer using HF AutoTokenizer...")

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_path)

        # Use an existing token as the padding token
        if self.tokenizer.pad_token is None:
            print(f"Tokenizer has no pad token. Reusing 'eos_token' ({self.tokenizer.eos_token_id}).")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Print tokenizer details
        self.vocab_size = len(self.tokenizer)
        print("self.vocab_size:", self.vocab_size)

        # Compute max_length from tokenizer
        self.max_length = self.tokenizer.model_max_length
        print(f"self.tokenizer max_length: {self.max_length}")

        # Handle excessive or default max_length values
        if self.max_length > MAX_TOKEN_LENGTH:
            print(f"Invalid max_length ({self.max_length}) detected. Adjusting to {MAX_TOKEN_LENGTH}.")
            self.max_length = MAX_TOKEN_LENGTH

        # Print tokenizer details for debugging
        print("Tokenizer configuration:")
        print(f"  Pad token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        print(f"  Max length: {self.max_length}")


    def tokenize(self, text):
        """
        Tokenize input text using the Hugging Face tokenizer.
        
        Args:
            text: Input string to tokenize.

        Returns:
            List of tokens.
        """
        if self.lowercase:
            text = text.lower()

        tokens = self.tokenizer.tokenize(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding
        )

        if self.remove_special_tokens:
            special_tokens = self.tokenizer.all_special_tokens
            tokens = [token for token in tokens if token not in special_tokens]

        return tokens


    def normalize_text(self, text):
        """Normalize text to handle special characters and encodings."""
        return unicodedata.normalize('NFKC', text)


    def get_vocab(self):
        """
        Return the vocabulary of the Hugging Face tokenizer.

        Returns:
            Dict of token-to-index mappings.
        """
        return self.tokenizer.get_vocab()


    def __call__(self, text):
        """
        Enable the object to be called as a function for tokenization.
        
        Args:
            text: Input string to tokenize.

        Returns:
            List of tokens.
        """
        return self.tokenize(text)


    def get_dataset_tokens(self, train_texts, val_texts, test_texts):
        """
        Compute the dataset vocabulary as token IDs that align with the tokenizer's vocabulary.

        Args:
            train_texts (list of str): Training set texts.
            val_texts (list of str): Validation set texts.
            test_texts (list of str): Test set texts.

        Returns:
            list: Relevant token IDs for the dataset vocabulary.
        """
        print("Computing dataset token set...")

        print(f"max_length: {self.max_length}, padding: {self.padding}, truncation: {self.truncation}")

        # Combine all texts from training, validation, and test sets
        all_texts = train_texts + val_texts + test_texts

        # Use a set to store unique token IDs from the dataset
        dataset_vocab_ids = set()

        # Tokenize each document with the same parameters used during input preparation
        for text in all_texts:

            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_attention_mask=False,  # No need for attention masks here
                return_token_type_ids=False,  # No need for token type IDs here
            )

            # Debug: Check for tokens not in tokenizer vocabulary
            tokenizer_vocab_size = len(self.tokenizer.get_vocab())
            out_of_vocab_tokens = [token_id for token_id in tokens['input_ids'] if token_id >= tokenizer_vocab_size]

            if out_of_vocab_tokens:
                print(f"[DEBUG] Found {len(out_of_vocab_tokens)} tokens outside vocabulary in text: {text[:50]}...")
                print(f"[DEBUG] Out-of-vocab token IDs: {out_of_vocab_tokens}")

            # Add token IDs to the dataset vocabulary
            dataset_vocab_ids.update(tokens['input_ids'])

        # Ensure token IDs are in the tokenizer's range
        tokenizer_vocab_size = len(self.tokenizer.get_vocab())
        relevant_token_ids = [
            token_id for token_id in dataset_vocab_ids if token_id < tokenizer_vocab_size
        ]

        print(f"Computed dataset vocabulary: {len(relevant_token_ids)} relevant tokens out of {tokenizer_vocab_size} total tokens in tokenizer.")

        return relevant_token_ids



    def get_dataset_tokens_orig(self, train_texts, val_texts, test_texts):
        """
        Compute the dataset vocabulary as token IDs that align with the tokenizer's vocabulary.

        Args:
            train_texts (list of str): Training set texts.
            val_texts (list of str): Validation set texts.
            test_texts (list of str): Test set texts.

        Returns:
            list: Relevant token IDs for the dataset vocabulary.
        """
        print("computing dataset token set...")

        print(f'max_length: {self.max_length}, padding: {self.padding}, truncation: {self.truncation}')

        # Combine all texts from training, validation, and test sets
        all_texts = train_texts + val_texts + test_texts

        # Use a set to store unique token IDs from the dataset
        dataset_vocab_ids = set()

        # Tokenize each document with the same parameters used during input preparation
        for text in all_texts:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_attention_mask=False,  # No need for attention masks here
                return_token_type_ids=False,  # No need for token type IDs here
            )
            # Add token IDs to the dataset vocabulary
            dataset_vocab_ids.update(tokens['input_ids'])

        # Ensure token IDs are in the tokenizer's range
        tokenizer_vocab_size = len(self.tokenizer.get_vocab())
        relevant_token_ids = [
            token_id for token_id in dataset_vocab_ids if token_id < tokenizer_vocab_size
        ]

        print(f"Computed dataset vocabulary: {len(relevant_token_ids)} relevant tokens out of {tokenizer_vocab_size} total tokens in tokenizer.")

        return relevant_token_ids




class LCTokenizer_orig:

    def __init__(self, tokenizer, max_length, lowercase=False, remove_special_tokens=False):
        """
        Wrapper around Hugging Face tokenizer for custom tokenization.

        Args:
            tokenizer: Hugging Face tokenizer object.
            max_length: Maximum token length for truncation.
            lowercase: Whether to convert text to lowercase.
            remove_special_tokens: Whether to remove special tokens from tokenized output.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lowercase = lowercase
        self.remove_special_tokens = remove_special_tokens

    def tokenize(self, text):
        """
        Tokenize input text using the Hugging Face tokenizer.
        
        Args:
            text: Input string to tokenize.

        Returns:
            List of tokens.
        """
        if self.lowercase:
            text = text.lower()

        tokens = self.tokenizer.tokenize(
            text,
            max_length=self.max_length,
            truncation=True
        )

        if self.remove_special_tokens:
            special_tokens = self.tokenizer.all_special_tokens
            tokens = [token for token in tokens if token not in special_tokens]

        return tokens

    def normalize_text(self, text):
        """Normalize text to handle special characters and encodings."""
        return unicodedata.normalize('NFKC', text)

    def get_vocab(self):
        """
        Return the vocabulary of the Hugging Face tokenizer.

        Returns:
            Dict of token-to-index mappings.
        """
        return self.tokenizer.get_vocab()


    def get_dataset_tokens(self, train_texts, val_texts, test_texts, padding=False, truncation=True):
        """
        Compute the dataset vocabulary as token IDs that align with the tokenizer's vocabulary.

        Args:
            train_texts (list of str): Training set texts.
            val_texts (list of str): Validation set texts.
            test_texts (list of str): Test set texts.
            padding (bool or str, optional): Padding strategy ('max_length', True, False). Defaults to False.
            truncation (bool, optional): Whether to truncate sequences to `max_length`. Defaults to True.

        Returns:
            list: Relevant token IDs for the dataset vocabulary.
        """
        print("get_dataset_tokens()...")

        # Combine all texts from training, validation, and test sets
        all_texts = train_texts + val_texts + test_texts

        # Use a set to store unique token IDs from the dataset
        dataset_vocab_ids = set()

        # Tokenize each document with the same parameters used during input preparation
        for text in all_texts:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=padding,
                truncation=truncation,
                return_attention_mask=False,  # No need for attention masks here
                return_token_type_ids=False,  # No need for token type IDs here
            )
            # Add token IDs to the dataset vocabulary
            dataset_vocab_ids.update(tokens['input_ids'])

        # Ensure token IDs are in the tokenizer's range
        tokenizer_vocab_size = len(self.tokenizer.get_vocab())
        relevant_token_ids = [
            token_id for token_id in dataset_vocab_ids if token_id < tokenizer_vocab_size
        ]

        print(f"Computed dataset vocabulary: {len(relevant_token_ids)} relevant tokens out of {tokenizer_vocab_size} total tokens in tokenizer.")
        return relevant_token_ids
    

    def __call__(self, text):
        """
        Enable the object to be called as a function for tokenization.
        
        Args:
            text: Input string to tokenize.

        Returns:
            List of tokens.
        """
        return self.tokenize(text)




class LCTFIDFVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, tokenizer, lowercase=False, debug=False):
        """
        Custom TF-IDF Vectorizer that aligns its vocabulary with the Hugging Face tokenizer.

        Args:
            tokenizer: Hugging Face tokenizer object.
            lowercase: Whether to convert text to lowercase.
            debug: Whether to enable debugging messages.
        """
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.debug = debug
        self.vocabulary_ = {token: idx for token, idx in tokenizer.get_vocab().items()}
        self.idf_ = None


    def fit(self, tokenized_documents, y=None):
        """
        Fit the vectorizer to the tokenized documents.

        Args:
            tokenized_documents: List of tokenized documents (lists of tokens or strings of tokenized text).
            y: Ignored, present for compatibility with sklearn pipelines.
        """
        print("Fitting LCTFIDFVectorizer to tokenized docs...")

        print("tokenized_documents: ", type(tokenized_documents), len(tokenized_documents))
        print("tokenized_documents[0]: ", type(tokenized_documents[0]), tokenized_documents[0])

        term_doc_counts = defaultdict(int)

        for doc_idx, tokens in enumerate(tokenized_documents):
            # Ensure tokens are processed using the tokenizer
            if isinstance(tokens, str):
                tokens = self.tokenizer.tokenize(tokens)
            
            # Filter out blank tokens and special tokens not in the vocabulary
            tokens = [token for token in tokens if token.strip()]
            unique_tokens = set(tokens)

            unmatched_tokens = []

            for token in unique_tokens:
                if token in self.vocabulary_:
                    term_doc_counts[token] += 1
                else:
                    unmatched_tokens.append(token)

            if self.debug and unmatched_tokens:
                print(f"[DEBUG] Document {doc_idx} has {len(unmatched_tokens)} unmatched tokens: {unmatched_tokens[:10]}")

        num_documents = len(tokenized_documents)
        self.idf_ = np.zeros(len(self.vocabulary_), dtype=np.float64)
        for token, idx in self.vocabulary_.items():
            doc_count = term_doc_counts.get(token, 0)
            self.idf_[idx] = np.log((1 + num_documents) / (1 + doc_count)) + 1

            if self.debug and self.idf_[idx] == 0:
                print(f"[DEBUG] IDF for token '{token}' is 0 during fit. "
                    f"Document count: {doc_count}, Total docs: {num_documents}.")

        if self.debug:
            # Debug: Check if special tokens are present
            special_tokens = self.tokenizer.all_special_tokens
            for token in special_tokens:
                if token not in self.vocabulary_:
                    print(f"[WARNING] Special token '{token}' not found in the vocabulary.")
                else:
                    print(f"[INFO] Special token '{token}' is correctly included in the vocabulary.")

        return self



    def transform(self, tokenized_documents, original_documents=None):
        """
        Transform the tokenized documents to TF-IDF features.

        Args:
            tokenized_documents: List of tokenized documents (lists of tokens or strings of tokenized text).
            original_documents: List of original documents (strings).

        Returns:
            Sparse matrix of TF-IDF features.
        """
        print("Transforming tokenized docs with fitted LCTFIDFVectorizer...")

        print("tokenized_documents: ", type(tokenized_documents), len(tokenized_documents))
        print("tokenized_documents[0]: ", type(tokenized_documents[0]), tokenized_documents[0])

        rows, cols, data = [], [], []
        empty_rows_details = []  # To collect details of empty rows

        for row_idx, doc in enumerate(tokenized_documents):

            # If the document is a string, split it into tokens
            if isinstance(doc, str):
                tokens = doc.split()
            else:
                raise ValueError("row in tokenized doc is not a 'str'")
                #tokens = doc

            # Save the original document tokens for debugging
            original_tokens = doc.split()

            # Filter out blank tokens
            tokens = [token for token in tokens if token.strip()]
            term_freq = defaultdict(int)
            unmatched_tokens = []

            for token in tokens:
                if token in self.vocabulary_:
                    term_freq[token] += 1
                else:
                    unmatched_tokens.append(token)
            
            if self.debug and unmatched_tokens:
                print(f"[WARNING] Document {row_idx} has {len(unmatched_tokens)} unmatched tokens: {unmatched_tokens[:10]}...")
            
            # Collect unmatched tokens for empty rows
            if not term_freq:  # No matched tokens
                if original_documents is not None:
                    empty_rows_details.append((row_idx, original_documents[row_idx], original_tokens, unmatched_tokens))
                else:
                    empty_rows_details.append((row_idx, None, original_tokens, unmatched_tokens))

            # Calculate TF-IDF
            for token, freq in term_freq.items():
                col_idx = self.vocabulary_[token]
                tf = freq / len(tokens)
                tfidf = tf * self.idf_[col_idx]

                """
                if self.debug:
                    print(f"[INFO] Document {row_idx}, Token '{token}': Frequency: {freq}, TF: {tf}, IDF: {self.idf_[col_idx]}, TF-IDF: {tfidf}")
                """

                rows.append(row_idx)
                cols.append(col_idx)
                data.append(tfidf)

        # construct sparse matrix
        matrix = csr_matrix((data, (rows, cols)), shape=(len(tokenized_documents), len(self.vocabulary_)))

        if self.debug:
            empty_rows = matrix.sum(axis=1).A1 == 0
            for row_idx, original_doc, original_tokens, unmatched_tokens in empty_rows_details:
                if empty_rows[row_idx]:
                    print(f"[WARNING] Row {row_idx} in TF-IDF matrix is empty.")
                    print(f"[INFO] Original document: {original_doc}")
                    print(f"[INFO] Original tokens: {original_tokens}")
                    print(f"[INFO] Unmatched tokens (not in vocab): {unmatched_tokens}")
                    # Manually tokenize with the custom tokenizer
                    if original_doc:
                        custom_tokens = self.tokenizer.tokenize(original_doc)
                        print(f"[DEBUG] Custom tokenizer tokens: {custom_tokens}")
                    
        return matrix


    def fit_transform(self, X, y=None, original_documents=None):
        """
        Fit to data, then transform it.

        Args:
            X: List of tokenized documents (lists of tokens or strings of tokenized text).
            y: Ignored, present for compatibility with sklearn pipelines.
            original_documents: List of original documents before tokenization, for debugging.

        Returns:
            Sparse matrix of TF-IDF features.
        """
        print("Fit-transforming LCTFIDFVectorizer to tokenized docs...")

        self.fit(X, y)
        return self.transform(X, original_documents=original_documents)



def vectorize(texts_train, texts_val, texts_test, lc_tokenizer, debug=False):

    print(f'vectorize(), max_length: {lc_tokenizer.max_length}')

    print("lc_tokenizer:\n", lc_tokenizer)
    
    preprocessed_train = [" ".join(lc_tokenizer(text)) for text in texts_train]
    preprocessed_val = [" ".join(lc_tokenizer(text)) for text in texts_val]
    preprocessed_test = [" ".join(lc_tokenizer(text)) for text in texts_test]

    # Debugging: Preprocessed data
    print("preprocessed_train:", type(preprocessed_train), len(preprocessed_train))
    print(f"preprocessed_train[0]: {preprocessed_train[0]}")
    
    tokenizer_vocab = lc_tokenizer.tokenizer.get_vocab()

    vectorizer = LCTFIDFVectorizer(
        tokenizer=lc_tokenizer.tokenizer, 
        debug=debug
        )

    Xtr = vectorizer.fit_transform(
        X=preprocessed_train,
        original_documents=texts_train
        )
    
    Xval = vectorizer.transform(
        X=preprocessed_val,
        original_documents=texts_val
        )
    
    Xte = vectorizer.transform(
        X=preprocessed_test,
        original_documents=texts_test
        )

    def check_empty_rows(matrix, name, original_texts):
        empty_rows = matrix.sum(axis=1).A1 == 0
        if empty_rows.any():
            print(f"[WARNING] {name} contains {empty_rows.sum()} empty rows.")
            for i in range(len(empty_rows)):
                if empty_rows[i]:
                    print(f"Empty row {i}: Original text: '{original_texts[i]}'")

    check_empty_rows(Xtr, "Xtr", texts_train)
    check_empty_rows(Xval, "Xval", texts_val)
    check_empty_rows(Xte, "Xte", texts_test)

    vec_vocab_size = len(vectorizer.vocabulary_)
    tok_vocab_size = len(tokenizer_vocab)

    assert vec_vocab_size == tok_vocab_size, \
        f"Vectorizer vocab size ({vec_vocab_size}) must equal tokenizer vocab size ({tok_vocab_size})"

    return vectorizer, Xtr, Xval, Xte



def get_vectorized_data(texts_train, texts_val, test_data, lc_tokenizer, dataset, pretrained, vtype='tfidf', debug=False):
    """
    Wrapper for vectorize() method to save and load from a pickle file.

    Parameters:
        texts_train (list): Training texts.
        texts_val (list): Validation texts.
        test_data (list): Test texts.
        lc_tokenizer: LCTokenizer instance.
        dataset (str): Dataset name.
        pretrained (str): Pretrained model name.
        vtype (str): Vectorization type.

    Returns:
        tuple: vectorizer, lc_tokenizer, Xtr, Xval, Xte
    """
    pickle_file = os.path.join(PICKLE_DIR, f'vectors_{dataset}.{pretrained}.{vtype}.pickle')

    # Check if the pickle file exists
    if os.path.exists(pickle_file):
        print(f"Loading vectorized data from {pickle_file}...")
        with open(pickle_file, 'rb') as f:
            vectorizer, lc_tokenizer, Xtr, Xval, Xte = pickle.load(f)
    else:
        print(f"Pickle file not found. Vectorizing data and saving to {pickle_file}...")
        vectorizer, Xtr, Xval, Xte = vectorize(
            texts_train, 
            texts_val, 
            test_data, 
            lc_tokenizer,
            debug=debug
        )
        # Save the results to the pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump((vectorizer, lc_tokenizer, Xtr, Xval, Xte), f)

    return vectorizer, Xtr, Xval, Xte



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------



class LCSequenceClassifier(nn.Module):

    def __init__(self, 
                hf_model: nn.Module,
                num_classes: int, 
                vocab_size: int, 
                relevant_tokens: list,                                  # subset of relevant tokens for the dataset
                lc_tokenizer: LCTokenizer,                             
                simple: bool = True,                                    # whether or not to use the simple classifier head
                class_type: str = 'single-label', 
                class_weights: torch.Tensor = None, 
                supervised: bool = False, 
                tce_matrix: torch.Tensor = None, 
                finetune: bool = False, 
                normalize_tces: bool = True,
                dropout_rate: float = 0.3, 
                comb_method: str = "cat", 
                debug: bool = False):
        """
        A Transformer-based Sequence Classifier with optional TCE integration and parameters for Layer Cake Text
        Classification testing. Supports both single label and multi-label classification.
        
        Args:
            hf_model: The HuggingFace pre-trained transformer model (e.g., BERT), preloaded.
            num_classes: Number of classes for classification.
            vocab_size: size of (tokenizer) vocabulary, for assertions
            relevant_tokens: List of tokens to be considered relevant for classification (from embeddings).
            lc_tokenizer: Custom tokenizer for the dataset.
            simple: Whether to use the simple classifier head (just the one Linear layer after HF SeuqenceClassifier).
            class_type: type of classification problem, either 'single-label' or 'multi-label'
            class_weights: Class weights for loss function.
            supervised: Boolean indicating if supervised embeddings are used.
            tce_matrix: Precomputed TCE matrix (Tensor) with shape [vocab_size, num_classes].
            finetune: Boolean indicating whether or not the Embedding layer is trainable.
            normalize_tce: Boolean indicating if TCE matrix is normalized.
            dropout: Dropout rate for TCE matrix.
            comb-method: Method to integrate WCE embeddings ("add", "dot" or "cat").            
            debug: Debug mode flag.
        """
        super(LCSequenceClassifier, self).__init__()

        print(f'LCSequenceClassifier:__init__()... class_type: {class_type}, simple: {simple}, num_classes: {num_classes}, finetune: {finetune}, supervised: {supervised}, debug: {debug}')

        if (supervised):
            print(f'normalize_tces: {normalize_tces}, dropout_rate: {dropout_rate}, comb_method: {comb_method}')

        self.debug = debug

        self.tokenizer = lc_tokenizer.tokenizer
        print("self.tokenizer:", self.tokenizer)

        if relevant_tokens is not None:
            #
            # Filter the transformer embeddings to only include relevant tokens
            # that are present from the Dataset vocabulary (as tokenized)
            #
            self.transformer, self.relevant_embeddings = self._filter_transformer_embeddings(hf_model=hf_model, relevant_tokens=relevant_tokens)                    
        else:
            self.transformer = hf_model
            self.relevant_embeddings = None

        print("self.transformer:", self.transformer)

        self.hidden_size = self.transformer.config.hidden_size          
        print("self.hidden_size:", self.hidden_size)

        self.num_classes = num_classes
        self.class_type = class_type

        self.supervised = supervised
        self.comb_method = comb_method
        self.normalize_tces = normalize_tces
        self.finetune = finetune
        self.class_weights = class_weights

        if (self.class_weights is not None):
            print("self.class_weights.shape:", self.class_weights.shape)
            if (self.debug):
                print("self.class_weights:", self.class_weights)

            # Loss functions
            if class_type == 'multi-label':
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            elif class_type == 'single-label':
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                raise ValueError("class_type must be 'single-label' or 'multi-label'")
            print("loss_fn:", self.loss_fn)

        else:
            print("self.class_weights is None")

            # Loss functions
            if class_type == 'multi-label':
                self.loss_fn = nn.BCEWithLogitsLoss()
            elif class_type == 'single-label':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                raise ValueError("class_type must be 'single-label' or 'multi-label'")
            print("loss_fn:", self.loss_fn)
        
        self.vocab_size = vocab_size
        print("self.vocab_size:", self.vocab_size)

        #
        # Optionally unfreeze the embedding layer if finetune is True
        #
        if self.finetune:
            print("finetuning == True, making the embedding layer (only) trainable")    

            # Freeze gradient computation for all transformer parameters
            for param in self.transformer.parameters():
                param.requires_grad = False

            # Enable training of only the embedding layer
            if hasattr(self.transformer, 'bert'):
                embedding_layer = self.transformer.bert.embeddings
            elif hasattr(self.transformer, 'roberta'):
                embedding_layer = self.transformer.roberta.embeddings
            elif hasattr(self.transformer, 'transformer'):
                embedding_layer = self.transformer.transformer.wte  # GPT-2
            else:
                raise AttributeError(f"Embeddings not found for the given model: {type(self.transformer)}")
            print("embedding_layer:", type(embedding_layer), embedding_layer)

            for param in embedding_layer.parameters():
                param.requires_grad = True
        else:            
            print("finetune == False, default model configuration ...")


        # Assert that tce_matrix is provided if supervised is True
        if self.supervised:
            assert tce_matrix is not None, "tce_matrix must be provided when supervised is True."

        self.tce_matrix = tce_matrix
        self.tce_layer = None               # we initialize this only if we are using TCEs 

        # initialize embedding dimensions to model embedding dimension
        # we over-write this only if we are using supervised tces with the 'cat' method
        combined_size = self.hidden_size

        if (self.supervised and self.tce_matrix is not None):

            print("supervised is True, original tce_matrix:", type(self.tce_matrix), self.tce_matrix.shape)
 
            with torch.no_grad():                           # normalization code should be in this block

                # Normalize TCE matrix if required
                if self.normalize_tces:

                    # compute the mean and std from the core model embeddings
                    embedding_layer = self.transformer.get_input_embeddings()
                    embedding_mean = embedding_layer.weight.mean(dim=0).to(device)
                    embedding_std = embedding_layer.weight.std(dim=0).to(device)
                    if (self.debug):
                        print(f"transformer embeddings mean: {embedding_mean.shape}, std: {embedding_std.shape}")

                    # normalize the TCE matrix
                    self.tce_matrix = self._normalize_tce(self.tce_matrix, embedding_mean, embedding_std)
                    print(f"Normalized TCE matrix: {type(self.tce_matrix)}, {self.tce_matrix.shape}")
            
                # initialize the TCE Embedding layer, freeze the embeddings if trainable_tces == False
                """
                if (trainable_tces):
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=False)
                else:
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=True)
                """

                self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=True)
                
                # Adapt classifier head based on combination method
                # 'concat' method is dimension_size + num_classes
                if (self.comb_method == 'cat'):
                    combined_size = self.hidden_size + self.tce_matrix.size(1)
                else:
                    combined_size = self.hidden_size                                # redundant but for clarity - good for 'add' or 'dot'

                """
                #
                # initialize the TCE Embedding layer, freeze the embeddings if trainable_tces == False
                # otherwise let the model train the tce embedding layer
                #
                if (finetune):
                    print("finetuning, retraining tce embedding layer...")
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=False)
                else:
                    print("not finetuning, not retraining tce embedding layer...")
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=True)
                """

        # -----------------------------------------------------------------------------------------------
        # 
        # initialize Classification head: maps the (potentially) combined input (transformer output or transformer output + optional TCE embeddings) 
        # to the final logits, introducing additional learnable parameters and allowing for flexibility to adapt the model to the specific task.
        #

        print("combined_size:", combined_size)

        # -----------------------------------------------------------------------------------------------
        # define classifier
        #
        if simple:
            # simplified classification head that adjusts the size of the Linear layer according  
            # to the method we are using to combine TCEs with built in transformer embeddings
            self.classifier = nn.Linear(combined_size, self.num_classes)                                  
        else:    
            self.classifier = nn.Sequential(
                nn.Linear(combined_size, 256),                  # First linear layer
                nn.ReLU(),                                      # non-linear activation function
                nn.Dropout(dropout_rate),                       # regularization
                nn.Linear(256, self.num_classes)                # FInal Linear layer
            )
        print("self.classifier:", self.classifier)
        # -----------------------------------------------------------------------------------------------

        #
        # force all of the tensors to be stored contiguously in memory
        #
        for param in self.transformer.parameters():
            param.data = param.data.contiguous()
    

    def _filter_transformer_embeddings(self, hf_model, relevant_tokens):
        """
        Replace the transformer embedding layer with a reduced embedding matrix and update the tokenizer.
        """
        print("Filtering transformer embeddings to align with dataset...")
        print(f"relevant_tokens: {type(relevant_tokens)} {len(relevant_tokens)}")

        # Extract original embeddings
        embedding_layer = hf_model.get_input_embeddings()
        original_embeddings = embedding_layer.weight.data

        # Validate relevant_tokens
        assert all(0 <= idx < original_embeddings.size(0) for idx in relevant_tokens), \
            "Relevant tokens contain indices out of range of the original embeddings."

        # Get embeddings for relevant tokens
        relevant_embeddings = original_embeddings[relevant_tokens, :]
        print("relevant_embeddings:", type(relevant_embeddings), relevant_embeddings.shape)

        # Replace the embedding layer
        reduced_embedding_layer = nn.Embedding.from_pretrained(relevant_embeddings)
        print("reduced_embedding_layer:", type(reduced_embedding_layer), reduced_embedding_layer)

        hf_model.set_input_embeddings(reduced_embedding_layer)
        print("updated hf_model:", hf_model)

        return hf_model, relevant_embeddings
    


    def _normalize_tce(self, tce_matrix, embedding_mean, embedding_std):
        """
        Normalize the TCE matrix to align with the transformer's embedding distribution.

        Args:
            tce_matrix: TCE matrix (vocab_size x num_classes).
            embedding_mean: Mean of the transformer embeddings (1D tensor, size=model_dim).
            embedding_std: Standard deviation of the transformer embeddings (1D tensor, size=model_dim).

        Returns:
            Normalized TCE matrix (vocab_size x num_classes).
        """

        target_dim = embedding_mean.shape[0]  # Set target_dim to model embedding size

        if (self.debug):
            print(f"tce_matrix: {tce_matrix.shape}, {tce_matrix.dtype}")
            #print("first row:", tce_matrix[0])
            print(f"embedding_mean: {embedding_mean.shape}, {embedding_mean.dtype}")
            #print(f'embedding_mean: {embedding_mean}')
            print(f"embedding_std: {embedding_std.shape}, {embedding_std.dtype}")
            #print(f'embedding_std: {embedding_std}')
            print("target_dim:", target_dim)

        device = embedding_mean.device                      # Ensure all tensors are on the same device
        tce_matrix = tce_matrix.to(device)

        # 1 Normalize TCE matrix row-wise (i.e. ompute mean and std per row)
        tce_mean = tce_matrix.mean(dim=1, keepdim=True)
        tce_std = tce_matrix.std(dim=1, keepdim=True)
        tce_std[tce_std == 0] = 1                           # Prevent division by zero

        if (self.debug):
            print(f"tce_mean: {tce_mean.shape}, {tce_mean.dtype}")
            #print(f'tce_mean: {tce_mean}')
            print(f"tce_std: {tce_std.shape}, {tce_std.dtype}")
            #print(f'tce_std: {tce_std}')

        normalized_tce = (tce_matrix - tce_mean) / tce_std

        if (self.debug):
            print(f"normalized_tce (pre-scaling): {normalized_tce.shape}")

        # 2. Scale to match embedding statistics
        normalized_tce = normalized_tce * embedding_std.mean() + embedding_mean.mean()

        # 3. Project normalized TCE into the target dimension (e.g., 128)
        projection = torch.nn.Linear(tce_matrix.size(1), target_dim, bias=False).to(device)
        projected_tce = projection(normalized_tce)

        if self.debug:
            print(f"Projected TCE matrix: {projected_tce.shape}")

        # check for Nan or Inf values after normalization
        if torch.isnan(projected_tce).any() or torch.isinf(projected_tce).any():
            print("[WARNING]: projected_tce contains NaN or Inf values after normalization.")
            #raise ValueError("[ERROR] projected_tce contains NaN or Inf values after normalization.")

        return projected_tce


    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for the LCSequenceClassifier, includes support for integrated
        TCE computation in the event that the Classifier has been set up with TCEs (ie supervised is True)

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size x seq_length).
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor, optional): Labels for computing loss.

        Returns:
            logits (torch.Tensor): Output logits from the classifier.
            loss (torch.Tensor, optional): Loss value if labels are provided.
        """
        if (self.debug):
            print("LCSequenceClassifier:forward()...")
            print(f"\tinput_ids: {type(input_ids)}, {input_ids.shape}")
            print("input_ids:", input_ids)
            print(f"\tattention_mask: {type(attention_mask)}, {attention_mask.shape}")
            print(f"\tlabels: {type(labels)}, {labels.shape}")

        assert input_ids.max() < self.vocab_size, f"Invalid token index: {input_ids.max()} >= {self.vocab_size}"
        assert input_ids.min() >= 0, f"Invalid token index: {input_ids.min()} < 0"

        # Pass inputs through the transformer model, ie Base model forward pass
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]  # Use hidden states if available
        else:
            raise AttributeError("Transformer model did not output hidden states. Ensure output_hidden_states=True is set.")
        if (self.debug):
            print(f"last_hidden_state: {type(last_hidden_state)}, {last_hidden_state.shape}")

        # default 
        combined_output = last_hidden_state     # USe only the transformer outputs

        #
        # Integrate TCEs if supervised is True
        #
        #if (self.supervised and (self.tce_matrix is not None)):
        if self.supervised:
            
            if self.tce_layer is None:
                raise ValueError("[ERROR]:supervised is True but tce_layer embedding layer is None.")
            
            if (self.debug):
                print("integrating TCEs into the model...")

            # Debug info: Check for out-of-range indices
            invalid_indices = input_ids[input_ids >= self.vocab_size]
            if invalid_indices.numel() > 0:
                print(f"[WARNING] Found invalid indices in input_ids: {invalid_indices.tolist()} (max valid index: {self.vocab_size - 1})")

            # Extract all relevant indices for pooling TCE embeddings
            tce_indices = input_ids                                     # Assuming all input tokens are relevant for TCEs
            tce_embeddings = self.tce_layer(tce_indices)                # (batch_size, seq_length, tce_dim)

            # Combine transformer outputs with TCE embeddings
            if self.comb_method == 'cat':
                combined_output = torch.cat((last_hidden_state, tce_embeddings), dim=-1)  # (batch_size, seq_length, hidden_size + tce_dim)
            elif self.comb_method == 'add':
                combined_output = last_hidden_state + tce_embeddings  # Element-wise addition
            elif self.comb_method == 'dot':
                combined_output = last_hidden_state * tce_embeddings  # Element-wise multiplication
            else:
                raise ValueError(f"Unsupported combination method: {self.comb_method}")

        if (self.debug):
            print(f"combined_output: {type(combined_output)}, {combined_output.shape}")

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(combined_output.size())
            masked_output = combined_output * mask_expanded
            sum_hidden_state = torch.sum(masked_output, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)  # Avoid division by zero
            pooled_output = sum_hidden_state / sum_mask
        else:
            pooled_output = combined_output.mean(dim=1)  # Fallback to mean pooling

        if self.debug:
            print(f"pooled_output: {type(pooled_output)}, {pooled_output.shape}")


        # Pool across the sequence dimension to reduce to (batch_size, combined_size)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(combined_output.size())
            masked_output = combined_output * mask_expanded
            sum_hidden_state = torch.sum(masked_output, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            pooled_output = sum_hidden_state / sum_mask.clamp(min=1e-9)  # Avoid division by zero
        else:
            pooled_output = combined_output.mean(dim=1)  # Simple mean pooling when no attention mask is provided

        if self.debug:
            print(f"pooled_output: {type(pooled_output)}, {pooled_output.shape}")

        # Classification head
        logits = self.classifier(pooled_output)
        if (self.debug):
            print(f'logits: {type(logits)}, {logits.shape}, {logits}')

        loss = None
        if labels is not None:
            if self.class_type in ['multi-label', 'multilabel']:
                # BCEWithLogitsLoss requires float labels for multi-label classification
                loss = self.loss_fn(logits, labels.float())
            elif self.class_type in ['single-label', 'singlelabel']:
                # CrossEntropyLoss expects long/int labels for single-label classification
                loss = self.loss_fn(logits, labels.long())
            else:
                raise ValueError(f"Unsupported classification type: {self.class_type}")
        else:
            print("WARNINMG: No labels provided for loss calculation.")

        return {"loss": loss, "logits": logits}
    

    def get_embedding_dims(self):
        """
        Retrieve the dimensions of the embedding layer.

        Returns:
            Tuple[int, int]: A tuple containing the vocabulary size and embedding dimension.
        """
        # Identify the embedding layer dynamically
        if hasattr(self.transformer, 'bert'):
            embedding_layer = self.transformer.bert.embeddings.word_embeddings
        elif hasattr(self.transformer, 'roberta'):
            embedding_layer = self.transformer.roberta.embeddings.word_embeddings
        elif hasattr(self.transformer, 'transformer'):  # GPT-2
            embedding_layer = self.transformer.transformer.wte
        elif hasattr(self.transformer, 'model') and hasattr(self.transformer.model, 'embed_tokens'):  # LLaMA
            embedding_layer = self.transformer.model.embed_tokens
        else:
            raise ValueError("Unsupported model type or embedding layer not found.")

        # Extract dimensions
        vocab_size, embedding_dim = embedding_layer.weight.size()

        if self.debug:
            print(f"Embedding layer dimensions: vocab_size={vocab_size}, embedding_dim={embedding_dim}")

        return vocab_size, embedding_dim




class LCSequenceClassifier_orig(nn.Module):

    def __init__(self, 
                hf_model: nn.Module, 
                num_classes: int, 
                vocab_size: int, 
                class_type: str = 'single-label', 
                class_weights: torch.Tensor = None, 
                supervised: bool = False, 
                tce_matrix: torch.Tensor = None, 
                finetune: bool = False, 
                normalize_tces: bool = True,
                dropout_rate: float = 0.3, 
                comb_method: str = "cat", 
                debug: bool = False):
        """
        A Transformer-based Sequence Classifier with optional TCE integration and parameters for Layer Cake Text
        Classification testing. Supports both single label and multi-label classification.
        
        Args:
            hf_model: The HuggingFace pre-trained transformer model (e.g., BERT), preloaded.
            num_classes: Number of classes for classification.
            vocab_size: size of (tokenizer) vocabulary, for assertions
            class_type: type of classification problem, either 'single-label' or 'multi-label'
            class_weights: Class weights for loss function.
            supervised: Boolean indicating if supervised embeddings are used.
            tce_matrix: Precomputed TCE matrix (Tensor) with shape [vocab_size, num_classes].
            finetune: Boolean indicating whether or not to make the model embedding layer, and the TCE matrix, trainable.
            normalize_tce: Boolean indicating if TCE matrix is normalized.
            dropout: Dropout rate for TCE matrix.
            comb-method: Method to integrate WCE embeddings ("add", "dot" or "cat").            
            debug: Debug mode flag.
        """
        super(LCSequenceClassifier, self).__init__()

        print(f'LCSequenceClassifier:__init__()... class_type: {class_type}, num_classes: {num_classes}, finetune: {finetune}, supervised: {supervised}, debug: {debug}')

        if (supervised):
            print(f'normalize_tces: {normalize_tces}, dropout_rate: {dropout_rate}, comb_method: {comb_method}')

        self.debug = debug

        self.transformer = hf_model
        #transformer_output_dim = self.transformer.config.hidden_size  # e.g., 768

        self.hidden_size = self.transformer.config.hidden_size          
        print("self.hidden_size:", self.hidden_size)

        self.num_classes = num_classes
        self.class_type = class_type

        self.supervised = supervised
        self.comb_method = comb_method
        self.normalize_tces = normalize_tces
        self.finetune = finetune
        self.class_weights = class_weights

        if (self.class_weights is not None):
            print("self.class_weights.shape:", self.class_weights.shape)
            if (self.debug):
                print("self.class_weights:", self.class_weights)

            # Loss functions
            if class_type == 'multi-label':
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            elif class_type == 'single-label':
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                raise ValueError("class_type must be 'single-label' or 'multi-label'")
            print("loss_fn:", self.loss_fn)

        else:
            print("self.class_weights is None")

            # Loss functions
            if class_type == 'multi-label':
                self.loss_fn = nn.BCEWithLogitsLoss()
            elif class_type == 'single-label':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                raise ValueError("class_type must be 'single-label' or 'multi-label'")
            print("loss_fn:", self.loss_fn)
        

        self.vocab_size = vocab_size
        print("self.vocab_size:", self.vocab_size)

        #
        # Optionally unfreeze the embedding layer if finetune is True
        #
        if self.finetune:
            print("finetuning == True, making the embedding layer (only) trainable")    

            # Freeze gradient computation for all transformer parameters
            for param in self.transformer.parameters():
                param.requires_grad = False

            # Enable training of only the embedding layer
            if hasattr(self.transformer, 'bert'):
                embedding_layer = self.transformer.bert.embeddings
            elif hasattr(self.transformer, 'roberta'):
                embedding_layer = self.transformer.roberta.embeddings
            elif hasattr(self.transformer, 'transformer'):
                embedding_layer = self.transformer.transformer.wte  # GPT-2
            else:
                raise AttributeError(f"Embeddings not found for the given model: {type(self.transformer)}")
            print("embedding_layer:", type(embedding_layer), embedding_layer)

            for param in embedding_layer.parameters():
                param.requires_grad = True
        else:            
            print("finetune == False, default model configuration ...")

        # Assert that tce_matrix is provided if supervised is True
        if self.supervised:
            assert tce_matrix is not None, "tce_matrix must be provided when supervised is True."

        self.tce_matrix = tce_matrix
        self.tce_layer = None               # we initialize this only if we are using TCEs 

        # initialize embedding dimensions to model embedding dimension
        # we over-write this only if we are using supervised tces with the 'cat' method
        combined_size = self.hidden_size

        if (self.supervised and self.tce_matrix is not None):

            print("supervised is True, original tce_matrix:", type(self.tce_matrix), self.tce_matrix.shape)
 
            with torch.no_grad():                           # normalization code should be in this block

                # Normalize TCE matrix if required
                if self.normalize_tces:

                    # compute the mean and std from the core model embeddings
                    embedding_layer = self.transformer.get_input_embeddings()
                    embedding_mean = embedding_layer.weight.mean(dim=0).to(device)
                    embedding_std = embedding_layer.weight.std(dim=0).to(device)
                    if (self.debug):
                        print(f"transformer embeddings mean: {embedding_mean.shape}, std: {embedding_std.shape}")

                    # normalize the TCE matrix
                    self.tce_matrix = self._normalize_tce(self.tce_matrix, embedding_mean, embedding_std)
                    print(f"Normalized TCE matrix: {type(self.tce_matrix)}, {self.tce_matrix.shape}")
            
                # initialize the TCE Embedding layer, freeze the embeddings if trainable_tces == False
                """
                if (trainable_tces):
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=False)
                else:
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=True)
                """

                self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=True)
                
                # Adapt classifier head based on combination method
                # 'concat' method is dimension_size + num_classes
                if (self.comb_method == 'cat'):
                    combined_size = self.hidden_size + self.tce_matrix.size(1)
                else:
                    combined_size = self.hidden_size                                # redundant but for clarity - good for 'add' or 'dot'

                """
                #
                # initialize the TCE Embedding layer, freeze the embeddings if trainable_tces == False
                # otherwise let the model train the tce embedding layer
                #
                if (finetune):
                    print("finetuning, retraining tce embedding layer...")
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=False)
                else:
                    print("not finetuning, not retraining tce embedding layer...")
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=True)
                """

        # -----------------------------------------------------------------------------------------------
        # 
        # initialize Classification head: maps the (potentially) combined input (transformer output or transformer output + optional TCE embeddings) 
        # to the final logits, introducing additional learnable parameters and allowing for flexibility to adapt the model to the specific task.
        #

        print("combined_size:", combined_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 256),                  # First linear layer
            nn.ReLU(),                                      # non-linear activation function
            nn.Dropout(dropout_rate),                       # regularization
            nn.Linear(256, self.num_classes)                # FInal Linear layer
        )
        
        print("self.classifier:", self.classifier)
        # -----------------------------------------------------------------------------------------------

        #
        # force all of the tensors to be stored contiguously in memory
        #
        for param in self.transformer.parameters():
            param.data = param.data.contiguous()
    
    
    def _normalize_tce(self, tce_matrix, embedding_mean, embedding_std):
        """
        Normalize the TCE matrix to align with the transformer's embedding distribution.

        Args:
            tce_matrix: TCE matrix (vocab_size x num_classes).
            embedding_mean: Mean of the transformer embeddings (1D tensor, size=model_dim).
            embedding_std: Standard deviation of the transformer embeddings (1D tensor, size=model_dim).

        Returns:
            Normalized TCE matrix (vocab_size x num_classes).
        """

        target_dim = embedding_mean.shape[0]  # Set target_dim to model embedding size

        if (self.debug):
            print(f"tce_matrix: {tce_matrix.shape}, {tce_matrix.dtype}")
            #print("first row:", tce_matrix[0])
            print(f"embedding_mean: {embedding_mean.shape}, {embedding_mean.dtype}")
            #print(f'embedding_mean: {embedding_mean}')
            print(f"embedding_std: {embedding_std.shape}, {embedding_std.dtype}")
            #print(f'embedding_std: {embedding_std}')
            print("target_dim:", target_dim)

        device = embedding_mean.device                      # Ensure all tensors are on the same device
        tce_matrix = tce_matrix.to(device)

        # 1 Normalize TCE matrix row-wise (i.e. ompute mean and std per row)
        tce_mean = tce_matrix.mean(dim=1, keepdim=True)
        tce_std = tce_matrix.std(dim=1, keepdim=True)
        tce_std[tce_std == 0] = 1                           # Prevent division by zero

        if (self.debug):
            print(f"tce_mean: {tce_mean.shape}, {tce_mean.dtype}")
            #print(f'tce_mean: {tce_mean}')
            print(f"tce_std: {tce_std.shape}, {tce_std.dtype}")
            #print(f'tce_std: {tce_std}')

        normalized_tce = (tce_matrix - tce_mean) / tce_std

        if (self.debug):
            print(f"normalized_tce (pre-scaling): {normalized_tce.shape}")

        # 2. Scale to match embedding statistics
        normalized_tce = normalized_tce * embedding_std.mean() + embedding_mean.mean()

        # 3. Project normalized TCE into the target dimension (e.g., 128)
        projection = torch.nn.Linear(tce_matrix.size(1), target_dim, bias=False).to(device)
        projected_tce = projection(normalized_tce)

        if self.debug:
            print(f"Projected TCE matrix: {projected_tce.shape}")

        # check for Nan or Inf values after normalization
        if torch.isnan(projected_tce).any() or torch.isinf(projected_tce).any():
            print("[WARNING]: projected_tce contains NaN or Inf values after normalization.")
            #raise ValueError("[ERROR] projected_tce contains NaN or Inf values after normalization.")

        return projected_tce


    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for the LCSequenceClassifier, includes support for integrated
        TCE computation in the event that the Classifier has been set up with TCEs (ie supervised is True)

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size x seq_length).
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor, optional): Labels for computing loss.

        Returns:
            logits (torch.Tensor): Output logits from the classifier.
            loss (torch.Tensor, optional): Loss value if labels are provided.
        """
        if (self.debug):
            print("LCSequenceClassifier:forward()...")
            print(f"\tinput_ids: {type(input_ids)}, {input_ids.shape}")
            print("input_ids:", input_ids)
            print(f"\tattention_mask: {type(attention_mask)}, {attention_mask.shape}")
            print(f"\tlabels: {type(labels)}, {labels.shape}")

        assert input_ids.max() < self.vocab_size, f"Invalid token index: {input_ids.max()} >= {self.vocab_size}"
        assert input_ids.min() >= 0, f"Invalid token index: {input_ids.min()} < 0"

        # Pass inputs through the transformer model, ie Base model forward pass
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        if (USE_MEAN_POOLING):
            # Extract the last hidden state (batch_size, seq_length, hidden_size)
            last_hidden_state = outputs.hidden_states[-1]

            # Apply mean pooling across the sequence dimension
            # Mask the padded tokens during mean pooling to avoid including their embeddings
            if attention_mask is not None:
                # Expand attention mask for the hidden size dimension
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                masked_hidden_state = last_hidden_state * mask_expanded
                sum_hidden_state = torch.sum(masked_hidden_state, dim=1)
                sum_mask = torch.sum(mask_expanded, dim=1)
                pooled_output = sum_hidden_state / sum_mask.clamp(min=1e-9)  # Avoid division by zero
            else:
                # Simple mean pooling when no attention mask is provided
                pooled_output = torch.mean(last_hidden_state, dim=1)
        else:
            #
            # retrieve the pooled output representation depending upon the underlying SequenceClassifier model
            # Use pooled output if available, if not and we are using a BERT based model, we get the
            # CLS token, ie the first token, otherwise (GPT2 or LlaMa) we use the last token
            #
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output       
            elif isinstance(self.transformer, (BertForSequenceClassification, 
                                            RobertaForSequenceClassification, 
                                            DistilBertForSequenceClassification)):
                pooled_output = outputs.hidden_states[-1][:, 0]                                                             # Use CLS token embedding
            elif isinstance(self.transformer, (GPT2ForSequenceClassification)):
                pooled_output = outputs.hidden_states[-1][:, -1]                                                            # Use last token embedding
            elif isinstance(self.transformer, XLNetForSequenceClassification):
                # XLNet-specific pooling logic
                #pooled_output = outputs.hidden_states[-1][:, 0, :]  # Use the first token representation

                # Use mean pooling over the last hidden layer
                last_hidden_state = outputs.hidden_states[-1]                       # Extract the last layer's hidden states
                pooled_output = torch.mean(last_hidden_state, dim=1)                # Mean pooling across the sequence dimension
            else:
                raise ValueError("Unsupported model type for pooling. Please implement pooling logic for this transformer.")

        if (self.debug):    
            print(f'pooled_output (pre combination): {type(pooled_output)}, {pooled_output.shape}')

        if torch.isnan(pooled_output).any() or torch.isinf(pooled_output).any():
            #print("[ERROR] pooled_output contains NaN or Inf values")
            raise ValueError("[ERROR] pooled_output contains NaN or Inf values")

        #
        # Integrate TCEs if supervised is True
        #
        #if (self.supervised and (self.tce_matrix is not None)):
        if self.supervised:
            
            if self.tce_layer is None:
                raise ValueError("[ERROR]:supervised is True but tce_layer embedding layer is None.")
            
            if (self.debug):
                print("integrating TCEs into the model...")

            # Debug info: Check for out-of-range indices
            invalid_indices = input_ids[input_ids >= self.vocab_size]
            if invalid_indices.numel() > 0:
                print(f"[WARNING] Found invalid indices in input_ids: {invalid_indices.tolist()} (max valid index: {self.vocab_size - 1})")

            # Extract all relevant indices for pooling TCE embeddings
            tce_indices = input_ids                                     # Assuming all input tokens are relevant for TCEs
            tce_embeddings = self.tce_layer(tce_indices)                # (batch_size, seq_length, tce_dim)

            # Apply pooling to TCE embeddings to obtain a single vector per example
            pooled_tce_embeddings = tce_embeddings.mean(dim=1)                                  # mean pooling, output (batch_size, tce_dim)
            #pooled_tce_embeddings, _ = tce_embeddings.max(dim=1)                               # max pooling, output (batch_size, tce_dim)
            
            if (self.debug):
                print("pooled_tce_embeddings:", type(pooled_tce_embeddings), pooled_tce_embeddings.shape)
                print("pooled_tce_embeddings[0]:", type(pooled_tce_embeddings[0]), pooled_tce_embeddings[0])
            
            if torch.isnan(pooled_tce_embeddings).any() or torch.isinf(pooled_tce_embeddings).any():
                #print("[ERROR] pooled_tce_embeddings contains NaN or Inf values")
                raise ValueError("[ERROR] pooled_tce_embeddings contains NaN or Inf values")

            # Combine transformer output and TCE embeddings
            if self.comb_method == 'cat':
                if (self.debug):
                    print("concatenating two matrices...")                                                
                assert pooled_output.size(0) == pooled_tce_embeddings.size(0), "Batch size mismatch between pooled output and pooled input TCE matrix"
                assert pooled_output.size(1) + pooled_tce_embeddings.size(1) == self.hidden_size + self.tce_matrix.size(1), \
                    f"Concat dimension mismatch: {pooled_output.size(1) + pooled_tce_embeddings.size(1)} != {self.hidden_size + self.tce_matrix.size(1)}"
                pooled_output = torch.cat((pooled_output, pooled_tce_embeddings), dim=1)        # (batch_size, combined_size)
            elif self.comb_method == 'add':
                if (self.debug):
                    print("adding two matricses...")
                assert pooled_output.size(0) == pooled_tce_embeddings.size(0), "Batch size mismatch between pooled output and pooled input TCE matrix"
                assert pooled_output.size(1) == pooled_tce_embeddings.size(1), \
                    f"Add dimension mismatch: {pooled_output.size(1)} != {pooled_tce_embeddings.size(1)}"
                pooled_output = pooled_output + pooled_tce_embeddings                               # (batch_size, hidden_size) 
            elif self.comb_method == 'dot':
                if self.debug:
                    print("computing dot product between embeddings...")
                assert pooled_output.size(0) == pooled_tce_embeddings.size(0), \
                    "Batch size mismatch between pooled output and pooled TCE matrix"
                assert pooled_output.size(1) == pooled_tce_embeddings.size(1), \
                    f"Dot dimension mismatch: {pooled_output.size(1)} != {pooled_tce_embeddings.size(1)}"
                pooled_output = (pooled_output * pooled_tce_embeddings)                             # Shape (batch_size, hidden_size)
            else:
                raise ValueError(f"Unsupported combination method: {self.comb_method}")    
            
        if (self.debug):    
            print(f'pooled_output (post combination): {type(pooled_output)}, {pooled_output.shape}')

        # Classification head
        logits = self.classifier(pooled_output)
        if (self.debug):
            print(f'logits: {type(logits)}, {logits.shape}, {logits}')

        loss = None
        if labels is not None:
            if self.class_type in ['multi-label', 'multilabel']:
                # BCEWithLogitsLoss requires float labels for multi-label classification
                loss = self.loss_fn(logits, labels.float())
            elif self.class_type in ['single-label', 'singlelabel']:
                # CrossEntropyLoss expects long/int labels for single-label classification
                loss = self.loss_fn(logits, labels.long())
            else:
                raise ValueError(f"Unsupported classification type: {self.class_type}")
        else:
            print("WARNINMG: No labels provided for loss calculation.")

        return {"loss": loss, "logits": logits}
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


def lc_class_weights(labels, task_type="single-label"):
    """
    Compute class weights for single-label or multi-label classification.

    Args:
        labels: List or numpy array.
                - Single-label: List of class indices (e.g., [0, 1, 2]).
                - Multi-label: Binary array of shape (num_samples, num_classes).
        task_type: "single-label" or "multi-label".

    Returns:
        Torch tensor of class weights.
    """

    print(f'Computing class weights for {task_type} task...')
    #print("labels:", labels)

    if task_type == "single-label":
        # Compute class weights using sklearn
        num_classes = len(np.unique(labels))
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float)

    elif task_type == "multi-label":
        # Compute pos_weights for BCEWithLogitsLoss
        labels = torch.tensor(labels, dtype=torch.float)
        num_samples = labels.shape[0]
        pos_counts = labels.sum(dim=0)  # Number of positive samples per class
        neg_counts = num_samples - pos_counts  # Number of negative samples per class

        pos_counts = torch.clamp(pos_counts, min=1.0)  # Avoid division by zero
        pos_weights = neg_counts / pos_counts
        return pos_weights

    else:
        raise ValueError("Invalid task_type. Use 'single-label' or 'multi-label'.")



# Get the full model identifier and load from local directory
def get_model_identifier(pretrained, cache_dir="../.vector_cache"):

    model_name = MODEL_MAP.get(pretrained, pretrained)
    model_path = os.path.join(cache_dir, pretrained)

    return model_name, model_path
    


class LCDataset(Dataset):
    """
    Dataset class for handling both input text and labels for LCHFTCEClassifier

    Parameters:
    - texts: List of input text samples.
    - labels: Multi-label binary vectors or single-label indices.
    - tokenizer: Hugging Face tokenizer for text tokenization.
    - max_length: Maximum length of tokenized sequences.
    - class_type: 'multi-label' or 'single-label' classification type.
    """

    def __init__(self, texts, labels, tokenizer, max_length=MAX_TOKEN_LENGTH, class_type='multi-label'):
        
        print(f'LCDataset() class_type: {class_type}, len(texts): {len(texts)}, len(labels): {len(labels)}, max_length: {max_length}')

        assert len(texts) == len(labels), "Mismatch between texts and labels lengths."

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_type = class_type


    def __len__(self):
        return len(self.texts)

        
    def shape(self):
        """
        Returns the shape of the dataset as a tuple: (number of texts, number of labels).
        """
        return len(self.texts), len(self.labels) if self.labels is not None else 0
    

    def __getitem__(self, idx):
        """
        Get an individual sample from the dataset.

        Returns:
        - item: Dictionary containing tokenized inputs, labels, and optionally supervised embeddings.
        """
        text = self.texts[idx]
        labels = self.labels[idx] if self.labels is not None else 0  # Default label is 0 if missing

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}  # Remove batch dim

        # Add labels
        if self.class_type == 'single-label':
            item["labels"] = torch.tensor(labels, dtype=torch.long)
        else:
            item["labels"] = torch.tensor(labels, dtype=torch.float)

        return item



def custom_data_collator(batch):
    """
    Custom data collator for handling variable-length sequences and TCEs.

    Parameters:
    - batch: List of individual samples from the dataset.

    Returns:
    - collated: Dictionary containing collated inputs, labels, and optionally TCEs.
    """
    collated = {
        "input_ids": torch.stack([f["input_ids"] for f in batch]),
        "attention_mask": torch.stack([f["attention_mask"] for f in batch]),
    }

    if "labels" in batch[0]:
        collated["labels"] = torch.stack([f["labels"] for f in batch])

    return collated



def compute_metrics(eval_pred, class_type='single-label', threshold=0.5):
    """
    Compute evaluation metrics for classification tasks.

    Args:
    - eval_pred: `EvalPrediction` object with `predictions` and `label_ids`.
    - class_type: 'single-label' or 'multi-label'.
    - threshold: Threshold for binary classification in multi-label tasks.

    Returns:
    - Dictionary of computed metrics.
    """
    """
    print(f'compute_metrics()... class_type: {class_type}')
    
    if (class_type in ['multi-label', 'multilabel']):
        print(f'threshold: {threshold}')
    """

    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    if class_type == 'single-label':
        # Convert predictions to class indices
        preds = np.argmax(predictions, axis=1)                  
    elif class_type == 'multi-label':
        # Threshold predictions for multi-label classification
        preds = (predictions > threshold).astype(int)
        labels = labels.astype(int)                                     # Ensure labels are binary
    else:
        raise ValueError(f"Unsupported class_type: {class_type}")

    """
    print("preds:", type(preds), preds.shape)
    print("labels:", type(labels), labels.shape)
    """
    
    # Compute metrics
    f1_micro = f1_score(labels, preds, average='micro', zero_division=1)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=1)
    precision = precision_score(labels, preds, average='micro', zero_division=1)
    recall = recall_score(labels, preds, average='micro', zero_division=1)

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
    }


def get_hf_models(model_name, model_path, num_classes, tokenizer):
    """
    Load Hugging Face transformer models with configurations to enable hidden states.
    Ensure pad_token_id is consistently fetched from the tokenizer.

    Parameters:
    - model_name: Name of the Hugging Face model.
    - model_path: Path to the Hugging Face model.
    - num_classes: Number of classes for classification.
    - tokenizer: Hugging Face tokenizer.

    Returns:
    - hf_trans_model: Hugging Face transformer model.
    - hf_trans_class_model: Hugging Face transformer model for classification.
    """

    print(f'get_hf_models(): model_name: {model_name}, model_path: {model_path}, num_classes: {num_classes}')

    # Ensure the tokenizer has a pad_token_id set
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer does not have a valid pad_token_id. Ensure the tokenizer is properly configured before calling this function.")

    pad_token_id = tokenizer.pad_token_id
    print(f"Using pad_token_id: {pad_token_id} (token: {tokenizer.pad_token})")

    # Initialize Hugging Face Transformer model
    hf_trans_model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
    
    # Ensure hidden states are enabled for the base model
    hf_trans_model.config.output_hidden_states = True

    # Ensure the model config uses the tokenizer's pad_token_id
    if hf_trans_model.config.pad_token_id is None:
        print("hf_trans_model padding token ID is None, setting to tokenizer.pad_token_id...")
        hf_trans_model.config.pad_token_id = pad_token_id

    # Initialize Hugging Face Transformer model for sequence classification
    hf_trans_class_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=model_path,
        num_labels=num_classes,      # Specify the number of classes for classification
        pad_token_id=pad_token_id    # Use tokenizer's pad_token_id
    )

    # Ensure hidden states are enabled for the classification model
    hf_trans_class_model.config.output_hidden_states = True

    # Ensure the model config uses the tokenizer's pad_token_id
    if hf_trans_class_model.config.pad_token_id is None:
        print("hf_trans_class_model padding token ID is None, setting to tokenizer.pad_token_id...")
        hf_trans_class_model.config.pad_token_id = pad_token_id

    return hf_trans_model, hf_trans_class_model







def compute_dataset_vocab(train_texts, val_texts, test_texts, tokenizer):
    """
    Compute the dataset vocabulary as token IDs that align with the tokenizer's vocabulary.
    
    Args:
        train_texts (list of str): Training set texts.
        val_texts (list of str): Validation set texts.
        test_texts (list of str): Test set texts.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for the transformer model.
        
    Returns:
        list: Relevant token IDs for the dataset vocabulary.
    """
    print("compute_dataset_vocab()...")

    # Combine all texts from training, validation, and test sets
    all_texts = train_texts + val_texts + test_texts

    # Use a set to store unique tokens from the dataset
    dataset_vocab_tokens = set()

    # Tokenize each document and add the tokens to the dataset vocabulary
    for text in all_texts:
        tokens = tokenizer.tokenize(text)  # Tokenize the text using the tokenizer
        dataset_vocab_tokens.update(tokens)  # Add tokens to the dataset vocabulary

    # Convert dataset tokens to their respective token IDs using the tokenizer's vocabulary
    relevant_tokens = [
        tokenizer.vocab[token]
        for token in dataset_vocab_tokens
        if token in tokenizer.vocab
    ]

    print(f"Computed dataset vocabulary: {len(relevant_tokens)} relevant tokens out of {len(tokenizer.vocab)} total tokens in tokenizer.")
    return relevant_tokens





# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Text Classification with Transformer Models.")
    
    # system params
    parser.add_argument('--dataset', required=True, type=str, choices=SUPPORTED_DATASETS, help='Dataset to use')
    parser.add_argument('--show-dist', action='store_true', default=True, help='Show dataset class distribution')
    #parser.add_argument('--dist', action='store_true', default=False, help='show class distribution plots')
    parser.add_argument('--vtype', type=str, default='tfidf', metavar='N', 
                        help=f'dataset base vectorization strategy, in [tfidf, count]')
    parser.add_argument('--pretrained', type=str, choices=['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2'], help='supported language model types for dataset representation')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--log-file', type=str, default='../log/trans_lc_nn_test.test', help='Path to log file')
    parser.add_argument('--force', action='store_true', default=False, help='do not check if this experiment has already been run')
    parser.add_argument('--simple', action='store_true', default=True, help='Use the simple classifier (just the one Linear layer after HF SeuqenceClassifier)')
    
    # model params
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=PATIENCE, help='Patience for early stopping')
    parser.add_argument('--dropprob', type=float, default=0.3, metavar='[0.0, 1.0]', help='dropout probability for TCE Embedding layer classifier head (default: 0.3)')
    parser.add_argument('--net', type=str, default='hf.sc.ff', metavar='str', help=f'net, defaults to hf.sc (only supported option)')
    parser.add_argument('--learnable', type=int, default=0, metavar='int', help='dimension of the learnable embeddings (default 0)')
    parser.add_argument('--droptype', type=str, default='sup', metavar='DROPTYPE',
                        help=f'chooses the type of dropout to apply after the embedding layer. Default is "sup" which '
                             f'only applies to word-class embeddings (if present). Other options include "none" which '
                             f'does not apply dropout (same as "sup" with no supervised embeddings), "full" which '
                             f'applies dropout to the entire embedding, or "learn" that applies dropout only to the '
                             f'learnable embedding.')
    parser.add_argument('--tunable', action='store_true', default=False,
                        help='pretrained embeddings are tunable from the beginning (default False, i.e., static)')
    
    # TCE params
    parser.add_argument('--supervised', action='store_true', help='Use supervised embeddings (TCEs')
    parser.add_argument('--sup-mode', type=str, default='cat', help='How to combine TCEs with model embeddings (add, dot, cat)')
    parser.add_argument('--nozscore', action='store_true', default=False,
                        help='disables z-scoring form the computation of TCE')
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
    parser.add_argument('--max-label-space', type=int, default=300, metavar='int',
                        help='larger dimension allowed for the feature-label embedding (if larger, then PCA with this '
                             'number of components is applied (default 300)')

    return parser.parse_args()



# Main
if __name__ == "__main__":

    program = 'trans_layer_cake'
    version = '11.0'
    print(f'program: {program}, version: {version}')
    
    print(f'\n\t--- TRANS_LAYER_CAKE Version: {version} ---')
    print()

    args = parse_args()
    print("args:", args)
    
    # Get the full model identifier and cache directory path for tokenizer/model
    model_name, model_path = get_model_identifier(args.pretrained)
    print("model_name:", model_name)
    print("model_path:", model_path)

    if (args.pretrained == 'bert'):
        args.bert_path = model_path
    elif (args.pretrained == 'roberta'):
        args.roberta_path = model_path
    elif (args.pretrained == 'distilbert'):
        args.distilbert_path = model_path
    elif (args.pretrained == 'xlnet'):
        args.xlnet_path = model_path
    elif (args.pretrained == 'gpt2'):
        args.gpt2_path = model_path
    else:
        raise ValueError("Unsupported pretrained model:", args.pretrained)
    
    """
    elif (args.pretrained == 'albert'):
        args.albert_path = model_path
    elif (args.pretrained == 'llama'):
        args.llama_path = model_path
    """
    
    print("args:", args)    

    # initialize logging and other system run variables
    already_modelled, logfile, method_name, pretrained, embeddings, embedding_type, emb_path, lm_type, mode, system = initialize_testing(args, program, version)

    # check to see if model params have been computed already
    if (already_modelled and not args.force):
        print(f'--- model {method_name} with embeddings {embeddings}, pretrained == {pretrained}, and wc_supervised == {args.supervised} for {args.dataset} already calculated, run with --force option to override. ---')
        exit(0)

    print("embedding_type:", embedding_type)
    print("embeddings:", embeddings)    
    print("embedding_path:", emb_path)
    print("lm_type:", lm_type)
    print("mode:", mode)
    print("system:", system)
    
    # Check for CUDA and MPS availability
    # Default to CPU if neither CUDA nor MPS is available
    if torch.cuda.is_available():
        device = torch.device("cuda")

        if (args.dataset == 'rcv1'):
            batch_size = DEFAULT_MIN_CUDA_BATCH_SIZE
        else:
            batch_size = DEFAULT_MAX_CUDA_BATCH_SIZE

        # for eval_step calculation
        num_devices = system.get_num_gpus()

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

        if (args.dataset == 'rcv1'):
            batch_size = DEFAULT_MIN_MPS_BATCH_SIZE
        else:
            batch_size = DEFAULT_MAX_MPS_BATCH_SIZE

        # for eval_step calculation
        num_devices = 1                                # MPS only supports a single GPU
    else:
        if (args.dataset == 'rcv1'):
            batch_size = DEFAULT_MIN_CPU_BATCH_SIZE
        else:
            batch_size = DEFAULT_MAX_CPU_BATCH_SIZE     
        
        # for eval_step calculation
        num_devices = 1                                # No GPUs available (CPU), so we default to 1

    print(f"device: {device}")
    print("num_devices:", num_devices)
    print("batch_size:", batch_size)

    torch.manual_seed(args.seed)

    print(f'\n\t--- Loading dataset {args.dataset} data ---')

    #
    # Load dataset training and test data, associated labels, 
    # as well as number of classes (labels) and the target names of 
    # the dataset 
    #
    #(train_data, train_target), (test_data, labels_test), num_classes, target_names, class_type = trans_lc_load_dataset(args.dataset, args.seed)
    train_data, train_target, test_data, labels_test, num_classes, target_names, class_type = get_dataset_data(args.dataset, args.seed)

    print("class_type:", class_type)
    print("num_classes:", num_classes)
    print("target_names:", target_names)

    print("train_data:", type(train_data), len(train_data))
    print("train_data[0]:", type(train_data[0]), train_data[0])
    print("train_target:", type(train_target), len(train_target))
    print("train_target[0]:", type(train_target[0]), train_target[0].shape, train_target[0])

    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), len(labels_test))
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])

    # Check for empty strings in train_data and test_data
    check_empty_docs(train_data, "Train")
    check_empty_docs(test_data, "Test")

    #tokenizer, max_length, pad_token_id, tok_vocab_size = init_hf_tokenizer()
    lc_tokenizer = LCTokenizer(
        model_name=model_name,
        model_path=model_path,
        lowercase=True,                                         # should align with the way dataset is (pre) processed
        remove_special_tokens=False,
        padding='max_length',
        truncation=True
    )
    print("lc_tokenizer:", type(lc_tokenizer), lc_tokenizer)
    print("Tokenizer configuration:")
    print(f"  Pad token: {lc_tokenizer.tokenizer.pad_token} (ID: {lc_tokenizer.tokenizer.pad_token_id})")
    print(f"  Max length: {lc_tokenizer.max_length}")

    # Split off validation data from the training set
    texts_train, texts_val, labels_train, labels_val = train_test_split(
                                                            train_data, 
                                                            train_target, 
                                                            test_size=VAL_SIZE, 
                                                            shuffle=True,
                                                            random_state=RANDOM_SEED
                                                            )

    print("texts_train:", type(texts_train), len(texts_train))
    print("texts_train[0]:", type(texts_train[0]), texts_train[0])
    print("labels_train:", type(labels_train), len(labels_train))
    print("labels_train[0]:", type(labels_train[0]), labels_train[0].shape, labels_train[0])

    print("texts_val:", type(texts_val), len(texts_val))
    print("texts_val[0]:", type(texts_val[0]), texts_val[0])
    print("labels_val:", type(labels_val), len(labels_val))
    print("labels_val[0]:", type(labels_val[0]), labels_val[0].shape, labels_val[0])

    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), len(labels_test))
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])

    #
    # Vectorize the text data
    #
    print("\n\t vectorizing dataset...")

    vectorizer, Xtr, Xval, Xte = get_vectorized_data(
        texts_train,
        texts_val,
        test_data,
        lc_tokenizer,
        args.dataset,
        args.pretrained,
        args.vtype,
        debug=True
    )

    print("vectorizer:\n", vectorizer)

    tokenizer = lc_tokenizer.tokenizer
    print("lc_tokenizer.tokenizer (tokenizer):", type(tokenizer), tokenizer)

    print("Xtr:", type(Xtr), Xtr.shape)
    print("Xtr[0]:", type(Xtr[0]), Xtr[0].shape, Xtr[0].toarray().flatten())
    #print("Xtr[1]:", type(Xtr[1]), Xtr[1].shape, Xtr[1])
    #print("Xtr[1]:", type(Xtr[2]), Xtr[2].shape, Xtr[2])

    print("Xval:", type(Xval), Xval.shape)
    print("Xval[0]:", type(Xval[0]), Xval[0].shape, Xval[0])
    
    print("Xte:", type(Xte), Xte.shape)
    print("Xte[0]:", type(Xte[0]), Xte[0].shape, Xte[0])

    # convert single label y values from array of scalaers to one hot encoded
    print("vectorizer.vocabulary_:", type(vectorizer.vocabulary_), len(vectorizer.vocabulary_))
    vec_vocab_size = len(vectorizer.vocabulary_)
    print("vec_vocab_size:", vec_vocab_size)
    tok_vocab_size = len(tokenizer.get_vocab())
    print("tok_vocab_size:", tok_vocab_size)

    #
    # validate that the vectorize and tokenizer vocabularies are mirrors of each other
    # this is imperative for the TCE matrix computation alignment with the (pretrained) 
    # model embeddings
    #
    assert set(vectorizer.vocabulary_.keys()).issubset(tokenizer.get_vocab().keys()), "Vectorizer vocabulary must be a subset of tokenizer vocabulary"

    # Assertion: Ensure vectorizer vocabulary size matches tokenizer vocabulary size
    assert vec_vocab_size == tok_vocab_size, \
        f"Vectorizer vocab size ({vec_vocab_size}) must equal tokenizer vocab size ({tok_vocab_size})"

    tok_vocab = set(tokenizer.get_vocab().keys())
    vec_vocab = set(vectorizer.vocabulary_.keys())
    print(f"Tokenizer vocab size: {len(tok_vocab)}, Vectorizer vocab size: {len(vec_vocab)}")
    print(f"Vocabulary intersection size: {len(tok_vocab & vec_vocab)}")

    #
    # spot check the vectorized data to make sure the vectorizer 
    # and tokenizer are in sync, this is crucial for TCE computation
    #
    spot_check_documents(
        documents=texts_train,
        vectorizer=vectorizer,
        tokenizer=lc_tokenizer,                # we pass in the LCTokenizer 
        vectorized_data=Xtr,       
        num_docs=3,
        debug=True
    )

    #
    # get the relevant tokens from the dataset to send to LCSequenceClassifier
    #
    relevant_tokens = lc_tokenizer.get_dataset_tokens(texts_train, texts_val, test_data)
    print("relevant_tokens:", type(relevant_tokens), len(relevant_tokens))

    # Get the size of the dataset-specific vocabulary
    filtered_vocab_size = len(relevant_tokens)
    print(f"Relevant tokens size: {filtered_vocab_size}")

    # Assert that all relevant tokens are valid token IDs within the tokenizer's vocabulary
    assert all(
        0 <= token_id < len(tokenizer) for token_id in relevant_tokens
    ), "Some relevant tokens are not valid token IDs in the tokenizer vocabulary."

    # 
    # compute supervised embeddings if need be by calling compute_supervised_embeddings( and 
    # then instantiate the LCSequenceClassifier model)
    #
    if (args.supervised):

        print("\n\tcomputing TCEs...")

        if (class_type in ['single-label', 'singlelabel']):

            print("single label, converting target labels to to one-hot for tce computation...")
            
            # Assuming labels are a numpy array of shape (num_samples,)
            def to_one_hot(labels, num_classes):
                encoder = OneHotEncoder(categories='auto', sparse_output=True, dtype=np.float32)
                labels = labels.reshape(-1, 1)
                one_hot = encoder.fit_transform(labels)  # Result is a sparse matrix
                return one_hot

            one_hot_labels_train = to_one_hot(labels_train, num_classes=num_classes)

        else:
            one_hot_labels_train = labels_train

        # Ensure one_hot_labels_train is a csr_matrix
        if not isinstance(one_hot_labels_train, csr_matrix):
            print("Converting one_hot_labels_train to csr_matrix...")
            one_hot_labels_train = csr_matrix(one_hot_labels_train)
            
        tce_matrix = compute_tces(
            vocabsize=vec_vocab_size,
            vectorized_training_data=Xtr,
            training_label_matrix=one_hot_labels_train,
            opt=args,
            debug=True
        )
        tce_matrix.to(device)           # must move the TCE embeddings to same device as model

        print("tce_matrix:", type(tce_matrix), tce_matrix.shape)
        print("tce_matrix[0]:", type(tce_matrix[0]), tce_matrix[0].shape, tce_matrix[0])

        if torch.isnan(tce_matrix).any() or torch.isinf(tce_matrix).any():
            print("[WARNING}: tce_matrix contains NaN or Inf values during initialization.")
            #raise ValueError("[ERROR] tce_matrix contains NaN or Inf values during initialization.")
    else:
        tce_matrix = None

    print("\n\tBuilding Classifier...")

    hf_trans_model, hf_trans_class_model = get_hf_models(
                    model_name, 
                    model_path, 
                    num_classes=num_classes, 
                    tokenizer=tokenizer)
    
    hf_trans_model.to(device)
    hf_trans_class_model.to(device)
    
    print("\nhf_trans_model:\n", hf_trans_model)
    print("\nhf_trans_class_model:\n", hf_trans_class_model)

    hf_trans_model = hf_trans_class_model

    class_weights = None
    if (class_type in ['multi-label', 'multilabel']):
        print("computing class weights...")
        class_weights = lc_class_weights(labels_train, task_type=class_type)
    else:
        print("no class weights computed...")

    #
    # if specified, show the cl;ass distribution
    # especially helpful for multi-label datasets
    # where the class is unevenly distributed and hence
    # affects micro-f1 scores out of testing (with smaller 
    # samples where under represented classes in training 
    # are further underrepresented in the test dataset)
    #
    if (args.show_dist):
        cls_wghts = show_class_distribution(
            labels=labels_train, 
            target_names=target_names, 
            class_type=class_type, 
            dataset_name=args.dataset
            )
    print("\n")
    
    #
    # note we instantiate only with relevant_tokens 
    # from our custom tokenizer (lc_tokenizer)
    #
    lc_model = LCSequenceClassifier(
        hf_model=hf_trans_model,                        # HuggingFace transformer model being used
        num_classes=num_classes,                        # number of classes for classification
        vocab_size=tok_vocab_size,
        #relevant_tokens=relevant_tokens,                # relevant tokens for the dataset
        relevant_tokens=None,
        lc_tokenizer=lc_tokenizer,                      # HuggingFace tokenizer
        simple=args.simple,                             # whethe or not to use the simple classifier head
        class_type=class_type,                          # classification type, options 'single-label' or 'multi-label'
        #class_weights=class_weights,                   # class weights for loss function
        supervised=args.supervised,
        tce_matrix=tce_matrix,
        finetune=args.tunable,                          # embeddings are trainable (True), default is False (static)
        normalize_tces=True,                 
        dropout_rate=args.dropprob,                     # dropout rate for TCEs
        comb_method=args.sup_mode,                      # combination method for TCEs with model embeddings, options 'cat', 'add', 'dot'
        #debug=True                                     # turns on active forware debugging
    ).to(device)
    print("\n\t-- Final LC Classifier Model --:\n", lc_model)

    # Get embedding size from the model
    dimensions, vec_size = lc_model.get_embedding_dims()
    print(f'dimensions: {dimensions}, vec_size: {vec_size}')

    # Concatenate supervised-specific dimensions if args.supervised is True
    if args.supervised:
        # Convert tce_matrix.shape to string before concatenating
        dimensions = f"{dimensions}:{str(tce_matrix.shape)}"
    # Log the dimensions
    print("dimensions (for logger):", dimensions)

    # Prepare datasets
    train_dataset = LCDataset(
        texts_train, 
        labels_train, 
        tokenizer, 
        max_length=lc_tokenizer.max_length,
        class_type=class_type 
    )

    val_dataset = LCDataset(
        texts_val, 
        labels_val, 
        tokenizer, 
        max_length=lc_tokenizer.max_length,
        class_type=class_type
    )

    test_dataset = LCDataset(
        test_data, 
        labels_test, 
        tokenizer, 
        max_length=lc_tokenizer.max_length,
        class_type=class_type
    )

    tinit = time()

    #
    # set Training arguments
    #
    """
    training_args_orig = TrainingArguments(
        output_dir='../out',
        evaluation_strategy="epoch",                        
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir='../log',
        run_name='trans_layer_cake',
        seed=args.seed,
        report_to="none"
    )
    """

    """
    # 
    # set Training arguments with evaluation strategy set to steps
    #
    steps_per_epoch = len(texts_train) / (batch_size * num_devices)         # Calculate steps per epoch
    print(f"steps_per_epoch: {steps_per_epoch}")

    desired_evaluations_per_epoch = 2                                       # Evaluate 2 times per epoch
    print("desired_evaluations_per_epoch:", desired_evaluations_per_epoch)
    
    #eval_steps = steps_per_epoch / desired_evaluations_per_epoch            # Calculate eval_steps
    eval_steps = int(round(steps_per_epoch / desired_evaluations_per_epoch))
    print(f"eval_steps: {eval_steps}")

    # Ensure save_steps is a multiple of eval_steps
    save_steps = eval_steps  # Save after every evaluation
    print(f"save_steps: {save_steps}")
    """

    training_args_opt = TrainingArguments(
        output_dir='../out',
        #evaluation_strategy="steps",                                # Change to "steps" for evaluation after a fixed number of steps
        #eval_steps=eval_steps,                                      # Add this to specify evaluation frequency (e.g., every 100 steps)
        #save_strategy="steps",                                      # Optional: Save the model checkpoint after evaluation
        #save_steps=eval_steps,                                      # Same frequency as eval_steps to save checkpoints
        evaluation_strategy="epoch",                        
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir='../log',
        run_name='trans_layer_cake',
        seed=args.seed,
        report_to="none"
    )

    # Trainer with custom data collator
    trainer = Trainer(
        model=lc_model,                                     # use LCSequenceClassifier model
        args=training_args_opt,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, class_type, threshold=MC_THRESHOLD),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    print("\n\tbuilding model...")

    # Enable anomaly detection during training
    with torch.autograd.set_detect_anomaly(True):
        try:
            model_deets = trainer.train()
            print("model_deets\n:", model_deets)
        except RuntimeError as e:
            print(f"An error occurred during training: {e}")
            
    final_loss = model_deets.training_loss
    print("final_loss:", final_loss)

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:", test_results)

    # Predictions
    preds = trainer.predict(test_dataset)
    if class_type == 'single-label':
        y_pred = np.argmax(preds.predictions, axis=1)
    else:
        y_pred = (preds.predictions > 0.5).astype(int)
    
    """
    print("labels_test:", type(labels_test), labels_test.shape)
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])
    print("y_pred:", type(y_pred), y_pred.shape)
    print("y_pred[0]:", type(y_pred[0]), y_pred[0].shape, y_pred[0])
    """

    print(classification_report(labels_test, y_pred, target_names=target_names, digits=4))

    macrof1, microf1, acc, h_loss, precision, recall, j_index = evaluation_nn(labels_test, y_pred, classification_type=class_type)
    print("\n--Layer Cake Metrics--")
    print(f"Macro-F1 = {macrof1:.4f}, Micro-F1 = {microf1:.4f}, Accuracy = {acc:.4f}, H-loss = {h_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, Jaccard = {j_index:.4f}")
    
    tend = time() - tinit

    measure_prefix = 'final-te'
    #epoch = trainer.state.epoch
    epoch = int(round(trainer.state.epoch))
    print("epoch:", epoch)

    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-macro-f1', value=macrof1, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-micro-f1', value=microf1, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-loss', value=final_loss, timelapse=tend)

    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-precision', value=precision, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-recall', value=recall, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-jacard-index', value=j_index, timelapse=tend)

    print("\n\t--- model training and evaluation complete---\n")


