import argparse
import os
import numpy as np
import pandas as pd
from time import time

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
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer

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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, DistilBertForSequenceClassification, AlbertForSequenceClassification
from transformers import XLNetForSequenceClassification, GPT2ForSequenceClassification, LlamaForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoConfig, PreTrainedModel

# Custom
from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1
from data.lc_dataset import trans_lc_load_dataset

from util.metrics import evaluation_nn
from util.common import initialize_testing, get_embedding_type

from embedding.supervised import get_supervised_embeddings

from embedding.pretrained import MODEL_MAP



#SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news", "ohsumed", "imdb", "cmu_movie_corpus"]
SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news", "ohsumed", "imdb", "arxiv"]

DATASET_DIR = "../datasets/"
VECTOR_CACHE = "../.vector_cache"

RANDOM_SEED = 42

#
# hyper parameters
#
MC_THRESHOLD = 0.5          # Multi-class threshold
PATIENCE = 5                # Early stopping patience
LEARNING_RATE = 1e-6        # Learning rate
EPOCHS = 10

MAX_TOKEN_LENGTH = 1024      # Maximum token length for transformer models models

# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 8
DEFAULT_MPS_BATCH_SIZE = 8
DEFAULT_CUDA_BATCH_SIZE = 8


TEST_SIZE = 0.15
VAL_SIZE = 0.15

#
# supported operations for transformer classifier
# combination method with TCEs
#
SUPPORTED_OPS = ["cat", "add", "dot"]


#
# whether or not to use mean_pooling in the
# forward method of the Classifier for all models
#
USE_MEAN_POOLING = False


# Get the full model identifier and load from local directory
def get_model_identifier(pretrained, cache_dir="../.vector_cache"):

    model_name = MODEL_MAP.get(pretrained, pretrained)
    model_path = os.path.join(cache_dir, pretrained)

    return model_name, model_path



def vectorize(texts_train, texts_val, texts_test, tokenizer, vtype):

    print(f'vectorize(), vtype: {vtype}')

    if vtype == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=5, lowercase=False, sublinear_tf=True, vocabulary=tokenizer.get_vocab())
    elif vtype == 'count':
        vectorizer = CountVectorizer(min_df=5, lowercase=False, vocabulary=tokenizer.get_vocab())

    Xtr = vectorizer.fit_transform(texts_train)
    Xval = vectorizer.transform(texts_val)
    Xte = vectorizer.transform(texts_test)

    Xtr.sort_indices()
    Xval.sort_indices()
    Xte.sort_indices()

    # Ensure X_vectorized is a sparse matrix (in case of word-based embeddings)
    if not isinstance(Xtr, csr_matrix):
        Xtr = csr_matrix(Xtr)

    if not isinstance(Xval, csr_matrix):
        Xval = csr_matrix(Xval)

    if not isinstance(Xte, csr_matrix):
        Xte = csr_matrix(Xte)

    return vectorizer, Xtr, Xval, Xte



def compute_supervised_embeddings(Xtr, Ytr, Xval, Yval, Xte, Yte, opt):
    """
    Compute supervised embeddings for a given vectorized datasets - include training, validation
    and test dataset output.

    Parameters:
    - vocabsize: Size of the (tokenizer and vectorizer) vocabulary.
    - Xtr: Vectorized training data (e.g., TF-IDF, count vectors).
    - Ytr: Multi-label binary matrix for training labels.
    - Xval: Vectorized validation data.
    - Yval: Multi-label binary matrix for validation labels.
    - Xte: Vectorized test data.
    - Yte: Multi-label binary matrix for test labels.
    - opt: Options object with configuration (e.g., supervised method, max label space).

    Returns:
    - training_tces: Supervised embeddings for the training data.
    - val_tces: Supervised embeddings for the validation data.
    - test_tces: Supervised embeddings for the test data.
    """

    print(f'compute_supervised_embeddings(), opt.supervised_method: {opt.supervised_method}, opt.max_label_space: {opt.max_label_space}')

    training_tces = get_supervised_embeddings(
        Xtr, 
        Ytr, 
        method=opt.supervised_method,
        max_label_space=opt.max_label_space,
        dozscore=(not opt.nozscore)
    )
    print("training_tces:", type(training_tces), training_tces.shape)

    val_tces = get_supervised_embeddings(
        Xval, 
        Yval, 
        method=opt.supervised_method,
        max_label_space=opt.max_label_space,
        dozscore=(not opt.nozscore)
    )
    print("val_tces:", type(val_tces), val_tces.shape)

    test_tces = get_supervised_embeddings(
        Xte, 
        Yte, 
        method=opt.supervised_method,
        max_label_space=opt.max_label_space,
        dozscore=(not opt.nozscore)
    )
    print("test_tces:", type(test_tces), test_tces.shape)

    return training_tces, val_tces, test_tces



def compute_tces(vocabsize, vectorized_training_data, training_label_matrix, opt, debug=False):
    """
    Computes TCEs - supervised embeddings at the tokenized level for the text, and labels/classes, in the underlying dataset.

    Parameters:
    - model: Hugging Face transformer model (e.g., `AutoModel.from_pretrained(...)`)
    - tokenizer: Hugging Face tokenizer (e.g., `AutoTokenizer.from_pretrained(...)`)
    - vocabsize: Size of the vocabulary.
    - word2index: Dictionary mapping words to their index.
    - out_of_vocabulary: List of words not found in the pretrained model.
    - vectorized_training_data: Vectorized training data (e.g., TF-IDF, count vectors).
    - training_label_matrix: Multi-label binary matrix for training labels.
    - opt: Options object with configuration (e.g., whether to include supervised embeddings).

    Returns:
    - pretrained_embeddings: A tensor containing pretrained embeddings for the text at hand
    - wce_matrix: computed supervised (Word-Class) embeddings, aka WCEs
    - sup_range: Range in the embedding matrix where supervised embeddings are located.
    """

    print(f'compute_tces(): vocabsize: {vocabsize}, opt.supervised: {opt.supervised}, opt.supervised_method: {opt.supervised_method}, opt.max_label_space: {opt.max_label_space}')
    
    Xtr = vectorized_training_data
    Ytr = training_label_matrix
    #print("\tXtr:", type(Xtr), Xtr.shape)
    #print("\tYtr:", type(Ytr), Ytr.shape)

    TCE = get_supervised_embeddings(
        Xtr, 
        Ytr, 
        method=opt.supervised_method,
        max_label_space=opt.max_label_space,
        dozscore=(not opt.nozscore),
        debug=debug
    )
    
    # Adjust TCE matrix size
    num_missing_rows = vocabsize - TCE.shape[0]
    if (debug):
        print("TCE:", type(TCE), TCE.shape)
        print("TCE[0]:", type(TCE[0]), TCE[0])
        print("num_missing_rows:", num_missing_rows)

    if (num_missing_rows > 0):
        TCE = np.vstack((TCE, np.zeros((num_missing_rows, TCE.shape[1]))))
    
    tce_matrix = torch.from_numpy(TCE).float()

    return tce_matrix



def embedding_matrices(embedding_layer, tokenizer, vocabsize, word2index, out_of_vocabulary, vectorized_training_data, training_label_matrix, opt):
    """
    Creates an embedding matrix that includes both pretrained and supervised embeddings.

    Parameters:
    - model: Hugging Face transformer model (e.g., `AutoModel.from_pretrained(...)`)
    - tokenizer: Hugging Face tokenizer (e.g., `AutoTokenizer.from_pretrained(...)`)
    - vocabsize: Size of the vocabulary.
    - word2index: Dictionary mapping words to their index.
    - out_of_vocabulary: List of words not found in the pretrained model.
    - vectorized_training_data: Vectorized training data (e.g., TF-IDF, count vectors).
    - training_label_matrix: Multi-label binary matrix for training labels.
    - opt: Options object with configuration (e.g., whether to include supervised embeddings).

    Returns:
    - pretrained_embeddings: A tensor containing pretrained embeddings for the text at hand
    - wce_matrix: computed supervised (Word-Class) embeddings, aka WCEs
    - sup_range: Range in the embedding matrix where supervised embeddings are located.
    """

    print(f'embedding_matrices(): opt.pretrained: {opt.pretrained}, vocabsize: {vocabsize}, opt.supervised: {opt.supervised}')

    pretrained_embeddings = []

    #embedding_layer = model.get_input_embeddings()
    embedding_dim = embedding_layer.embedding_dim
    embedding_matrix = torch.zeros((vocabsize, embedding_dim))

    # Pretrained embeddings
    if opt.pretrained:
        for word, idx in word2index.items():
            token_id = tokenizer.convert_tokens_to_ids(word)
            if token_id is not None and token_id < embedding_layer.num_embeddings:
                with torch.no_grad():
                    embedding = embedding_layer.weight[token_id].cpu()
                embedding_matrix[idx] = embedding
            else:
                out_of_vocabulary.append(word)

        pretrained_embeddings.append(embedding_matrix)
        print(f'\t[pretrained-matrix] {embedding_matrix.shape}')
    
    # Supervised embeddings (WCEs)
    wce_matrix = None
    if opt.supervised:
        print(f'computing supervised embeddings...')
        Xtr = vectorized_training_data
        Ytr = training_label_matrix
        print("\tXtr:", type(Xtr), Xtr.shape)
        print("\tYtr:", type(Ytr), Ytr.shape)

        WCE = get_supervised_embeddings(Xtr, Ytr, method=opt.supervised_method,
                                         max_label_space=opt.max_label_space,
                                         dozscore=(not opt.nozscore),
                                         transformers=True)

        # Adjust WCE matrix size
        num_missing_rows = vocabsize - WCE.shape[0]
        WCE = np.vstack((WCE, np.zeros((num_missing_rows, WCE.shape[1]))))
        wce_matrix = torch.from_numpy(WCE).float()
        print('\t[supervised-matrix]', wce_matrix.shape)

    return embedding_matrix, wce_matrix



def embedding_matrix(model, tokenizer, vocabsize, word2index, out_of_vocabulary, vectorized_training_data, training_label_matrix, opt):
    """
    Creates an embedding matrix that includes both pretrained and supervised embeddings.

    Parameters:
    - model: Hugging Face transformer model (e.g., `AutoModel.from_pretrained(...)`)
    - tokenizer: Hugging Face tokenizer (e.g., `AutoTokenizer.from_pretrained(...)`)
    - vocabsize: Size of the vocabulary.
    - word2index: Dictionary mapping words to their index.
    - out_of_vocabulary: List of words not found in the pretrained model.
    - opt: Options object with configuration (e.g., whether to include supervised embeddings).

    Returns:
    - pretrained_embeddings: A tensor containing combined pretrained and supervised embeddings.
    - sup_range: Range in the embedding matrix where supervised embeddings are located.
    """
    print(f'embedding_matrix(): opt.pretrained: {opt.pretrained},  vocabsize: {vocabsize}, opt.supervised: {opt.supervised}')
          
    pretrained_embeddings = []
    sup_range = None

     # Get the embedding layer for pretrained embeddings
    embedding_layer = model.get_input_embeddings()                  # Works across models like BERT, RoBERTa, DistilBERT, GPT, XLNet, LLaMA
    embedding_dim = embedding_layer.embedding_dim
    embedding_matrix = torch.zeros((vocabsize, embedding_dim))

    # If pretrained embeddings are needed
    if opt.pretrained:

        # Populate embedding matrix with pretrained embeddings
        for word, idx in word2index.items():
            token_id = tokenizer.convert_tokens_to_ids(word)
            if token_id is not None and token_id < embedding_layer.num_embeddings:
                with torch.no_grad():
                    embedding = embedding_layer.weight[token_id].cpu()
                embedding_matrix[idx] = embedding
            else:
                out_of_vocabulary.append(word)

        pretrained_embeddings.append(embedding_matrix)
        print(f'\t[pretrained-matrix] {embedding_matrix.shape}')

    # If supervised embeddings are needed
    if opt.supervised:

        print(f'computing supervised embeddings...')
        #Xtr, _ = vectorize_data(word2index, dataset)                # Function to vectorize the dataset
        #Ytr = dataset.devel_labelmatrix                             # Assuming devel_labelmatrix is the label matrix for training data
        
        Xtr = vectorized_training_data
        Ytr = training_label_matrix
        
        print("\tXtr:", type(Xtr), Xtr.shape)
        print("\tYtr:", type(Ytr), Ytr.shape)

        F = get_supervised_embeddings(
            Xtr, 
            Ytr,
            method=opt.supervised_method,
            max_label_space=opt.max_label_space,
            dozscore=(not opt.nozscore),
            transformers=True
        )
        
        # Adjust supervised embedding matrix to match vocabulary size
        num_missing_rows = vocabsize - F.shape[0]
        F = np.vstack((F, np.zeros((num_missing_rows, F.shape[1]))))
        F = torch.from_numpy(F).float()
        print('\t[supervised-matrix]', F.shape)

        # Concatenate supervised embeddings
        offset = pretrained_embeddings[0].shape[1] if pretrained_embeddings else 0
        sup_range = [offset, offset + F.shape[1]]
        pretrained_embeddings.append(F)

    # Concatenate all embeddings along the feature dimension
    pretrained_embeddings = torch.cat(pretrained_embeddings, dim=1) if pretrained_embeddings else None
    print(f'\t[final pretrained_embeddings] {pretrained_embeddings.shape}')

    return pretrained_embeddings, sup_range, pretrained_embeddings.shape[1]





class LCSequenceClassifier(nn.Module):

    def __init__(self, 
                hf_model: nn.Module, 
                num_classes: int, 
                vocab_size: int, 
                class_type: str = 'single-label', 
                class_weights: torch.Tensor = None, 
                supervised: bool = False, 
                tce_matrix: torch.Tensor = None, 
                trainable_tces: bool = False, 
                normalize_tces: bool = True,
                dropout_rate: float = 0.3, 
                comb_method: str = "cat", 
                debug: bool = False):
        """
        A Transformer-based classifier with optional TCE integration.
        
        Args:
            hf_model: The HuggingFace pre-trained transformer model (e.g., BERT), preloaded.
            num_classes: Number of classes for classification.
            vocab_size: size of (tokenizer) vocabulary, for assertions
            class_type: type of classification problem, either 'single-label' or 'multi-label'
            class_weights: Class weights for loss function.
            supervised: Boolean indicating if supervised embeddings are used.
            tce_matrix: Precomputed TCE matrix (Tensor) with shape [vocab_size, num_classes].
            trainable_tce: Boolean indicating if TCE matrix is trainable.
            normalize_tce: Boolean indicating if TCE matrix is normalized.
            dropout: Dropout rate for TCE matrix.
            comb-method: Method to integrate WCE embeddings ("add", "dot" or "cat").            
            debug: Debug mode flag.
        """
        super(LCSequenceClassifier, self).__init__()

        print(f'LCSequenceClassifier:__init__()... class_type: {class_type}, num_classes: {num_classes}, supervised: {supervised}, debug: {debug}')

        if (supervised):
            print(f'normalize_tces: {normalize_tces}, trainable_tces: {trainable_tces}, dropout_rate: {dropout_rate}, comb_method: {comb_method}')

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
        self.trainable_tces = trainable_tces
        self.class_weights = class_weights

        if (self.class_weights is not None):
            print("self.class_weights.shape:", self.class_weights.shape)
            if (self.debug):
                print("self.class_weights:", self.class_weights)
        else:
            print("self.class_weights is None")

        self.vocab_size = vocab_size
        print("self.vocab_size:", self.vocab_size)

        # Loss functions
        if class_type == 'multi-label':
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        elif class_type == 'single-label':
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError("class_type must be 'single-label' or 'multi-label'")
        print("loss_fn:", self.loss_fn)

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
 
            with torch.no_grad():

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
                    if (trainable_tces):
                        self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=False)
                    else:
                        self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=True)
                
                # Adapt classifier head based on combination method
                # 'concat' method is dimension_size + num_classes
                if (self.comb_method == 'cat'):
                    combined_size = self.hidden_size + self.tce_matrix.size(1)
                else:
                    combined_size = self.hidden_size                                # redundant but for clarity - good for 'add' or 'dot'
        
        # Classification head: maps the (potentially) combined input (transformer output or transformer output + optional TCE embeddings) 
        # to the final logits, introducing additional learnable parameters and allowing for flexibility to adapt the model to the specific task.
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 256),                  # First linear layer
            nn.ReLU(),                                      # non-linear activation function
            nn.Dropout(dropout_rate),                       # regularization
            nn.Linear(256, self.num_classes)                # FInal Linear layer
        )
        print("combined_size:", combined_size)
        print("self.classifier:", self.classifier)

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
            raise ValueError("[ERROR] projected_tce contains NaN or Inf values after normalization.")

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
            #print("input_ids:", input_ids)
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
                                            DistilBertForSequenceClassification, 
                                            AlbertForSequenceClassification)):
                pooled_output = outputs.hidden_states[-1][:, 0]                                                             # Use CLS token embedding
            elif isinstance(self.transformer, (GPT2ForSequenceClassification, 
                                            LlamaForSequenceClassification)):
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
            print("[ERROR] pooled_output contains NaN or Inf values")

        #
        # Integrate TCEs if supervised is True
        #
        #if (self.supervised and (self.tce_matrix is not None)):
        if self.tce_layer is not None:
            
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
                print("[ERROR] pooled_tce_embeddings contains NaN or Inf values")

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




def get_embedding_dims(hf_model):
    """
    Retrieve the embedding dimensions from the Hugging Face model.

    Parameters:
    - hf_model: A Hugging Face model instance.

    Returns:
    - Tuple[int]: The shape of the embedding layer (vocab_size, embedding_dim).
    """
    print("get_embedding_dims()...")
    #print("hf_model:\n", type(hf_model), hf_model)

    # Find the embedding layer dynamically
    if hasattr(hf_model, "embeddings"):                                             # BERT, RoBERTa, DistilBERT, ALBERT
        embedding_layer = hf_model.embeddings.word_embeddings
    elif hasattr(hf_model, "wte"):                                                  # GPT-2
        embedding_layer = hf_model.wte    
    elif hasattr(hf_model, "word_embedding"):                                       # XLNet        
        embedding_layer = hf_model.word_embedding
    elif hasattr(hf_model, "llama"):
        embedding_layer = hf_model.model.embed_tokens
    else:
        raise ValueError("Model not supported for automatic embedding extraction")

    print("embedding_layer:", type(embedding_layer), embedding_layer)

    model_size = embedding_layer.weight.shape
    embedding_size = hf_model.config.hidden_size
    
    return model_size, embedding_size
    


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
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    if class_type == 'single-label':
        # Convert predictions to class indices
        preds = np.argmax(predictions, axis=1)
        #labels = np.argmax(labels, axis=1)                              # Convert one-hot to class indices if needed
    elif class_type == 'multi-label':
        # Threshold predictions for multi-label classification
        preds = (predictions > threshold).astype(int)
        labels = labels.astype(int)                                     # Ensure labels are binary
    else:
        raise ValueError(f"Unsupported class_type: {class_type}")

    print("preds:", type(preds), preds.shape)
    print("labels:", type(labels), labels.shape)

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
    """

    print(f'get_hf_models(): model_name: {model_name}, model_path: {model_path}, num_classes: {num_classes}')

    # Initialize Hugging Face Transformer model
    hf_trans_model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
    
    # Ensure hidden states are enabled for the base model
    hf_trans_model.config.output_hidden_states = True

    # GPT2 does not have a native pad token ID, set to tokenizer.pad_token_id
    """
    if hf_trans_model.config.pad_token_id is None:
        print("hf_trans_model padding token ID is None, setting to tokenizer.pad_token_id...")
        hf_trans_model.config.pad_token_id = tokenizer.pad_token_id
    """

    if hf_trans_model.config.pad_token_id is None:
        print("hf_trans_model padding token ID is None, setting to tokenizer.eos_token id...")
        hf_trans_model.config.pad_token_id = tokenizer.eos_token  # Set PAD to end-of-sequence token    

    # Initialize Hugging Face Transformer model for sequence classification
    hf_trans_class_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=model_path,
        num_labels=num_classes,                         # Specify the number of classes for classification
        pad_token_id=tokenizer.pad_token_id             # Set PAD to end-of-sequence token
    )

    # Ensure hidden states are enabled for the classification model
    hf_trans_class_model.config.output_hidden_states = True

    # GPT2 does not have a native pad token ID, set to tokenizer.pad_token_id
    """
    if hf_trans_class_model.config.pad_token_id is None:
        print("hf_trans_class_model padding token ID is None, setting to tokenizer.pad_token_id...")
        hf_trans_class_model.config.pad_token_id = tokenizer.pad_token_id
    """

    if hf_trans_class_model.config.pad_token_id is None:
        print("hf_trans_class_model padding token ID is None, setting to tokenizer.pad_token_id...")
        hf_trans_class_model.config.pad_token_id = tokenizer.eos_token

    return hf_trans_model, hf_trans_class_model



class LCTrainer(Trainer):

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward(retain_graph=True)
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        elif self.deepspeed:
            self.deepspeed.backward(loss, retain_graph=True)
        else:
            loss.backward(retain_graph=True)

        return loss.detach()



class CustomTrainer(Trainer):
    
    def training_step(self, model, inputs):
        """
        Perform a training step on the model using inputs.

        Args:
            model: The model to train.
            inputs: The inputs and targets of the model.

        Returns:
            torch.Tensor: The training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward pass
        outputs = model(**inputs)
        loss = outputs["loss"]

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Backward pass with retain_graph=True
        #loss.backward(retain_graph=True)  # Adjust based on need
        self.accelerator.backward(loss, retain_graph=True)
        
        return loss.detach()
    


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Text Classification with Transformer Models.")
    
    parser.add_argument('--dataset', required=True, type=str, choices=SUPPORTED_DATASETS, help='Dataset to use')

    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--pretrained', type=str, choices=['bert', 'roberta', 'distilbert', 'albert', 'xlnet', 'gpt2', 'llama'], help='Pretrained embeddings')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--supervised', action='store_true', help='Use supervised embeddings (TCEs')
    parser.add_argument('--sup-mode', type=str, default='add', help='How to combine TCEs with model embeddings (add, dot, cat)')
    parser.add_argument('--dist', action='store_true', default=False, help='show class distribution plots')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=PATIENCE, help='Patience for early stopping')
    parser.add_argument('--log-file', type=str, default='../log/lc_nn_test.test', help='Path to log file')
    parser.add_argument('--force', action='store_true', default=False, help='do not check if this experiment has already been run')
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
                        help='TCEs are tunable, ie unfrozen from the beginning (default False, i.e., static)')
    parser.add_argument('--nozscore', action='store_true', default=False,
                        help='disables z-scoring form the computation of WCE')
    parser.add_argument('--vtype', type=str, default='tfidf', metavar='N', 
                        help=f'dataset base vectorization strategy, in [tfidf, count]')
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
    parser.add_argument('--max-label-space', type=int, default=300, metavar='int',
                        help='larger dimension allowed for the feature-label embedding (if larger, then PCA with this '
                             'number of components is applied (default 300)')

    return parser.parse_args()



# Main
if __name__ == "__main__":

    print("\n\t--- TRANS_LAYER_CAKE Version 4.1 ---")
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
    elif (args.pretrained == 'albert'):
        args.albert_path = model_path
    elif (args.pretrained == 'xlnet'):
        args.xlnet_path = model_path
    elif (args.pretrained == 'gpt2'):
        args.gpt2_path = model_path
    elif (args.pretrained == 'llama'):
        args.llama_path = model_path
    else:
        raise ValueError("Unsupported pretrained model:", args.pretrained)
    
    print("args:", args)    

    # initialize logging and other system run variables
    already_modelled, logfile, method_name, pretrained, embeddings, embedding_type, emb_path, lm_type, mode, system = initialize_testing(args)

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
        print("CUDA is available")
        device = torch.device("cuda")
        batch_size = DEFAULT_CUDA_BATCH_SIZE
    elif torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("mps")
        batch_size = DEFAULT_MPS_BATCH_SIZE
    else:
        print("Neither CUDA nor MPS is available, using CPU")
        device = torch.device("cpu")
        batch_size = DEFAULT_CPU_BATCH_SIZE     

    print(f"Using device: {device}")
    print("batch_size:", batch_size)

    torch.manual_seed(args.seed)

    # Load dataset and print class information
    (train_data, train_target), (test_data, labels_test), num_classes, target_names, class_type = trans_lc_load_dataset(args.dataset)

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


    print("\n\tinitializing tokenizer, vectorizer and dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
    
    # Add padding token if it doesn't exist
    # NB: it is necessary to stay within the 
    # boundaries of the model vocabulary size
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token               # Align pad token to eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id         # set pad_token_id
    print("tokenizer:\n", tokenizer)

    # Get the pad_token_id
    pad_token_id = tokenizer.pad_token_id

    tok_vocab_size = len(tokenizer)
    print("tok_vocab_size:", tok_vocab_size)

    # Compute max_length from tokenizer
    max_length = tokenizer.model_max_length
    print(f"Tokenizer max_length: {max_length}")

    if max_length > MAX_TOKEN_LENGTH:                                               # Handle excessive or default values
        print(f"Invalid max_length ({max_length}) detected. Adjusting to {MAX_TOKEN_LENGTH}.")
        max_length = MAX_TOKEN_LENGTH

    # Print tokenizer details for debugging
    print("Tokenizer configuration:")
    print(f"  Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  Max length: {max_length}")

    #
    # Split train into train and validation
    #
    texts_train, texts_val, labels_train, labels_val = train_test_split(train_data, train_target, test_size=VAL_SIZE, random_state=RANDOM_SEED)

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
    vectorizer, Xtr, Xval, Xte = vectorize(texts_train, texts_val, test_data, tokenizer, vtype=args.vtype)
    print("vectorizer:\n", vectorizer)

    print("Xtr:", type(Xtr), Xtr.shape)
    #print("Xtr[0]:", type(Xtr[0]), Xtr[0].shape, Xtr[0])
    
    print("Xval:", type(Xval), Xval.shape)
    #print("Xval[0]:", type(Xval[0]), Xval[0].shape, Xval[0])
    
    print("Xte:", type(Xte), Xte.shape)
    #print("Xte[0]:", type(Xte[0]), Xte[0].shape, Xte[0])

    # convert single label y values from array of scalaers to one hot encoded
    print("vectorizer.vocabulary_:", type(vectorizer.vocabulary_), len(vectorizer.vocabulary_))
    vec_vocab_size = len(vectorizer.vocabulary_)
    print("vec_vocab_size:", vec_vocab_size)
    tok_vocab_size = len(tokenizer.get_vocab())
    print("tok_vocab_size:", tok_vocab_size)

    assert set(vectorizer.vocabulary_.keys()).issubset(tokenizer.get_vocab().keys()), "Vectorizer vocabulary must be a subset of tokenizer vocabulary"

    # Assertion: Ensure vectorizer vocabulary size matches tokenizer vocabulary size
    assert vec_vocab_size == tok_vocab_size, \
        f"Vectorizer vocab size ({vec_vocab_size}) must equal tokenizer vocab size ({tok_vocab_size})"

    # 
    # compute supervised embeddings if need be by calling compute_supervised_embeddings( and 
    # then instantiate the LCSequenceClassifier model)
    #
    if (args.supervised):

        print("\n\tcomputing tces...")

        if (class_type in ['single-label', 'singlelabel']):

            print("single label, converting target labels to to one-hot for tce computation...")

            """
            label_binarizer = LabelBinarizer()        
            one_hot_labels_train = label_binarizer.fit_transform(labels_train)
            """
            from sklearn.preprocessing import OneHotEncoder
            
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
            #debug=True
        )

        print("tce_matrix:", type(tce_matrix), tce_matrix.shape)
        print("tce_matrix[0]:", type(tce_matrix[0]), tce_matrix[0].shape, tce_matrix[0])

        if torch.isnan(tce_matrix).any() or torch.isinf(tce_matrix).any():
            raise ValueError("[ERROR] tce_matrix contains NaN or Inf values during initialization.")

        tce_matrix.to(device)           # must move the TCE embeddings to same device as model
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
    """
    print("\n\thf_trans_model:\n", hf_trans_model)
    print("\n\thf_trans_class_model:\n", hf_trans_class_model)
    """

    # Get embedding size from the model
    dimensions, vec_size = get_embedding_dims(hf_trans_model)
    # Print for debugging
    print(f'model size: {dimensions}, embedding dimension: {vec_size}')
    # Concatenate supervised-specific dimensions if args.supervised is True
    if args.supervised:
        # Convert tce_matrix.shape to string before concatenating
        dimensions = f"{dimensions}:{str(tce_matrix.shape)}"
    # Log the dimensions
    print("dimensions (for logger):", dimensions)

    hf_trans_model = hf_trans_class_model
    #print("LC HF Sequence Classfier (hf_trans_model):\n", hf_trans_model)

    class_weights = None
    if (class_type in ['multi-label', 'multilabel']):
        print("computing class weights...")
        class_weights = lc_class_weights(labels_train, task_type=class_type)
        print("class weights:", class_weights)
    else:
        print("no class weights computed...")

    lc_model = LCSequenceClassifier(
        hf_model=hf_trans_model,
        num_classes=num_classes,
        vocab_size=tok_vocab_size,
        class_type=class_type,
        class_weights=class_weights,
        supervised=args.supervised,
        tce_matrix=tce_matrix,
        trainable_tces=args.tunable,
        normalize_tces=True,
        dropout_rate=args.dropprob,
        comb_method=args.sup_mode,
        #debug=True
    ).to(device)
    print("lc_model:\n", lc_model)

    # Prepare datasets
    train_dataset = LCDataset(
        texts_train, 
        labels_train, 
        tokenizer, 
        max_length=max_length,
        class_type=class_type 
    )

    val_dataset = LCDataset(
        texts_val, 
        labels_val, 
        tokenizer, 
        max_length=max_length,
        class_type=class_type
    )

    test_dataset = LCDataset(
        test_data, 
        labels_test, 
        tokenizer, 
        max_length=max_length,
        class_type=class_type
    )

    tinit = time()

    # Training arguments
    training_args = TrainingArguments(
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

    # Trainer with custom data collator
    trainer = Trainer(
        model=lc_model,                                     # be sure to use LC_Model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, class_type),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    
    """
    trainer = LCTrainer(
        model=lc_model,                                 # Use LCClassifier
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, class_type),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    """

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

    measure_prefix = 'final'
    #epoch = trainer.state.epoch
    epoch = int(round(trainer.state.epoch))
    print("epoch:", epoch)

    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=macrof1, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=microf1, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-loss', value=final_loss, timelapse=tend)

    logfile.insert(dimensions=dimensions, epoch=epoch, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure='te-precision', value=precision, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure='te-recall', value=recall, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure='te-jacard-index', value=j_index, timelapse=tend)

    print("\n\t--- model training and evaluation complete---\n")


