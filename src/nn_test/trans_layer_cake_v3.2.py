import argparse
import os
import numpy as np
import pandas as pd
from time import time

import matplotlib.pyplot as plt
from tqdm import tqdm

import nltk
from nltk.corpus import reuters

from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Adam
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR

import transformers     # Hugging Face transformers
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


from transformers import PreTrainedModel
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel

from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1

from util.metrics import evaluation_nn
from util.common import initialize_testing, get_embedding_type

from embedding.supervised import get_supervised_embeddings

from embedding.pretrained import MODEL_MAP

from scipy.sparse import csr_matrix



SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news", "ohsumed"]


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
DEFAULT_CPU_BATCH_SIZE = 16
DEFAULT_GPU_BATCH_SIZE = 64
DEFAULT_MPS_BATCH_SIZE = 16

TEST_SIZE = 0.15
VAL_SIZE = 0.15




# Load dataset
def load_dataset(name):

    print("Loading dataset:", name)

    if name == "20newsgroups":

        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

        target_names = list(set(train_data.target_names))  # Ensures unique class names
        num_classes = len(target_names)
        #print(f"num_classes: {len(target_names)}")
        #print("class_names:", target_names)

        class_type = 'single-label'

        return (train_data.data, train_data.target), (test_data.data, test_data.target), num_classes, target_names, class_type
    
    elif name == "bbc-news":

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
            random_state = RANDOM_SEED,
        )

        # reset indeces
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

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

        return (train_data.tolist(), train_target_encoded), (test_data.tolist(), test_target_encoded), num_classes, target_names, class_type
    
    elif name == "reuters21578":
        
        data_path = os.path.join(DATASET_DIR, 'reuters21578')    
        print("data_path:", data_path)  

        train_labelled_docs = fetch_reuters21578(subset='train', data_path=data_path)
        test_labelled_docs = fetch_reuters21578(subset='test', data_path=data_path)

        train_data = train_labelled_docs.data
        train_target = train_labelled_docs.target
        test_data = list(test_labelled_docs.data)
        test_target = test_labelled_docs.target

        class_type = 'multi-label'

        train_target, test_target, target_names = _label_matrix(train_target, test_target)

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        target_names = train_labelled_docs.target_names
        num_classes = len(target_names)
        #print(f"num_classes: {len(target_names)}")
        #print("class_names:", target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type
    

    elif name == "ohsumed":
        
        data_path = os.path.join(DATASET_DIR, 'ohsumed50k')
        devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        test = fetch_ohsumed50k(subset='test', data_path=data_path)

        train_data, train_target = devel.data, devel.target
        test_data, test_target = test.data, test.target

        class_type = 'multi-label'
        #self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        #self.devel_raw, self.test_raw = devel.data, test.data
        
        train_target, test_target, target_names = _label_matrix(train_target, test_target)

        target_names = devel.target_names

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        num_classes = len(target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type

    elif name == "rcv1":

        data_path = os.path.join(DATASET_DIR, 'rcv1')
        
        devel = fetch_RCV1(subset='train', data_path=data_path)
        test = fetch_RCV1(subset='test', data_path=data_path)

        train_data, train_target = devel.data, devel.target
        test_data, test_target = test.data, test.target

        class_type = 'multi-label'
        #self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        #self.devel_raw, self.test_raw = devel.data, test.data
        
        train_target, test_target, target_names = _label_matrix(train_target, test_target)

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        num_classes = len(target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type

    else:
        raise ValueError("Unsupported dataset:", name)



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



def compute_tces(vocabsize, vectorized_training_data, training_label_matrix, opt):
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

    print(f'compute_tces(): vocabsize: {vocabsize}, opt.supervised: {opt.supervised}')
    
    Xtr = vectorized_training_data
    Ytr = training_label_matrix
    #print("\tXtr:", type(Xtr), Xtr.shape)
    #print("\tYtr:", type(Ytr), Ytr.shape)

    TCE = get_supervised_embeddings(
        Xtr, 
        Ytr, 
        method=opt.supervised_method,
        max_label_space=opt.max_label_space,
        dozscore=(not opt.nozscore)
    )

    # Adjust TCE matrix size
    num_missing_rows = vocabsize - TCE.shape[0]
    print("num_missing_rows:", num_missing_rows)

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




SUPPORTED_OPS = ["concat", "add", "dot"]



class LCHFTCEClassifier2(nn.Module):

    def __init__(self, hf_model, num_classes, class_type='single-label', supervised=False, tce_matrix=None, comb_method="concat"):
        """
        A Transformer-based classifier with optional TCE integration.
        
        Args:
            hf_model: The HuggingFace pre-trained transformer model (e.g., BERT), preloaded.
            num_classes: Number of classes for classification.
            supervised: Boolean indicating if supervised embeddings are used.
            tce_matrix: Precomputed TCE matrix (Tensor) with shape [vocab_size, num_classes].
            comb-method: Method to integrate WCE embeddings ("add" or "concat").
        """
        super(LCHFTCEClassifier2, self).__init__()

        print(f'LCHFTCEClassifier2:__init__()... class_type: {class_type}, num_classes: {num_classes}, supervised: {supervised}, comb_method: {comb_method}')

        self.transformer = hf_model
        self.hidden_size = self.transformer.config.hidden_size
        print("self.hidden_size:", self.hidden_size)

        self.num_classes = num_classes
        self.class_type = class_type
        self.supervised = supervised
        self.comb_method = comb_method

        self.tce_matrix = tce_matrix

        if (self.tce_matrix is not None):
            print("self.tce_matrix:", type(self.tce_matrix), self.tce_matrix.shape)
        else:
            print("self.tce_matrix: None")

        # Adjust classifier input size if TCEs are used
        if self.supervised and self.comb_method == "concat":
            self.classifier_input_size = self.hidden_size + self.num_classes
        else:
            self.classifier_input_size = self.hidden_size
        
        self.classifier = nn.Linear(self.classifier_input_size, self.num_classes)

        """
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        """

    def forward(self, input_ids=None, attention_mask=None, labels=None, tces=None):
        """
        Forward pass with optional supervised embeddings (TCEs).
        """

        # Move TCE matrix to the same device as input_ids if necessary
        if self.tce_matrix is not None:
            self.tce_matrix = self.tce_matrix.to(input_ids.device)

        # Base model forward pass
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Pool the CLS token or equivalent for XLNet
        if hasattr(outputs, "last_hidden_state"):                                   # For models like BERT
            pooled_output = outputs.last_hidden_state[:, 0]
        elif hasattr(outputs, "hidden_states"):                                     # XLNet or similar
            sequence_output = outputs.hidden_states[-1]                             # Access the last layer's hidden states
            pooled_output = sequence_output[:, -1, :]                               # Use the last token for pooling
        else:
            raise ValueError("Unsupported model output structure. Ensure compatibility.")

        if self.supervised and self.tce_matrix is not None:
                
            # Map input_ids to TCE embeddings
            tces = self.tce_matrix[input_ids]  # Shape: (batch_size, seq_len, tce_dim)

            if self.comb_method == "concat":
                pooled_output = torch.cat([pooled_output, tces.mean(dim=1)], dim=1)  # Concatenate TCEs
            elif self.comb_method == "add":
                pooled_output = pooled_output + tces.mean(dim=1)  # Add TCEs
            else:
                raise ValueError(f"Unsupported comb_method: {self.comb_method}")            

        # Classification head
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            #if labels.dtype == torch.long:                         
            if class_type in ['single-label', 'singlelabel']:       # Single-label classification
                loss_fn = nn.CrossEntropyLoss()
            else:                                                   # Multi-label classification
                loss_fn = nn.BCEWithLogitsLoss()

            loss = loss_fn(logits, labels)
        else:
            print("WARNINMG: No labels provided for loss calculation.")

        return {"loss": loss, "logits": logits}
    




class LCClassifier3(nn.Module):

    def __init__(self, hf_model, num_classes, class_type='single-label', supervised=False, tce_matrix=None, comb_method="dot", debug=False):
        """
        A Transformer-based classifier with optional TCE integration.

        Args:
            hf_model: A Hugging Face Sequence Classification model (e.g., BertForSequenceClassification).
            num_classes: Number of classes for classification.
            class_type: 'single-label' or 'multi-label'.
            supervised: Boolean indicating if supervised embeddings are used.
            tce_matrix: Precomputed TCE matrix (Tensor) with shape [vocab_size, num_classes].
            comb_method: Method to integrate TCE embeddings ("add" or "concat" or "dot").
            debug: Boolean flag for debug mode.
        """
        super(LCClassifier3, self).__init__()

        print(f"LCClassifier3:__init__()... class_type: {class_type}, num_classes: {num_classes}, supervised: {supervised}, comb_method: {comb_method}, debug: {debug}")

        self.transformer = hf_model

        # all the tensors need to be stored contiguously in memory
        for param in self.transformer.parameters():
            param.data = param.data.contiguous()
        
        self.num_classes = num_classes
        self.class_type = class_type
        self.supervised = supervised
        self.comb_method = comb_method
        self.debug = debug  
        self.tce_matrix = tce_matrix

        if self.tce_matrix is not None:
            print("-- original tce matrix --:", type(self.tce_matrix), self.tce_matrix.shape)

            # Normalize the TCE matrix using transformer embedding stats
            embedding_layer = self.transformer.get_input_embeddings()
            embedding_mean = embedding_layer.weight.mean(dim=0).to(device)
            embedding_std = embedding_layer.weight.std(dim=0).to(device)
            print(f"transformer embeddings mean: {embedding_mean.shape}, std: {embedding_std.shape}")

            self.tce_matrix = self._normalize_tce(self.tce_matrix, embedding_mean, embedding_std)
            print(f"Normalized TCE matrix: {type(self.tce_matrix)}, {self.tce_matrix.shape}")

            # Ensure the TCE matrix is normalized to the same distribution as the transformer embeddings
            self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=True)
            print(f"TCE embedding layer created with shape: {self.tce_layer.weight.shape}")
        else:
            print("self.tce_matrix: None")
            

        # Extract the hidden size from the model configuration
        self.hidden_size = self.transformer.config.hidden_size
        print("self.hidden_size:", self.hidden_size)

        # Add a projection layer for `dot` operation to ensure hidden_size compatibility
        if self.supervised and comb_method == "dot":
            self.projection_layer = nn.Linear(self.tce_matrix.size(1), self.hidden_size)

        if self.supervised is None and self.tce_matrix is None:
            combined_size = self.hidden_size
        else:
            # Adapt classifier head based on combination method
            combined_size = self.hidden_size if comb_method in ["dot", "add"] else self.hidden_size + self.tce_matrix.size(1)
            
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        
        
    def _normalize_tce(self, tce_matrix, embedding_mean, embedding_std):
        """
        Normalize the TCE matrix to align with the transformer's embedding space.

        Args:
        - tce_matrix: TCE matrix (vocab_size x num_classes).
        - embedding_mean: Mean of the transformer embeddings (1D tensor, size=model_dim).
        - embedding_std: Standard deviation of the transformer embeddings (1D tensor, size=model_dim).

        Returns:
        - Normalized TCE matrix (vocab_size x model_dim).
        """
        device = embedding_mean.device  # Ensure all tensors are on the same device
        tce_matrix = tce_matrix.to(device)

        tce_mean = tce_matrix.mean(dim=1, keepdim=True).to(device)
        tce_std = tce_matrix.std(dim=1, keepdim=True).to(device)
        tce_std[tce_std == 0] = 1  # Avoid division by zero

        # Expand TCE dimensions and normalize
        tce_matrix_expanded = tce_matrix.unsqueeze(-1).expand(-1, -1, embedding_mean.size(0)).mean(dim=1).to(device)
        normalized_tce = (tce_matrix_expanded - tce_mean) / tce_std  # Normalize
        normalized_tce = normalized_tce * embedding_std + embedding_mean  # Scale to transformer embedding stats

        return normalized_tce


    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass with optional supervised embeddings (TCEs).
        """
        # Pass inputs through the transformer model
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Access the pooled output or the CLS token representation
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output                                               # Use pooled output if available
            if (self.debug):
                print(f"[DEBUG] outputs.pooler_output.shape: {pooled_output.shape}")
        else:
            pooled_output = outputs.hidden_states[-1][:, 0]                                     # Use CLS token embedding from the last hidden layer
            if (self.debug):
                print(f"[DEBUG] outputs.hiudden_states[-1][:, 0].shape: {pooled_output.shape}")

        #
        # Integrate TCEs if supervised is True
        #
        if (self.supervised and self.tce_matrix is not None):

            print("integrating TCEs into the model...")

            self.tce_matrix = self.tce_matrix.to(input_ids.device)                              # Ensure `self.tce_matrix` is on the same device as `input_ids`

            # Debug info: Check for out-of-range indices
            vocab_size = self.tce_matrix.size(0)
            invalid_indices = input_ids[input_ids >= vocab_size]
            if invalid_indices.numel() > 0:
                print(f"[WARNING] Found invalid indices in input_ids: {invalid_indices.tolist()} (max valid index: {vocab_size - 1})")

            # Lookup TCEs for the input tokens
            tces = self.tce_matrix[input_ids].mean(dim=1)                                       # Average TCEs across tokens
            
            # Debug info: TCEs shape and match checks    
            if (self.debug):
                print(f"[DEBUG] tces.shape: {tces.shape}")
                print(f"[DEBUG] pooled_output.shape: {pooled_output.shape}")
            
            if self.comb_method == "concat":    
                print("concatenating two matrices...")                                                
                assert pooled_output.size(0) == tces.size(0), "Batch size mismatch between pooled output and TCEs"
                assert pooled_output.size(1) + tces.size(1) == self.hidden_size + self.tce_matrix.size(1), \
                    f"Concat dimension mismatch: {pooled_output.size(1) + tces.size(1)} != {self.hidden_size + self.tce_matrix.size(1)}"
                combined_output = torch.cat([pooled_output, tces], dim=1)                         
            elif self.comb_method == "add":                                                  
                print("adding two matricses...")
                assert pooled_output.size(0) == tces.size(0), "Batch size mismatch between pooled output and TCEs"
                assert pooled_output.size(1) == tces.size(1), \
                    f"Add dimension mismatch: {pooled_output.size(1)} != {tces.size(1)}"
                combined_output = pooled_output + tces   
            elif self.comb_method == "dot":
                print("multiplying two matricses...")
                assert pooled_output.size(0) == tces.size(0), "Batch size mismatch between pooled output and TCEs"
                tces_projected = self.projection_layer(tces)                            # Project TCEs to hidden_size
                combined_output = pooled_output * tces_projected                        # Element-wise multiplication
            else:
                raise ValueError(f"Unsupported combination method: {self.comb_method}")    
        
            pooled_output = combined_output                                             # Update pooled output with combined representation

        if (self.debug):
            print(f'pooled_output: type: {type(pooled_output)}, shape: {pooled_output.shape}')
        
        # Pass the combined representation through the classifier
        logits = self.classifier(pooled_output)

        if labels is None:
            print("WARNING: labels are None, cant compuet lose, returning logits only...")
            return {"logits": logits}
        
        # Compute loss if labels are provided
        if self.class_type in ["single-label", "singlelabel"]:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}



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
    if hf_trans_model.config.pad_token_id is None:
        print("hf_trans_model padding token ID is None, setting to tokenizer.pad_token_id...")
        hf_trans_model.config.pad_token_id = tokenizer.pad_token_id
        

    # Initialize Hugging Face Transformer model for sequence classification
    hf_trans_class_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=model_path,
        num_labels=num_classes  # Specify the number of classes for classification
    )

    # Ensure hidden states are enabled for the classification model
    hf_trans_class_model.config.output_hidden_states = True

    # GPT2 does not have a native pad token ID, set to tokenizer.pad_token_id
    if hf_trans_class_model.config.pad_token_id is None:
        print("hf_trans_class_model padding token ID is None, setting to tokenizer.pad_token_id...")
        hf_trans_class_model.config.pad_token_id = tokenizer.pad_token_id
    
    return hf_trans_model, hf_trans_class_model



def get_hf_models_old(model_name, model_path, num_classes, tokenizer):

    print(f'get_hf_models(): model_name: {model_name}, model_path: {model_path}, num_classes: {num_classes}')

    # Initialize Hugging Face Transfromer model
    hf_trans_model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
    
    # GPT2 does not have a native pad token ID, set to tokenizer.pad_token_id
    if (hf_trans_model.config.pad_token_id is None):
        print("hf_trans_model padding token ID is None, setting to tokenizer.pad_token_id...")
        hf_trans_model.config.pad_token_id = tokenizer.pad_token_id
        
    # Initialize Hugging Face Transformer model for sequence classification
    hf_trans_class_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=model_path,
        num_labels=num_classes  # Specify the number of classes for classification
    )

    # GPT2 does not have a native pad token ID, set to tokenizer.pad_token_id
    if (hf_trans_class_model.config.pad_token_id is None):
        print("hf_trans_class_model padding token ID is None, setting to tokenizer.pad_token_id...")
        hf_trans_class_model.config.pad_token_id = tokenizer.pad_token_id
    
    return hf_trans_model, hf_trans_class_model



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
        loss.backward(retain_graph=True)  # Adjust based on need
        return loss.detach()
    


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Text Classification with Transformer Models.")
    
    parser.add_argument('--dataset', required=True, type=str, choices=SUPPORTED_DATASETS, help='Dataset to use')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--pretrained', type=str, choices=['bert', 'roberta', 'distilbert', 'albert', 'xlnet', 'gpt2', 'llama'], help='Pretrained embeddings')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--supervised', action='store_true', help='Use supervised embeddings')
    parser.add_argument('--dist', action='store_true', default=False, help='show class distribution plots')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=PATIENCE, help='Patience for early stopping')
    parser.add_argument('--log-file', type=str, default='../log/lc_nn_test.test', help='Path to log file')
    parser.add_argument('--force', action='store_true', default=False, help='do not check if this experiment has already been run')
    parser.add_argument('--dropprob', type=float, default=0.1, metavar='[0.0, 1.0]', help='dropout probability (default: 0.1)')
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

    print("\n\t--- TRANS_LAYER_CAKE 3.0 ---")

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

    #embedding_type = get_embedding_type(args)
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
        batch_size = DEFAULT_GPU_BATCH_SIZE
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
    (train_data, train_target), (test_data, labels_test), num_classes, target_names, class_type = load_dataset(args.dataset)

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

    #
    # check set up the tokenizer
    #    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
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

    # 
    # compute supervised embeddings if need be by calling compute_supervised_embeddings( and 
    # then instantiate the LCSequenceClassifier model)
    #
    if (args.supervised):

        if (class_type in ['single-label', 'singlelabel']):

            print("single label, converting target labels to to one-hot for tce computation...")

            label_binarizer = LabelBinarizer()        
            one_hot_labels_train = label_binarizer.fit_transform(labels_train)

        tce_matrix = compute_tces(
            vocabsize=vec_vocab_size,
            vectorized_training_data=Xtr,
            training_label_matrix=one_hot_labels_train,
            opt=args
        )

        print("tce_matrix:", type(tce_matrix), tce_matrix.shape)

        tce_matrix.to(device)           # must move the TCE embeddings to same device as model
    else:
        tce_matrix = None

    hf_trans_model, hf_trans_class_model = get_hf_models(
                    model_name, 
                    model_path, 
                    num_classes=num_classes, 
                    tokenizer=tokenizer)
    
    print("\n\thf_trans_model:\n", hf_trans_model)
    print("\n\thf_trans_class_model:\n", hf_trans_class_model)

    hf_trans_model.to(device)
    hf_trans_class_model.to(device)

    # Get embedding size from the model
    dimensions, vec_size = get_embedding_dims(hf_trans_model)
    print(f'model size: {dimensions}, embedding dimension: {vec_size}')

    hf_trans_model = hf_trans_class_model

    lc_model = LCHFTCEClassifier2(
        hf_model=hf_trans_model,
        num_classes=num_classes,
        class_type=class_type,
        supervised=args.supervised,
        tce_matrix=tce_matrix,
        comb_method="dot"
    ).to(device)
    print("lc_model:\n", lc_model)

    # Prepare datasets
    train_dataset = LCDataset(
        texts_train, 
        labels_train, 
        tokenizer, 
        max_length=max_length,
        class_type=class_type,
        #supervised_embeddings=train_tces 
    )

    val_dataset = LCDataset(
        texts_val, 
        labels_val, 
        tokenizer, 
        max_length=max_length,
        class_type=class_type,
        #supervised_embeddings=val_tces
    )

    test_dataset = LCDataset(
        test_data, 
        labels_test, 
        tokenizer, 
        max_length=max_length,
        class_type=class_type,
        #supervised_embeddings=test_tces 
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
    trainer = CustomTrainer(
        model=lc_model,  # Use LCClassifier
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, class_type),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    """
    
    print("\n\t--- model training and evaluation ---\n")

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
    
    print("labels_test:", type(labels_test), labels_test.shape)
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])
    print("y_pred:", type(y_pred), y_pred.shape)
    print("y_pred[0]:", type(y_pred[0]), y_pred[0].shape, y_pred[0])

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


