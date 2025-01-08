import argparse
import os
import numpy as np
import pandas as pd
from time import time

import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, coo_matrix

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

from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1

from util.metrics import evaluation_nn
from util.common import initialize_testing, get_embedding_type, get_word_list, index_dataset

from embedding.supervised import get_supervised_embeddings
from data.lc_dataset import LCDataset, loadpt_data




SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news"]


# Define a default model mapping (optional) to avoid invalid identifiers
MODEL_MAP = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "xlnet": "xlnet-base-cased",
    "gpt2": "gpt2",
    "llama": "meta-llama/Llama-2-7b-chat-hf"  # Example for a possible LLaMA identifier
}


MAX_LENGTH = 512  # Max sequence length for the transformer models

TEST_SIZE = 0.2
VAL_SIZE = 0.2

DATASET_DIR = "../datasets/"
VECTOR_CACHE = "../.vector_cache"

RANDOM_SEED = 42

#
# hyper parameters
#
BATCH_SIZE = 8
MC_THRESHOLD = 0.5          # Multi-class threshold
PATIENCE = 5                # Early stopping patience
LEARNING_RATE = 1e-6        # Learning rate
EPOCHS = 10


# Check device
def get_device():
    if torch.cuda.is_available():
        print("CUDA is available")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS is available")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


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

        train_target_sparse, test_target_sparse, target_names = _label_matrix(train_target, test_target)
        """
        print("label_matrix output")
        print("train_target:", type(train_target), train_target.shape)
        print("test_target:", type(test_target), test_target.shape)
        """

        train_target_dense = train_target_sparse.toarray()                                     # Convert to dense
        test_target_dense = test_target_sparse.toarray()                                       # Convert to dense
        """
        print("array output")
        print("train_target:", type(train_target), train_target.shape)
        print("test_target:", type(test_target), test_target.shape)
        """

        target_names = train_labelled_docs.target_names
        num_classes = len(target_names)
        #print(f"num_classes: {len(target_names)}")
        #print("class_names:", target_names)

        return (train_data, train_target_dense, train_target_sparse), (test_data, test_target_dense, test_target_sparse), num_classes, target_names, class_type
    
    elif name == "rcv1":

        rcv1 = fetch_rcv1()
        
        data = rcv1.data      # Sparse matrix of token counts
        target = rcv1.target  # Multi-label binarized format
        
        class_type = 'multi-label'
        
        # Split data into train and test
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, test_size=0.2, random_state=42, stratify=target
        )
        
        target_names = rcv1.target_names
        num_classes = len(target_names)
        #print(f"num_classes: {num_classes}")
        #print("class_names:", target_names)
        
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
        vectorizer = TfidfVectorizer(
            min_df=5, 
            lowercase=False, 
            sublinear_tf=True, 
            #vocabulary=tokenizer.get_vocab()
        )
    elif vtype == 'count':
        vectorizer = CountVectorizer(
            min_df=5, 
            lowercase=False, 
            #vocabulary=tokenizer.get_vocab()
        )

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



def embedding_matrix_split(model, tokenizer, vocabsize, word2index, out_of_vocabulary, vectorized_training_data, training_label_matrix, opt):
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
    sup_range = None
    embedding_layer = model.get_input_embeddings()
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



def embedding_matrix_cat(model, tokenizer, vocabsize, word2index, out_of_vocabulary, vectorized_training_data, training_label_matrix, opt):
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



def embedding_matrix_dot(model, tokenizer, vocabsize, word2index, out_of_vocabulary, vectorized_training_data, training_label_matrix, opt):
    """
    Creates a single embedding matrix by combining pretrained and supervised embeddings.

    Parameters:
    - model: Hugging Face transformer model (e.g., `AutoModel.from_pretrained(...)`)
    - tokenizer: Hugging Face tokenizer (e.g., `AutoTokenizer.from_pretrained(...)`)
    - vocabsize: Size of the vocabulary.
    - word2index: Dictionary mapping words to their index.
    - out_of_vocabulary: List of words not found in the pretrained model.
    - opt: Options object with configuration (e.g., whether to include supervised embeddings).

    Returns:
    - final_matrix: Tensor containing the result of multiplying pretrained embeddings with the word-class embedding matrix.
    """
    print(f'\n\tembedding_matrix(): opt.pretrained: {opt.pretrained}, vocabsize: {vocabsize}, opt.supervised: {opt.supervised}')
          
    # Get the embedding layer for pretrained embeddings
    embedding_layer = model.get_input_embeddings()                  # Works across models like BERT, RoBERTa, DistilBERT, GPT, XLNet, LLaMA
    embedding_dim = embedding_layer.embedding_dim                   # Pretrained embedding dimension
    print(f'\tembedding_dim: {embedding_dim}') 

    # initialize variables
    pretrained_matrix = torch.zeros((vocabsize, embedding_dim))  
    wce_matrix = None
    OOV = 0

    # If pretrained embeddings are needed, populate pretrained_matrix 
    # with pretrained model embeddings for each word in word2index
    if opt.pretrained:

        #print("\tcomputing pretrained embeddings...")

        # Wrap word2index items in tqdm for progress tracking
        for word, idx in tqdm(word2index.items(), desc="computing pretrained embeddings", total=len(word2index)):
        #for word, idx in word2index.items():
            token_id = tokenizer.convert_tokens_to_ids(word)
            if token_id is not None and token_id < embedding_layer.num_embeddings:
                with torch.no_grad():
                    pretrained_matrix[idx] = embedding_layer.weight[token_id].cpu()
            else:
                out_of_vocabulary.append(word)
                OOV += 1
        
        print("\tOOV words:", OOV)
        print(f'\n\t[pretrained-matrix]: {type(pretrained_matrix)}, {pretrained_matrix.shape}')


    # Compute Word-Class Embeddings (WCE) if supervised embeddings are needed
    if opt.supervised:
        print(f'\tcomputing supervised embeddings...')
        
        Xtr = vectorized_training_data
        Ytr = training_label_matrix

        """
        # Ensure Xtr is converted to numpy.ndarray
        if isinstance(Xtr, csr_matrix):
            Xtr = Xtr.toarray()
            print(f'Converted Xtr to numpy.ndarray: {type(Xtr)}, {Xtr.shape}')
        else:
            print(f'Xtr is already a numpy.ndarray: {type(Xtr)}, {Xtr.shape}')
        """

        """
        # Ensure Ytr is converted to csr_matrix
        if not isinstance(Ytr, csr_matrix):
            Ytr = csr_matrix(Ytr)
            print(f'Converted Ytr to csr_matrix: {type(Ytr)}, {Ytr.shape}')
        else:
            print(f'Ytr is already a csr_matrix: {type(Ytr)}, {Ytr.shape}')
        """

        print("Xtr:", type(Xtr), Xtr.shape)
        print("Xtr[0]:", type(Xtr[0]), Xtr[0])
        print("Ytr:", type(Ytr), Ytr.shape)
        print("Ytr[0]:", type(Ytr[0]), Ytr[0])
        
        WCE = get_supervised_embeddings(
            Xtr, 
            Ytr,
            method=opt.supervised_method,
            max_label_space=opt.max_label_space,
            dozscore=(not opt.nozscore),
            transformers=True
        )

        print("WCE:\n", type(WCE), WCE.shape, WCE)
    
        # Check if the matrix is a COO matrix and if 
        # so convert to desnse array for vstack operation
        if isinstance(WCE, coo_matrix):
            print('converting WCE to dense array...')
            WCE = WCE.toarray()
        print("WCE:\n", type(WCE), WCE.shape, WCE)
    
        num_missing_rows = vocabsize - WCE.shape[0]
        print("\tnum_missing_rows:", num_missing_rows)

        wce_matrix = np.vstack((WCE, np.zeros((num_missing_rows, WCE.shape[1]))))
        #wce_matrix = torch.from_numpy(WCE).float()
        print(f'\t[wce-matrix]: {type(wce_matrix)}, {wce_matrix.shape}')
        print(f'\twce_matrix[0]: {type(wce_matrix[0])}, {wce_matrix[0]}')

        # Project WCE matrix into the pretrained embedding space
        print("\tProjecting WCE matrix into pretrained embedding space...")

        # Transpose pretrained_matrix for dot product
        pretrained_matrix_np = pretrained_matrix.numpy()  # Convert to numpy for faster operations if needed
        print(f'\tpretrained_matrix_np: {type(pretrained_matrix_np)}, {pretrained_matrix_np.shape}')
    
        # Step 1: Define a projection matrix to map WCE into pretrained embedding space
        num_classes = wce_matrix.shape[1]
        embedding_dim = pretrained_matrix_np.shape[1]
        
        projection_layer = nn.Linear(num_classes, embedding_dim, bias=False)  # Project 115 -> 768
        torch.nn.init.xavier_uniform_(projection_layer.weight)  # Initialize weights

        # Step 2: Convert WCE to torch tensor for projection
        wce_matrix_tensor = torch.from_numpy(wce_matrix).float()  # Shape [30522, 115]

        # Step 3: Perform projection
        wce_projected = projection_layer(wce_matrix_tensor).detach().numpy()  # Shape [30522, 768]

        # Step 4: Combine with pretrained matrix
        final_matrix_np = pretrained_matrix_np + wce_projected  # Element-wise addition
        print(f'\t[final_matrix_np]: {type(final_matrix_np)}, {final_matrix_np.shape}')
        print(f'\tfinal_matrix_np[0]:, {type(final_matrix_np[0])}, {final_matrix_np[0]}')

        final_matrix = final_matrix_np

    else:
        # If no WCE, return pretrained embeddings as the final matrix
        final_matrix = pretrained_matrix

    print(f'\t[final matrix]: {type(final_matrix)}, {final_matrix.shape}')
    print(f'\tfinal_matrix[0]:, {type(final_matrix[0])}, {final_matrix[0]}')
    
    return final_matrix



def embedding_matrix(dataset, pretrained, vocabsize, word2index, out_of_vocabulary, opt):

    print(f'embedding_matrix()... dataset: {dataset}, pretrained: {pretrained}, vocabsize: {vocabsize}, supervised: {opt.supervised}')

    pretrained_embeddings = None
    sup_range = None
    
    if opt.pretrained or opt.supervised:
        pretrained_embeddings = []

        if pretrained is not None:
            word_list = get_word_list(word2index, out_of_vocabulary)
            weights = pretrained.extract(word_list)
            pretrained_embeddings.append(weights)
            print('\t[pretrained-matrix]', weights.shape)
            del pretrained

        if opt.supervised:
            Xtr, _ = dataset.vectorize()
            Ytr = dataset.devel_labelmatrix

            print("Xtr:", type(Xtr), Xtr.shape)
            print("Xtr[0]:", type(Xtr[0]), Xtr[0])
            print("Ytr:", type(Ytr), Ytr.shape)
            print("Ytr[0]:", type(Ytr[0]), Ytr[0])

            F = get_supervised_embeddings(Xtr, Ytr,
                                          method=opt.supervised_method,
                                          max_label_space=opt.max_label_space,
                                          dozscore=(not opt.nozscore))
            
            # Check if the matrix is a COO matrix and if 
            # so convert to desnse array for vstack operation
            if isinstance(F, coo_matrix):
                F = F.toarray()
            print("F:\n", type(F), F.shape, F)

            num_missing_rows = vocabsize - F.shape[0]
            F = np.vstack((F, np.zeros(shape=(num_missing_rows, F.shape[1]))))
            F = torch.from_numpy(F).float()
            print('\t[supervised-matrix]', F.shape)

            offset = 0
            if pretrained_embeddings:
                offset = pretrained_embeddings[0].shape[1]
            sup_range = [offset, offset + F.shape[1]]
            pretrained_embeddings.append(F)

        pretrained_embeddings = torch.cat(pretrained_embeddings, dim=1)
        print('\t[final pretrained_embeddings]\n\t', pretrained_embeddings.shape)

    return pretrained_embeddings, sup_range




class CustomTransformerForSequenceClassification(nn.Module):

    def __init__(self, model_name, num_labels, pretrained_embeddings=None, class_type='single-label', dropprob=0.1, freeze_embeddings=True):
        
        super(CustomTransformerForSequenceClassification, self).__init__()
        
        self.model_name = model_name

        self.class_type = class_type

        # Load the transformer model and its configuration
        self.transformer = AutoModel.from_pretrained(model_name)
        self.config = self.transformer.config

        # Handle dropout dynamically based on the model's configuration
        if hasattr(self.config, "hidden_dropout_prob"):
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        elif hasattr(self.config, "dropout"):
            self.dropout = nn.Dropout(self.config.dropout)
        else:
            self.dropout = nn.Dropout(dropprob)                         # Default to 0.1 if no dropout attribute is found

        # Pretrained embeddings
        self.pretrained_embeddings = pretrained_embeddings
    
        embedding_dim = pretrained_embeddings.size(1)

        # Optionally freeze the embeddings if required
        if freeze_embeddings:
            self.transformer.get_input_embeddings().weight.requires_grad = False

        # Fully connected classifier
        self.classifier = nn.Linear(embedding_dim, num_labels)


    def forward(self, input_ids, attention_mask=None, labels=None):

        # Ensure all tensors are on the same device as the model
        device = self.transformer.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if self.pretrained_embeddings is not None:
            self.pretrained_embeddings = self.pretrained_embeddings.to(device)

        # Extract transformer outputs
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Get the [CLS] token representation
        pooled_output = transformer_outputs.last_hidden_state[:, 0, :]

        # Optionally integrate pretrained embeddings
        if self.pretrained_embeddings is not None:
            batch_embeddings = self.pretrained_embeddings[input_ids]  # Shape: (batch_size, seq_len, embedding_dim)

            # Aggregate embeddings along the sequence length dimension (e.g., mean pooling)
            batch_embeddings = batch_embeddings.mean(dim=1)  # Shape: (batch_size, embedding_dim)

            # Concatenate the pooled transformer output with the aggregated embeddings
            pooled_output = torch.cat([pooled_output, batch_embeddings], dim=1)  # Shape: (batch_size, hidden_size + embedding_dim)

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Pass through the classifier
        logits = self.classifier(pooled_output)

        # Compute loss if labels are provided
        if labels is not None:
            if self.class_type in ['single-label', 'singlelabel']:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            elif self.class_type in ['multi-label', 'multilabel']:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
            else:
                raise ValueError(f"Unsupported problem type: {self.config.problem_type}")
            return loss, logits

        return logits
    



class LCDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH, class_type='single-label'):
        """
        Dataset class for handling tokenized inputs and labels.

        Parameters:
        - texts: List of input text samples.
        - labels: Binary vectors (multi-label format) or indices (single-label).
        - tokenizer: Hugging Face tokenizer for text tokenization.
        - max_length: Maximum length of tokenized sequences.
        - class_type: 'multi-label' or 'single-label' classification type.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_type = class_type

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get an individual sample from the dataset.

        Returns:
        - item: Dictionary containing tokenized inputs and labels.
        """
        text = self.texts[idx]
        labels = self.labels[idx] if self.labels is not None else [0]  # Default label if missing

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}  # Remove batch dimension

        # Add labels
        if self.class_type == 'single-label':
            item["labels"] = torch.tensor(labels, dtype=torch.long)
        else:
            item["labels"] = torch.tensor(labels, dtype=torch.float)

        return item




class LCDatasetOld(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH, class_type='single-label', pretrained_embeddings=None):
        """
        Dataset class for handling both input text and labels, with optional support for 
        pretrained embeddings and supervised ranges.

        Parameters:
        - texts: List of input text samples.
        - labels: Multi-label binary vectors or single-label indices.
        - tokenizer: Hugging Face tokenizer for text tokenization.
        - max_length: Maximum length of tokenized sequences.
        - class_type: 'multi-label' or 'single-label' classification type.
        - pretrained_embeddings: Tensor containing pretrained embeddings.
        - sup_range: Range of supervised embeddings within the concatenated embeddings.
        """
        self.texts = texts
        self.labels = labels                                    # Binary vectors (multi-label format) or indices (single-label)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_type = class_type
        self.pretrained_embeddings = pretrained_embeddings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get an individual sample from the dataset.

        Returns:
        - item: Dictionary containing tokenized inputs and labels.
        """
        text = self.texts[idx]
        labels = self.labels[idx] if self.labels is not None else [0]           # Default label if labels are missing
        
        """
        # Add debug statements
        print(f"Fetching item {idx}:")
        print(f"Labels: {labels}")
        """

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

        # Add pretrained embeddings if provided
        if self.pretrained_embeddings is not None:
            item["pretrained_embeddings"] = torch.tensor(
                self.pretrained_embeddings[idx], dtype=torch.float
            )

        return item



def compute_metrics(eval_pred, class_type='single-label', threshold=MC_THRESHOLD):
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

    if class_type in ['single-label', 'singlelabel']:
        # Convert predictions to class indices
        preds = np.argmax(predictions, axis=1)
        #labels = np.argmax(labels, axis=1)                                         # Convert one-hot to class indices if needed
    elif class_type in ['multi-label', 'multilabel']:
        # Threshold predictions for multi-label classification
        preds = (predictions > threshold).astype(int)
        labels = labels.astype(int)                                                 # Ensure labels are binary
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

    print(f'f1_micro: {f1_micro}, f1_macro: {f1_macro}, precision: {precision}, recall: {recall}')
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
    }




def compute_embedding_dimensions(model, num_classes, opt):
    """
    Compute the total embedding dimensions based on the model and supervised embeddings.

    Parameters:
    - model: Hugging Face transformer model (e.g., `AutoModel.from_pretrained(...)`)
    - num_classes: Number of classes in the dataset (used for supervised embeddings).
    - opt: Options object with configuration (e.g., whether supervised embeddings are enabled).

    Returns:
    - total_dimensions: Total embedding dimensions (pretrained + supervised if enabled).
    """
    # Extract the base model's hidden size
    base_dimensions = model.config.hidden_size

    # Add supervised dimensions if supervised embeddings are enabled
    supervised_dimensions = num_classes if opt.supervised else 0

    # Compute total dimensions
    total_dimensions = base_dimensions + supervised_dimensions

    print(f"Base dimensions: {base_dimensions}")
    if opt.supervised:
        print(f"Supervised dimensions (num_classes): {supervised_dimensions}")
    print(f"Total embedding dimensions: {total_dimensions}")

    return total_dimensions, base_dimensions, supervised_dimensions


def custom_data_collator(batch):
    """
    Custom data collator for handling variable-length sequences and labels.

    Parameters:
    - batch: List of individual samples from the dataset.

    Returns:
    - collated: Dictionary containing collated inputs and labels.
    """

    collated = {
        "input_ids": torch.stack([f["input_ids"] for f in batch]),
        "attention_mask": torch.stack([f["attention_mask"] for f in batch]),
    }

    if "labels" in batch[0]:
        collated["labels"] = torch.stack([f["labels"] for f in batch])
    else:
        print("Missing 'labels' key in batch!")  # Debugging missing labels
        collated["labels"] = torch.zeros(len(batch), dtype=torch.long)

    if "pretrained_embeddings" in batch[0]:
        collated["pretrained_embeddings"] = torch.stack([f["pretrained_embeddings"] for f in batch])

    #print(f"Batch sizes - input_ids: {collated['input_ids'].size()}, labels: {collated['labels'].size()}")

    return collated



# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Text Classification with Transformer Models.")
    
    parser.add_argument('--dataset', required=True, type=str, choices=['20newsgroups', 'reuters21578', 'bbc-news', 'rcv1'], help='Dataset to use')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--pretrained', type=str, choices=['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2', 'llama'], help='Pretrained embeddings')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--supervised', action='store_true', help='Use supervised embeddings')
    parser.add_argument('--dist', action='store_true', default=False, help='show class distribution plots')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=PATIENCE, help='Patience for early stopping')
    parser.add_argument('--log_file', type=str, default='../log/lc_nn_test.test', help='Path to log file')
    parser.add_argument('--force', action='store_true', default=False, help='do not check if this experiment has already been run')
    parser.add_argument('--dropprob', type=float, default=0.1, metavar='[0.0, 1.0]', help='dropout probability (default: 0.1)')
    parser.add_argument('--net', type=str, default='ff', metavar='str', help=f'net, defaults to ff (only supported option)')
    parser.add_argument('--learnable', type=int, default=0, metavar='int', help='dimension of the learnable embeddings (default 0)')
    parser.add_argument('--droptype', type=str, default='sup', metavar='DROPTYPE',
                        help=f'chooses the type of dropout to apply after the embedding layer. Default is "sup" which '
                             f'only applies to word-class embeddings (if present). Other options include "none" which '
                             f'does not apply dropout (same as "sup" with no supervised embeddings), "full" which '
                             f'applies dropout to the entire embedding, or "learn" that applies dropout only to the '
                             f'learnable embedding.')
    parser.add_argument('--tunable', action='store_true', default=True,
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

    print("\n\ttrans layer cake2")

    args = parse_args()
    print("args:", args)
    
    # Get the full model identifier and cache directory path for tokenizer/model
    model_name, model_path = get_model_identifier(args.pretrained)
    print("model_name:", model_name)
    print("model_path:", model_path)

    if (args.pretrained == 'llama'):
        args.llama_path = model_path
    elif (args.pretrained == 'gpt2'):
        args.gpt2_path = model_path
    elif (args.pretrained == 'bert'):
        args.bert_path = model_path
    elif (args.pretrained == 'roberta'):
        args.roberta_path = model_path
    elif (args.pretrained == 'distilbert'):
        args.distilbert_path = model_path
    elif (args.pretrained == 'xlnet'):
        args.xlnet_path = model_path
    else:
        raise ValueError("Unsupported pretrained model:", args.pretrained)
    
    print("args:", args)    

    # initialize logging and other system run variables
    already_modelled, logfile, method_name, pretrained, embeddings, emb_path, lm_type, mode, system = initialize_testing(args)

    # check to see if model params have been computed already
    if (already_modelled and not args.force):
        print(f'--- model {method_name} with embeddings {embeddings}, pretrained == {pretrained}, and wc_supervised == {args.supervised} for {args.dataset} already calculated, run with --force option to override. ---')
        exit(0)

    embedding_type = get_embedding_type(args)
    print("embedding_type:", embedding_type)
    print("embeddings:", embeddings)    
    print("embedding_path:", emb_path)

    device = get_device()
    print("device:", device)
    
    torch.manual_seed(args.seed)

    
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    #
    # Load the dataset and the associated (pretrained) embedding structures
    # to be fed into the model
    #                                                          
    lcd = loadpt_data(
        dataset=args.dataset,                       # Dataset name
        vtype=args.vtype,                           # Vectorization type
        pretrained=args.pretrained,                 # pretrained embeddings type
        embedding_path=emb_path,                    # path to pretrained embeddings
        emb_type=embedding_type                     # embedding type (word or token)
        )                                                

    print("loaded LCDataset object:", type(lcd))
    print("lcd:", lcd.show())

    pretrained_vectors = lcd.lcr_model
    pretrained_vectors.show()

    if (args.pretrained is None):
        pretrained_vectors = None

    if args.pretrained in ['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2', 'llama']:
        toke = lcd.tokenizer
        transformer_model = True
    else:
        toke = None
        transdformer_model = False

    word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset=lcd, pretrained=pretrained_vectors)
    print("word2index:", type(word2index), len(word2index))
    print("out_of_vocabulary:", type(out_of_vocabulary), len(out_of_vocabulary))

    """
    print("training and validation data split...")

    val_size = min(int(len(devel_index) * VAL_SIZE), 20000)                   # dataset split tr/val/test
    print("val_size:", val_size)

    train_index, val_index, ytr, yval = train_test_split(
        devel_index, lcd.devel_target, test_size=val_size, random_state=args.seed, shuffle=True
    )
    """

    print("lcd.devel_target:", type(lcd.devel_target), lcd.devel_target.shape)
    print("lcd.devel_target[0]:\n", type(lcd.devel_target[0]), lcd.devel_target[0])

    print("lcd.test_target:", type(lcd.test_target), lcd.test_target.shape)
    print("lcd.test_target[0]:\n", type(lcd.test_target[0]), lcd.test_target[0])
    
    print("lcd.devel_labelmatrix:", type(lcd.devel_labelmatrix), lcd.devel_labelmatrix.shape)
    print("lcd.devel_labelmatrix[0]:\n", type(lcd.devel_labelmatrix[0]), lcd.devel_labelmatrix[0])

    print("lcd.test_labelmatrix:", type(lcd.test_labelmatrix), lcd.test_labelmatrix.shape)
    print("lcd.test_labelmatrix[0]:\n", type(lcd.test_labelmatrix[0]), lcd.test_labelmatrix[0])


    #texts_train, texts_val, labels_train, labels_val = train_test_split(lcd.Xtr, lcd.devel_labelmatrix, test_size=VAL_SIZE, random_state=RANDOM_SEED)

    # Splitting the data using `train_test_split` and the sparse labels
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        lcd.Xtr, lcd.devel_labelmatrix, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    # Ensuring the same split indices for dense labels
    _, _, labels_train_dense, labels_val_dense = train_test_split(
        lcd.Xtr, lcd.ytr_encoded, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    print("texts_train:", type(texts_train), len(texts_train))
    print("texts_train[0]:", type(texts_train[0]), texts_train[0])
    print("labels_train:", type(labels_train), labels_train.shape)
    print("labels_traint[0]:", type(labels_train[0]), labels_train[0].shape, labels_train[0])

    print("texts_val:", type(texts_val), len(texts_val))
    print("texts_val[0]:", type(texts_val[0]), texts_val[0])
    print("labels_val:", type(labels_val), labels_val.shape)
    print("labels_val[0]:", type(labels_val[0]), labels_val[0].shape, labels_val[0])

    labels_test_sparse = lcd.test_labelmatrix
    labels_test_dense = lcd.yte_encoded
    test_data = lcd.Xte

    labels_test = labels_test_sparse
    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), labels_test.shape)
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])


    # Compute class weights based on the training set
    # Convert sparse matrix to dense array
    dense_labels = lcd.devel_labelmatrix.toarray()

    """
    class_weights = []
    for i in range(dense_labels.shape[1]):
        class_weight = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=dense_labels[:, i])
        class_weights.append(class_weight)

    # Convert to a tensor for PyTorch
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=opt.device)

    #class_weights = torch.tensor(class_weights, device=opt.device)
    #class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(lcd.devel_labelmatrix), y=lcd.devel_labelmatrix)
    #print("class_weights:", class_weights)
    """

    yte = lcd.test_target

    """
    print("ytr:", type(ytr), ytr.shape)
    print("ytr:\n", ytr)

    print("yte:", type(yte), yte.shape)
    print("yte:\n", yte)
    """

    vocabsize = len(word2index) + len(out_of_vocabulary)
    print("vocabsize:", {vocabsize})

    pretrained_embeddings, sup_range = embedding_matrix(lcd, pretrained_vectors, vocabsize, word2index, out_of_vocabulary, args)
    if pretrained_embeddings is not None:
        print("pretrained_embeddings:", type(pretrained_embeddings), pretrained_embeddings.shape)
    else:
        print("pretrained_embeddings: None")
    

    """
    # Load dataset and print class information
    (train_data, train_target_dense, train_target_sparse), (test_data, labels_test_dense, labels_test_sparse), num_classes, target_names, class_type = load_dataset(args.dataset)

    print("train_data:", type(train_data), len(train_data))
    print("train_data[0]:", type(train_data[0]), train_data[0])
    print("train_target_dense:", type(train_target_dense), len(train_target_dense))
    print("train_target_dense[0]:", type(train_target_dense[0]), train_target_dense[0].shape, train_target_dense[0])
    
    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test_dense:", type(labels_test_dense), len(labels_test_dense))
    print("labels_test_dense[0]:", type(labels_test_dense[0]), labels_test_dense[0].shape, labels_test_dense[0])
    
    print("\n")
    print("train_target_sparse:", type(train_target_sparse), train_target_sparse.shape)
    print("train_target_sparse[0]:", type(train_target_sparse[0]), train_target_sparse[0].shape, train_target_sparse[0])
    print("labels_test_sparse:", type(labels_test_sparse), labels_test_sparse.shape)
    print("labels_test_sparse[0]:", type(labels_test_sparse[0]), labels_test_sparse[0].shape, labels_test_sparse[0])
    print("\n")

    print("class_type:", class_type)
    print("num_classes:", num_classes)
    print("target_names:", target_names)
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("tokenizer:\n", tokenizer)

    # Get the pad_token_id
    pad_token_id = tokenizer.pad_token_id

    tok_vocab_size = len(tokenizer)
    print("tok_vocab_size:", tok_vocab_size)

    target_names = lcd.target_names
    print("target_names:", target_names)

    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names))
    # Set the pad_token_id in the model configuration
    hf_model.config.pad_token_id = tokenizer.pad_token_id
    hf_model.to(device)
    print("model:\n", hf_model)

    num_classes = lcd.nC
    print("num_classes:", num_classes)

    total_dims, pt_base_dims, supervised_dims = compute_embedding_dimensions(hf_model, num_classes, args)
    print("total_dims:", total_dims)
    print("pt_base_dims:", pt_base_dims)
    print("supervised_dims:", supervised_dims)
    


    """
    texts_train, texts_val, labels_train, labels_val = train_test_split(train_data, train_target_dense, test_size=VAL_SIZE, random_state=RANDOM_SEED)
    print("texts_train:", type(texts_train), len(texts_train))
    print("texts_train[0]:", type(texts_train[0]), texts_train[0])
    print("labels_train:", type(labels_train), len(labels_train))
    print("labels_traint[0]:", type(labels_train[0]), labels_train[0].shape, labels_train[0])

    print("texts_val:", type(texts_val), len(texts_val))
    print("texts_val[0]:", type(texts_val[0]), texts_val[0])
    print("labels_val:", type(labels_val), len(labels_val))
    print("labels_val[0]:", type(labels_val[0]), labels_val[0].shape, labels_val[0])

    labels_test = labels_test_dense
    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), len(labels_test))
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])
    """

    """
    texts_train, texts_val, labels_train, labels_val = train_test_split(train_data, train_target_sparse, test_size=VAL_SIZE, random_state=RANDOM_SEED)
    print("texts_train:", type(texts_train), len(texts_train))
    print("texts_train[0]:", type(texts_train[0]), texts_train[0])
    print("labels_train:", type(labels_train), labels_train.shape)
    print("labels_traint[0]:", type(labels_train[0]), labels_train[0].shape, labels_train[0])

    print("texts_val:", type(texts_val), len(texts_val))
    print("texts_val[0]:", type(texts_val[0]), texts_val[0])
    print("labels_val:", type(labels_val), labels_val.shape)
    print("labels_val[0]:", type(labels_val[0]), labels_val[0].shape, labels_val[0])

    labels_test = labels_test_sparse
    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), labels_test.shape)
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])

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

    if (class_type in ['single-label', 'singlelabel']):

        print("single label, converting target labels to to one-hot")

        label_binarizer = LabelBinarizer()        
        labels_train = label_binarizer.fit_transform(labels_train)
        labels_val = label_binarizer.transform(labels_val)
        labels_test = label_binarizer.transform(labels_test)

        print("labels_train:", type(labels_train), labels_train.shape)
        print("labels_train[0]:", type(labels_train[0]), labels_train[0])
        print("labels_val:", type(labels_val), labels_val.shape)
        print("labels_val[0]:", type(labels_val[0]), labels_val[0])
        print("labels_test:", type(labels_test), labels_test.shape)
        print("labels_test[0]:", type(labels_test[0]), labels_test[0])
    """

    """
    # Call `embedding_matrix` with the loaded model and tokenizer
    pretrained_embeddings = embedding_matrix_dot(
        model=hf_model,
        tokenizer=tokenizer,
        vocabsize=vec_vocab_size,
        word2index=vectorizer.vocabulary_,
        out_of_vocabulary=[],
        vectorized_training_data=Xtr,
        training_label_matrix=labels_train,
        opt=args
    )
    """

    class_type = lcd.class_type
    print("class_type:", class_type)

    print("labels_train_dense:", type(labels_train_dense), len(labels_train_dense))
    print("labels_train_dense[0]:", type(labels_train_dense[0]), labels_train_dense[0].shape, labels_train_dense[0])

    # Prepare datasets
    train_dataset = LCDataset(
        texts_train, 
        #labels_train, 
        labels_train_dense,
        tokenizer, 
        class_type=class_type, 
        #pretrained_embeddings=pretrained_embeddings
    )

    """
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        assert "labels" in sample, f"Missing 'labels' in sample {i}"
        print(f"Sample {i} is valid.")
    """

    print("labels_val_dense:", type(labels_val_dense), len(labels_val_dense))
    print("labels_val_dense[0]:", type(labels_val_dense[0]), labels_val_dense[0].shape, labels_val_dense[0])

    val_dataset = LCDataset(
        texts_val, 
        #labels_val,
        labels_val_dense, 
        tokenizer, 
        class_type=class_type, 
        #pretrained_embeddings=pretrained_embeddings
    )

    print("labels_test_dense:", type(labels_test_dense), len(labels_test_dense))
    print("labels_test_dense[0]:", type(labels_test_dense[0]), labels_test_dense[0].shape, labels_test_dense[0])

    test_dataset = LCDataset(
        test_data, 
        #labels_test,
        labels_test_dense, 
        tokenizer, 
        class_type=class_type, 
        #pretrained_embeddings=pretrained_embeddings
    )
    
    """
    # debug datasets
    print("\ntrain_dataset:", train_dataset)
    for i in range(3):                  # Sample a few batches
        sample = train_dataset[i]
        print("Sample Keys:", sample.keys())
        print("Pretrained Embeddings Shape:", sample["pretrained_embeddings"].shape)
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Attention Mask shape: {sample['attention_mask'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        
    print("\nval_dataset:", val_dataset)        
    for i in range(3):                  # Sample a few batches
        sample = val_dataset[i]
        print("Sample Keys:", sample.keys())
        print("Pretrained Embeddings Shape:", sample["pretrained_embeddings"].shape)
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Attention Mask shape: {sample['attention_mask'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")

    print("\ntest_dataset:", test_dataset)    
    for i in range(3):                  # Sample a few batches
        sample = test_dataset[i]
        print("Sample Keys:", sample.keys())
        print("Pretrained Embeddings Shape:", sample["pretrained_embeddings"].shape)
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Attention Mask shape: {sample['attention_mask'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
    """


    # Initialize the custom model
    lc_model = CustomTransformerForSequenceClassification(
        model_name=model_name, 
        num_labels=num_classes,
        pretrained_embeddings=pretrained_embeddings, 
        class_type=class_type,
        dropprob=args.dropprob,
        freeze_embeddings=True                                             # Optionally freeze the embeddings
    ).to(device)

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
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir='../log',
        run_name='trans_layer_cake',
        seed=args.seed,
        report_to="none"
    )

    # Trainer with custom data collator
    trainer = Trainer(
        model=lc_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, class_type),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # Train and evaluate
    trainer.train()

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:", test_results)

    # Predictions
    preds = trainer.predict(test_dataset)
    if class_type == 'single-label':
        y_pred = np.argmax(preds.predictions, axis=1)
    else:
        y_pred = (preds.predictions > MC_THRESHOLD).astype(int)

    """
    if (class_type in ['single-label', 'singlelabel']):
        # Convert one-hot encoded labels and predictions to class indices
        labels_test = np.argmax(labels_test, axis=1)  # Convert one-hot to class indices
    """
    
    """
    print("labels_test:", type(labels_test), labels_test.shape)
    print("y_pred:", type(y_pred), y_pred.shape)
    """

    print(classification_report(labels_test, y_pred, target_names=target_names, digits=4))

    macrof1, microf1, acc, h_loss, precision, recall, j_index = evaluation_nn(labels_test, y_pred, classification_type=class_type)
    print("\n--Layer Cake Metrics--")
    print(f"Macro-F1 = {macrof1:.4f}, Micro-F1 = {microf1:.4f}, Accuracy = {acc:.4f}, H-loss = {h_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, Jaccard = {j_index:.4f}")
    
    tend = time() - tinit

    measure_prefix = 'final'
    epoch = trainer.state.epoch
    
    logfile.insert(dimensions=pretrained_embeddings.shape, epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=macrof1, timelapse=tend)
    logfile.insert(dimensions=pretrained_embeddings.shape, epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=microf1, timelapse=tend)
    logfile.insert(dimensions=pretrained_embeddings.shape, epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    #logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

    logfile.insert(dimensions=pretrained_embeddings.shape, epoch=epoch, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dimensions=pretrained_embeddings.shape, epoch=epoch, measure='te-precision', value=precision, timelapse=tend)
    logfile.insert(dimensions=pretrained_embeddings.shape, epoch=epoch, measure='te-recall', value=recall, timelapse=tend)
    logfile.insert(dimensions=pretrained_embeddings.shape, epoch=epoch, measure='te-jacard-index', value=j_index, timelapse=tend)

    print("\n\t--- model training and evaluation complete---\n")


