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

from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1

from util.metrics import evaluation_nn
from util.common import initialize_testing, get_embedding_type

from embedding.supervised import get_supervised_embeddings

from scipy.sparse import csr_matrix




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

        train_target, test_target, target_names = _label_matrix(train_target, test_target)

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        target_names = train_labelled_docs.target_names
        num_classes = len(target_names)
        #print(f"num_classes: {len(target_names)}")
        #print("class_names:", target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type
    
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




def embedding_matrices_split(model, tokenizer, vocabsize, word2index, out_of_vocabulary, vectorized_training_data, training_label_matrix, opt):
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

    print(f'\nembedding_matrices(): opt.pretrained: {opt.pretrained}, vocabsize: {vocabsize}, opt.supervised: {opt.supervised}')
          
    pretrained_embeddings = []
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
    - combined_embeddings: Tensor containing combined pretrained and supervised embeddings.
    - sup_range: Range in the embedding matrix where supervised embeddings are located.
    - total_dim: Total embedding dimensions (pretrained + supervised).
    """
    print(f'\n\tembedding_matrix(): opt.pretrained: {opt.pretrained},  vocabsize: {vocabsize}, opt.supervised: {opt.supervised}')
          
     # Get the embedding layer for pretrained embeddings
    embedding_layer = model.get_input_embeddings()                          # Works across models like BERT, RoBERTa, DistilBERT, GPT, XLNet, LLaMA
    embedding_dim = embedding_layer.embedding_dim
    pretrained_matrix = torch.zeros((vocabsize, embedding_dim))             # Initialize pretrained embedding matrix
    wce_matrix = None
    sup_range = None
    
    # If pretrained embeddings are needed, populate pretrained_matrix 
    # with pretrained model embeddings for each word in word2index
    if opt.pretrained:

        for word, idx in word2index.items():
            token_id = tokenizer.convert_tokens_to_ids(word)
            if token_id is not None and token_id < embedding_layer.num_embeddings:
                with torch.no_grad():
                    pretrained_matrix[idx] = embedding_layer.weight[token_id].cpu()
            else:
                out_of_vocabulary.append(word)

        print(f"[pretrained-matrix]: {type(pretrained_matrix)}, {pretrained_matrix.shape}")

    # If supervised embeddings are needed
    if opt.supervised:
        print(f'computing supervised embeddings...')
        
        Xtr = vectorized_training_data
        Ytr = training_label_matrix
        #print("\tXtr:", type(Xtr), Xtr.shape)
        #print("\tYtr:", type(Ytr), Ytr.shape)

        WCE = get_supervised_embeddings(
            Xtr, 
            Ytr,
            method=opt.supervised_method,
            max_label_space=opt.max_label_space,
            dozscore=(not opt.nozscore),
            transformers=True
        )
        
        # Adjust supervised embedding matrix to match vocabulary size
        num_missing_rows = vocabsize - WCE.shape[0]
        WCE = np.vstack((WCE, np.zeros((num_missing_rows, WCE.shape[1]))))
        wce_matrix = torch.from_numpy(WCE).float()
        print(f'[supervised-matrix]: {type(wce_matrix)}, {wce_matrix.shape}')

        # Concatenate supervised embeddings
        offset = pretrained_matrix.shape[1]
        sup_range = [offset, offset + wce_matrix.shape[1]]
        print("sup_range: ", sup_range)


    # Concatenate pretrained and supervised embeddings
    if wce_matrix is not None:
        combined_embeddings = torch.cat([pretrained_matrix, wce_matrix], dim=1)
    else:
        combined_embeddings = pretrained_matrix

    total_dim = combined_embeddings.shape[1]
    print(f"[final combined_embeddings]: {combined_embeddings.shape}")

    return combined_embeddings, sup_range, total_dim




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
    embedding_layer = model.get_input_embeddings()  # Works across models like BERT, RoBERTa, DistilBERT, GPT, XLNet, LLaMA
    embedding_dim = embedding_layer.embedding_dim  # Pretrained embedding dimension
    pretrained_matrix = torch.zeros((vocabsize, embedding_dim))  # Initialize pretrained embedding matrix
    wce_matrix = None
    OOV = 0

    # If pretrained embeddings are needed, populate pretrained_matrix 
    # with pretrained model embeddings for each word in word2index
    if opt.pretrained:

        for word, idx in word2index.items():
            token_id = tokenizer.convert_tokens_to_ids(word)
            if token_id is not None and token_id < embedding_layer.num_embeddings:
                with torch.no_grad():
                    pretrained_matrix[idx] = embedding_layer.weight[token_id].cpu()
            else:
                out_of_vocabulary.append(word)
                OOV += 1
        
        print("OOV words:", OOV)
        print(f'\n\t[pretrained-matrix]: {type(pretrained_matrix)}, {pretrained_matrix.shape}')


    # Compute Word-Class Embeddings (WCE) if supervised embeddings are needed
    if opt.supervised:
        print(f'Computing supervised embeddings...')
        
        Xtr = vectorized_training_data
        Ytr = training_label_matrix

        """
        print("Xtr:", type(Xtr), Xtr.shape)
        print("Xtr[0]:", type(Xtr[0]), Xtr[0])
        print("Ytr:", type(Ytr), Ytr.shape)
        print("Ytr[0]:", type(Ytr[0]), Ytr[0])
        """

        WCE = get_supervised_embeddings(
            Xtr, 
            Ytr,
            method=opt.supervised_method,
            max_label_space=opt.max_label_space,
            dozscore=(not opt.nozscore),
            transformers=True
        )
        
        # Adjust WCE matrix to match vocabulary size
        num_missing_rows = vocabsize - WCE.shape[0]
        WCE = np.vstack((WCE, np.zeros((num_missing_rows, WCE.shape[1]))))
        wce_matrix = torch.from_numpy(WCE).float()
        print(f'\n\t[supervised-matrix]: {type(wce_matrix)}, {wce_matrix.shape}')

        # Project WCE matrix into the pretrained embedding space via matrix multiplication
        # Shape: (vocabsize, num_classes) @ (num_classes, embedding_dim) -> (vocabsize, embedding_dim)
        if wce_matrix.shape[1] == pretrained_matrix.shape[1]:
            final_matrix = wce_matrix @ pretrained_matrix.t()
        else:
            projection_layer = nn.Linear(wce_matrix.shape[1], embedding_dim, bias=False)
            torch.nn.init.xavier_uniform_(projection_layer.weight)
            wce_matrix = projection_layer(wce_matrix)
            final_matrix = wce_matrix                           # No explicit addition, using WCE projection alone

    else:
        # If no WCE, return pretrained embeddings as the final matrix
        final_matrix = pretrained_matrix

    print(f'\t[final matrix]: {type(final_matrix)}, {final_matrix.shape}')
    
    return final_matrix



class LCDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH, class_type='single-label', embeddings=None):
        """
        Dataset class for handling both input text and labels, with optional support for 
        pretrained embeddings and supervised ranges.

        Parameters:
        - texts: List of input text samples.
        - labels: Multi-label binary vectors or single-label indices.
        - tokenizer: Hugging Face tokenizer for text tokenization.
        - max_length: Maximum length of tokenized sequences.
        - class_type: 'multi-label' or 'single-label' classification type.
        - embeddings: Tensor containing computed embeddings for model.
        """
        self.texts = texts
        self.labels = labels                                    # Binary vectors (multi-label format) or indices (single-label)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_type = class_type
        self.embeddings = embeddings

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
        if self.class_type in ['single-label', 'singlelabel']:
            item["labels"] = torch.tensor(labels, dtype=torch.long)
        else:
            item["labels"] = torch.tensor(labels, dtype=torch.float)

        # Add embeddings
        item["embeddings"] = torch.tensor(self.embeddings[idx], dtype=torch.float)

        return item



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

    if class_type in ['single-label', 'singlelabel']:
        # Convert predictions to class indices
        preds = np.argmax(predictions, axis=1)
        #labels = np.argmax(labels, axis=1)                              # Convert one-hot to class indices if needed
    elif class_type in ['multi-label', 'multilabel']:
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

    if "embeddings" in batch[0]:
        collated["embeddings"] = torch.stack([f["embeddings"] for f in batch])

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

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("tokenizer:\n", tokenizer)

    # Get the pad_token_id
    pad_token_id = tokenizer.pad_token_id

    tok_vocab_size = len(tokenizer)
    print("tok_vocab_size:", tok_vocab_size)

    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names))
    # Set the pad_token_id in the model configuration
    hf_model.config.pad_token_id = tokenizer.pad_token_id
    hf_model.to(device)
    print("model:\n", hf_model)

    total_dims, pt_base_dims, supervised_dims = compute_embedding_dimensions(hf_model, num_classes, args)
    print("total_dims:", total_dims)
    print("pt_base_dims:", pt_base_dims)
    print("supervised_dims:", supervised_dims)

    # Split train into train and validation
    texts_train, texts_val, labels_train, labels_val = train_test_split(train_data, train_target, test_size=VAL_SIZE, random_state=RANDOM_SEED)

    print("texts_train:", type(texts_train), len(texts_train))
    print("texts_train[0]:", type(texts_train[0]), texts_train[0])
    print("labels_train:", type(labels_train), len(labels_train))
    print("labels_traint[0]:", type(labels_train[0]), labels_train[0].shape, labels_train[0])

    print("texts_val:", type(texts_val), len(texts_val))
    print("texts_val[0]:", type(texts_val[0]), texts_val[0])
    print("labels_val:", type(labels_val), len(labels_val))
    print("labels_val[0]:", type(labels_val[0]), labels_val[0].shape, labels_val[0])

    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), len(labels_test))
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

    # Call `embedding_matrix` with the loaded model and tokenizer
    pt_embeddings, wce_embeddings = embedding_matrices_split(
        model=hf_model,
        tokenizer=tokenizer,
        vocabsize=vec_vocab_size,
        word2index=vectorizer.vocabulary_,
        out_of_vocabulary=[],
        vectorized_training_data=Xtr,
        training_label_matrix=labels_train,
        opt=args
    )

    print("pt_embeddings:", type(pt_embeddings), pt_embeddings.shape)
    if (wce_embeddings is not None):
        print("wce_embeddings:", type(wce_embeddings), wce_embeddings.shape)

    # Prepare datasets
    train_dataset = LCDataset(
        texts_train, 
        labels_train, 
        tokenizer, 
        class_type=class_type, 
        embeddings=pt_embeddings
    )

    val_dataset = LCDataset(
        texts_val, 
        labels_val, 
        tokenizer, 
        class_type=class_type, 
        embeddings=pt_embeddings
    )

    test_dataset = LCDataset(
        test_data, 
        labels_test, 
        tokenizer, 
        class_type=class_type, 
        embeddings=pt_embeddings
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
        model=hf_model,
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
    if class_type in ['single-label', 'singlelabel']:
        y_pred = np.argmax(preds.predictions, axis=1)
    else:
        y_pred = (preds.predictions > MC_THRESHOLD).astype(int)
    
    print("labels_test:", type(labels_test), len(labels_test))
    print("y_pred:", type(y_pred), len(y_pred))

    print(classification_report(labels_test, y_pred, target_names=target_names, digits=4))

    macrof1, microf1, acc, h_loss, precision, recall, j_index = evaluation_nn(labels_test, y_pred, classification_type=class_type)
    print("\n--Layer Cake Metrics--")
    print(f"Macro-F1 = {macrof1:.4f}, Micro-F1 = {microf1:.4f}, Accuracy = {acc:.4f}, H-loss = {h_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, Jaccard = {j_index:.4f}")
    
    tend = time() - tinit

    measure_prefix = 'final'
    epoch = trainer.state.epoch
    
    logfile.insert(dimensions=embeddings.shape, epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=macrof1, timelapse=tend)
    logfile.insert(dimensions=embeddings.shape, epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=microf1, timelapse=tend)
    logfile.insert(dimensions=embeddings.shape, epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    #logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

    logfile.insert(dimensions=embeddings.shape, epoch=epoch, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dimensions=embeddings.shape, epoch=epoch, measure='te-precision', value=precision, timelapse=tend)
    logfile.insert(dimensions=embeddings.shape, epoch=epoch, measure='te-recall', value=recall, timelapse=tend)
    logfile.insert(dimensions=embeddings.shape, epoch=epoch, measure='te-jacard-index', value=j_index, timelapse=tend)

    print("\n\t--- model training and evaluation complete---\n")


