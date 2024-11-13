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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1

from util.metrics import evaluation_nn
from util.common import initialize_testing, get_embedding_type

from embedding.supervised import get_supervised_embeddings



SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news"]


# Define a default model mapping (optional) to avoid invalid identifiers
MODEL_MAP = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "xlnet": "xlnet-base-uncased",
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
        print(f"num_classes: {len(target_names)}")
        print("class_names:", target_names)
        
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
        print(f"num_classes: {len(target_names)}")
        print("class_names:", target_names)
        
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
        test_labelled_doc = fetch_reuters21578(subset='test', data_path=data_path)

        train_data = train_labelled_docs.data
        train_target = train_labelled_docs.target
        test_data = test_labelled_doc.data
        test_target = test_labelled_doc.target

        class_type = 'multi-label'

        train_target, test_target, target_names = _label_matrix(train_target, test_target)

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        target_names = train_labelled_docs.target_names
        num_classes = len(target_names)
        print(f"num_classes: {len(target_names)}")
        print("class_names:", target_names)

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
        print(f"num_classes: {num_classes}")
        print("class_names:", target_names)
        
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



def show_multilabel_class_distribution(targets, target_names=None, dataset_name="Dataset"):
    """
    Show and plot the distribution of each class in a multi-label dataset.
    
    Parameters:
    - targets: List or ndarray of one-hot encoded targets. Each row corresponds to a document, and each column to a class.
    - target_names: List of class names, matching the number of columns in the targets.
    - dataset_name: Name of the dataset (for display purposes).
    """
    # Check that the number of columns in targets matches the length of target_names
    if targets.shape[1] != len(target_names):
        raise ValueError("The number of target columns does not match the number of target names.")

    # Convert targets to a DataFrame using target_names as columns
    target_df = pd.DataFrame(targets, columns=target_names)
    
    # Sum each column to get the count of documents per class
    class_counts = target_df.sum()
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.index, class_counts.values)
    plt.xticks(rotation=90)
    plt.xlabel("Class Labels")
    plt.ylabel("Document Count")
    plt.title(f"Class Distribution in {dataset_name}")
    plt.tight_layout()
    plt.show()

    return class_counts



def show_class_distribution(target, target_names=None, dataset_name="Dataset"):
    """
    Show and plot the distribution of each class in a single-label dataset.
    
    Parameters:
    - data: List or array of document data.
    - target: List or array of target labels.
    - target_names: Optional list of class names corresponding to each unique label in the target.
    - dataset_name: Name of the dataset (for display purposes).
    """
    # Calculate the class distribution
    class_counts = pd.Series(target).value_counts().sort_index()
    
    # Map the class indices to class names if provided
    if target_names:
        class_counts.index = [target_names[idx] for idx in class_counts.index]
    
    # Print class distribution
    print(f"\nClass distribution in {dataset_name}:")
    print(class_counts)
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.index, class_counts.values)
    plt.xticks(rotation=90)
    plt.xlabel("Class Labels")
    plt.ylabel("Document Count")
    plt.title(f"Class Distribution in {dataset_name}")
    plt.tight_layout()
    plt.show()
    
    return class_counts





def vectorize(train_data, val_data, test_data, vtype='tfidf'):

    print(f'vectorize(), vtype: {vtype}')

    if vtype == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=5, lowercase=False, sublinear_tf=True)
    elif vtype == 'count':
        vectorizer = CountVectorizer(min_df=5, lowercase=False)

    Xtr = vectorizer.fit_transform(train_data)
    Xval = vectorizer.transform(val_data)
    Xte = vectorizer.transform(test_data)

    Xtr.sort_indices()
    Xval.sort_indices()
    Xte.sort_indices()

    return vectorizer, Xtr, Xval, Xte



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
        #Xtr, _ = vectorize_data(word2index, dataset)                # Function to vectorize the dataset
        Xtr = vectorized_training_data
        #Ytr = dataset.devel_labelmatrix                             # Assuming devel_labelmatrix is the label matrix for training data
        Ytr = training_label_matrix
        F = get_supervised_embeddings(Xtr, Ytr,
                                      method=opt.supervised_method,
                                      max_label_space=opt.max_label_space,
                                      dozscore=(not opt.nozscore))
        
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

    return pretrained_embeddings, sup_range





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
    parser.add_argument('--dropprob', type=float, default=0.5, metavar='[0.0, 1.0]', help='dropout probability (default: 0.5)')
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


class LCDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, pretrained_embeddings=None, max_length=512, class_type='multi-label'):
        """
        Initializes the LCDataset with texts, labels, and additional pretrained embeddings.

        Parameters:
        - texts: List of input texts.
        - labels: Corresponding labels for each text.
        - tokenizer: Hugging Face tokenizer for tokenizing input texts.
        - pretrained_embeddings: Pretrained embedding matrix (e.g., from embedding_matrix function).
        - max_length: Maximum length for padding/truncating tokenized sequences.
        - class_type: Type of classification ('single-label' or 'multi-label').
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.pretrained_embeddings = pretrained_embeddings  # Pretrained embeddings matrix
        self.max_length = max_length
        self.class_type = class_type

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Convert labels based on the classification type
        if self.class_type == 'single-label':
            item["labels"] = torch.tensor(labels, dtype=torch.long)  # Single-label classification
        else:
            item["labels"] = torch.tensor(labels, dtype=torch.float)  # Multi-label classification

        # Incorporate pretrained embeddings if available
        if self.pretrained_embeddings is not None:
            # Get token IDs from the tokenizer output
            input_ids = item["input_ids"]
            
            # Use input_ids to index into pretrained embeddings
            # Shape: (seq_len, embedding_dim)
            word_embeddings = self.pretrained_embeddings[input_ids]
            
            # Add word embeddings to the item for use in the model
            item["word_embeddings"] = word_embeddings  # Shape: (seq_len, embedding_dim)

        return item



class LCDatasetOld(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, class_type='multi-label'):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_type = class_type

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Modify: Handle labels based on classification type
        if self.class_type == 'single-label':
            item["labels"] = torch.tensor(labels, dtype=torch.long)  # Single-label
        else:
            item["labels"] = torch.tensor(labels, dtype=torch.float)  # Multi-label
        return item


# Metrics function
def compute_metrics(pred, class_type='single-label', threshold=0.5):
    
    labels = pred.label_ids
    
    if class_type == 'single-label':
        # Single-label classification: use argmax to get class predictions
        preds = np.argmax(pred.predictions, axis=1)
    else:
        # Multi-label classification: threshold predictions to get binary matrix
        preds = pred.predictions > threshold            # Adjust threshold for multi-label classification
    
    f1_micro = f1_score(labels, preds, average='micro', zero_division=1)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=1)

    return {'f1_micro': f1_micro, 'f1_macro': f1_macro}



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

    #pretrained, pretrained_vectors = load_pretrained_embeddings(opt.pretrained, opt)

    embedding_type = get_embedding_type(args)
    print("embedding_type:", embedding_type)
    print("embeddings:", embeddings)    
    print("embedding_path:", emb_path)

    device = get_device()
    print("device:", device)
    
    torch.manual_seed(args.seed)

    # Load dataset and print class information
    (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type = load_dataset(args.dataset)

    print("num_classes:", num_classes)
    print("target_names:", target_names)

    print("train_data:", type(train_data), len(train_data))
    print("train_data[0]:", type(train_data[0]), train_data[0])
    print("train_target:", type(train_target), len(train_target))
    print("train_target[0]:", type(train_target[0]), train_target[0].shape, train_target[0])

    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("test_target:", type(test_target), len(test_target))
    print("test_target[0]:", type(test_target[0]), test_target[0].shape, test_target[0])

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("tokenizer:", tokenizer)

    vocab_size = len(tokenizer)
    print("vocab_size:", vocab_size)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names))
    # Set the pad_token_id in the model configuration
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    print("model:", model)

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

    vectorizer, Xtr, Xval, Xte = vectorize(texts_train, texts_val, test_data, vtype=args.vtype)
    print("vectorizer:", vectorizer)

    print("Xtr:", type(Xtr), Xtr.shape)
    print("Xtr[0]:", type(Xtr[0]), Xtr[0].shape, Xtr[0])
    
    print("Xval:", type(Xval), Xval.shape)
    print("Xval[0]:", type(Xval[0]), Xval[0].shape, Xval[0])
    
    print("Xte:", type(Xte), Xte.shape)
    print("Xte[0]:", type(Xte[0]), Xte[0].shape, Xte[0])

    # Call `embedding_matrix` with the loaded model and tokenizer
    pretrained_embeddings, sup_range = embedding_matrix(
        model=model,
        tokenizer=tokenizer,
        vocabsize=vocab_size,
        word2index=tokenizer.get_vocab(),
        out_of_vocabulary=[],
        vectorized_training_data=Xtr,
        training_label_matrix=labels_train,
        opt=args
    )

    # Prepare datasets
    train_dataset = LCDataset(texts_train, labels_train, tokenizer, class_type=class_type)
    val_dataset = LCDataset(texts_val, labels_val, tokenizer, class_type=class_type)
    test_dataset = LCDataset(test_data, test_target, tokenizer, class_type=class_type)
    
    tinit = time()
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='../out',
        eval_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir='../log',
        run_name='layer_cake'
    )

    # Early stopping variables
    patience = args.patience
    best_f1_macro = 0
    patience_counter = 0

    # Define trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: compute_metrics(pred, class_type=class_type, threshold=MC_THRESHOLD),
    )

    # Custom training loop with early stopping
    for epoch in range(training_args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{training_args.num_train_epochs}")
        trainer.train()
        
        # Evaluate on validation data
        results = trainer.evaluate(val_dataset)
        val_f1_macro = results["eval_f1_macro"]
        print(f"Validation F1 Macro: {val_f1_macro:.4f}")

        # Check early stopping condition
        if val_f1_macro > best_f1_macro:
            best_f1_macro = val_f1_macro
            patience_counter = 0  # Reset patience counter if improvement
        else:
            patience_counter += 1  # Increment if no improvement
            print(f"Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Final evaluation on test set
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:", test_results)
    preds = trainer.predict(test_dataset)

    if class_type == 'single-label':
        # Single-label classification: use argmax to get class predictions
        y_pred = np.argmax(preds.predictions, axis=1)
    else:
        # Multi-label classification: threshold predictions to get binary matrix
        y_pred = (preds.predictions > 0.5).astype(int)

    print(classification_report(test_target, y_pred, target_names=target_names, digits=4))

    macrof1, microf1, acc, h_loss, precision, recall, j_index = evaluation_nn(test_target, y_pred, classification_type=class_type)
    print("\n--Layer Cake Metrics--")
    print(f"Macro-F1 = {macrof1:.4f}, Micro-F1 = {microf1:.4f}, Accuracy = {acc:.4f}, H-loss = {h_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, Jaccard = {j_index:.4f}")
    
    tend = time() - tinit

    embedding_size = -1
    measure_prefix = 'final'
    
    logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=macrof1, timelapse=tend)
    logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=microf1, timelapse=tend)
    logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    #logfile.insert(dimensions=embedding_size, epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

    logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-precision', value=precision, timelapse=tend)
    logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-recall', value=recall, timelapse=tend)
    logfile.insert(dimensions=embedding_size, epoch=epoch, measure='te-jacard-index', value=j_index, timelapse=tend)

    print("\n\t--- model training and evaluation complet---\n")


