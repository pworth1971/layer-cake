import os
import re
import string
import random
import json
import torch
import unicodedata
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm

from collections import defaultdict

import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoTokenizer

from joblib import Parallel, delayed

# Custom
from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1
from data.arxiv_reader import fetch_arxiv, sci_field_map

from util.common import PICKLE_DIR, VECTOR_CACHE, DATASET_DIR, OUT_DIR, preprocess, get_model_identifier


# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news", "ohsumed", "imdb", "arxiv", "arxiv_protoformer"]

#
# Disable Hugging Face tokenizers parallelism to avoid fork issues
#
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TEST_SIZE = 0.175                       # test size for train/test split
VAL_SIZE = 0.175                        # percentage of data to be set aside for model validation

NUM_DL_WORKERS = 3                      # number of workers to handle DataLoader tasks

RANDOM_SEED = 29

MAX_TOKEN_LENGTH = 512              # Maximum token length for transformer models models
#
# ------------------------------------------------------------------------------------------------------------------------------------------------------


def get_dataset_data(dataset_name, seed=RANDOM_SEED, pickle_dir=PICKLE_DIR):
    """
    Load dataset data from a pickle file if it exists; otherwise, call the dataset loading method,
    save the returned data to a pickle file, and return the data.

    Parameters:
    - dataset_name (str): Name of the dataset.
    - seed (int): Random seed for reproducibility.
    - pickle_dir (str): Directory where the pickle file is stored.

    Returns:
    - train_data: Training data.
    - train_target: Training labels.
    - test_data: Test data.
    - labels_test: Test labels.
    - num_classes: Number of classes in the dataset.
    - target_names: Names of the target classes.
    - class_type: Classification type (e.g., 'multi-label', 'single-label').
    """

    print(f'get_dataset_data()... dataset_name: {dataset_name}, pickle_dir: {pickle_dir}, seed: {seed}')

    pickle_file = os.path.join(pickle_dir, f"trans_lc.{dataset_name}.pickle")
    
    # Ensure the pickle directory exists
    os.makedirs(pickle_dir, exist_ok=True)

    if os.path.exists(pickle_file):
        print(f"Loading dataset from pickle file: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
    else:
        print(f"Pickle file not found. Loading dataset using `trans_lc_load_dataset` for {dataset_name}...")
        
        data = trans_lc_load_dataset(
            name=dataset_name, 
            seed=seed) 
        
        # Save the dataset to a pickle file
        print(f"Saving dataset to pickle file: {pickle_file}")
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)

    # Unpack and return the data
    (train_data, train_target), (test_data, labels_test), num_classes, target_names, class_type = data

    return train_data, train_target, test_data, labels_test, num_classes, target_names, class_type




# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Load dataset method for transformer based neural models
#
def trans_lc_load_dataset(name, seed):

    print(f'trans_lc_load_dataset(): dataset: {name}, seed: {seed}...')

    if name == "20newsgroups":

        import os

        data_path = os.path.join(DATASET_DIR, '20newsgroups')    
        print("data_path:", data_path)  

        class_type = 'single-label'

        metadata = ('headers', 'footers', 'quotes')
        train_data = fetch_20newsgroups(subset='train', remove=metadata, data_home=data_path, random_state=seed)
        test_data = fetch_20newsgroups(subset='test', remove=metadata, data_home=data_path, random_state=seed)

        target_names = list(set(train_data.target_names))  # Ensures unique class names
        num_classes = len(target_names)
  
        train_data_processed = preprocess(
            pd.Series(train_data.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False
            )

        test_data_processed = preprocess(
            pd.Series(test_data.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False
            )

        return (train_data_processed, train_data.target), (test_data_processed, test_data.target), num_classes, target_names, class_type
        

    elif name == "reuters21578":
        
        import os

        data_path = os.path.join(DATASET_DIR, 'reuters21578')    
        print("data_path:", data_path)  

        class_type = 'multi-label'

        train_labelled_docs = fetch_reuters21578(subset='train', data_path=data_path)
        test_labelled_docs = fetch_reuters21578(subset='test', data_path=data_path)

        train_data = preprocess(
            pd.Series(train_labelled_docs.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)
        
        test_data = preprocess(
            pd.Series(test_labelled_docs.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)
        
        """
        train_data = preprocess_text(train_labelled_docs.data)
        test_data = preprocess_text(list(test_labelled_docs.data))
        """

        train_target = train_labelled_docs.target
        test_target = test_labelled_docs.target
        
        train_target, test_target, labels = _label_matrix(train_target, test_target)

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        target_names = train_labelled_docs.target_names
        num_classes = len(target_names)
        
        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type
        

    elif name == "ohsumed":

        import os
        
        data_path = os.path.join(DATASET_DIR, 'ohsumed50k')

        devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        test = fetch_ohsumed50k(subset='test', data_path=data_path)

        """
        train_data = preprocess_text(devel.data)
        test_data = preprocess_text(test.data)
        """
        
        """
        train_data = _preprocess(pd.Series(devel.data), remove_punctuation=False)
        test_data = _preprocess(pd.Series(test.data), remove_punctuation=False)
        """

        train_data = preprocess(
            pd.Series(devel.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)
        
        test_data = preprocess(
            pd.Series(test.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)

        train_target, test_target = devel.target, test.target
        class_type = 'multi-label'
        
        train_target, test_target, target_names = _label_matrix(train_target, test_target)

        target_names = devel.target_names

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        num_classes = len(target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type
    

    elif name == "bbc-news":

        import os

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
            random_state = seed,
        )

        """
        train_data = preprocess_text(train_data.tolist())
        test_data = preprocess_text(test_data.tolist())
        """

        train_data = preprocess(
            train_data, 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )

        test_data = preprocess(
            test_data, 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )
        
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

        return (train_data, train_target_encoded), (test_data, test_target_encoded), num_classes, target_names, class_type


    elif name == "rcv1":

        import os
        
        data_path = os.path.join(DATASET_DIR, 'rcv1')
        
        class_type = 'multi-label'

        """
        from sklearn.datasets import fetch_rcv1

        devel_data, devel_target = fetch_rcv1(
            subset='train', 
            data_home=data_path, 
            download_if_missing=True,
            return_X_y=True,
            #shuffle-True
            )

        test_data, test_target = fetch_rcv1(
            subset='test', 
            data_home=data_path, 
            download_if_missing=True,
            return_X_y=True,
            #shuffle-True
            )
       
        train_data = preprocess(
            pd.DataFrame(devel_data.toarray()).apply(lambda x: ' '.join(x.astype(str)), axis=1),
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
        )

        test_data = preprocess(
            pd.DataFrame(test_data.toarray()).apply(lambda x: ' '.join(x.astype(str)), axis=1),
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
        )
 
        """

        devel = fetch_RCV1(subset='train', data_path=data_path)
        test = fetch_RCV1(subset='test', data_path=data_path)
        
        train_data = preprocess(
            pd.Series(devel.data),
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
        )

        test_data = preprocess(
            pd.Series(test.data),
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
        )

        train_target, test_target = devel.target, test.target
                
        train_target, test_target, target_names = _label_matrix(train_target, test_target)

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        num_classes = len(target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type


    elif name == 'imdb':

        from datasets import load_dataset
        import os

        class_type = 'single-label'

        data_path = os.path.join(DATASET_DIR, 'imdb')

        # Load IMDB dataset using the Hugging Face Datasets library
        imdb_dataset = load_dataset('imdb', cache_dir=data_path)

        #train_data = preprocess_text(imdb_dataset['train']['text'])
        #train_data = _preprocess(imdb_dataset['train']['text'], remove_punctuation=False)
        train_data = imdb_dataset['train']['text']

        # Split dataset into training and test data
        #train_data = imdb_dataset['train']['text']
        train_target = np.array(imdb_dataset['train']['label'], dtype=np.int64)  # Convert to numpy array of type int64

        #test_data = imdb_dataset['test']['text']
        #test_data = preprocess_text(imdb_dataset['test']['text'])
        #test_data = _preprocess(imdb_dataset['test']['text'], remove_punctuation=False)
        test_data = imdb_dataset['test']['text']

        test_target = np.array(imdb_dataset['test']['label'], dtype=np.int64)  # Convert to numpy array of type int64

        train_data = preprocess(
            pd.Series(train_data), 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )

        test_data = preprocess(
            pd.Series(test_data), 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )

        # Define target names
        target_names = ['negative', 'positive']
        num_classes = len(target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type


    elif name == 'arxiv_protoformer':

        import os

        class_type = 'single-label'

        #print("loading data...")

        #
        # dataset from https://paperswithcode.com/dataset/arxiv-10
        #
        data_path = os.path.join(DATASET_DIR, 'arxiv_protoformer')

        file_path = data_path + '/arxiv100.csv'
        print("file_path:", file_path)

        # Load datasets
        full_data_set = pd.read_csv(file_path)
        
        target_names = full_data_set['label'].unique()
        num_classes = len(full_data_set['label'].unique())
        print(f"num_classes: {len(target_names)}")
        print("target_names:", target_names)
        
        papers_dataframe = pd.DataFrame({
            'title': full_data_set['title'],
            'abstract': full_data_set['abstract'],
            'label': full_data_set['label']
        })

        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())

        print("proeprocessing...")
        """

        # preprocess text
        #papers_dataframe['abstract'] = papers_dataframe['abstract'].apply(lambda x: x.replace("\n",""))
        #papers_dataframe['abstract'] = papers_dataframe['abstract'].apply(lambda x: x.strip())
        papers_dataframe['text'] = papers_dataframe['title'] + '. ' + papers_dataframe['abstract']

        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())
        """

        # Ensure the 'categories' column value counts are calculated and indexed properly
        categories_counts = papers_dataframe['label'].value_counts().reset_index(name="count")

        papers_dataframe['text'] = preprocess(
            papers_dataframe['text'],
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False,
            remove_special_chars=True
        )
        
        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())        
        """

        # we split the train data into train and test here because that is
        # not done for us with the BBC News dataset (Test data is not labeled)
        train_data, test_data, train_target, test_target = train_test_split(
            papers_dataframe['text'], 
            papers_dataframe['label'], 
            train_size = 1-TEST_SIZE, 
            random_state = seed,
        )
        
        """
        print("train_data:", type(train_data), train_data.shape)
        print("train_target:", type(train_target), train_target.shape)
        print("test_data:", type(test_data), test_data.shape)
        print("test_target:", type(test_target), test_target.shape)
        """

        #
        # set up label targets
        # Convert target labels to 1D arrays
        train_target_arr = np.array(train_target)                   # Flattening the training labels into a 1D array
        test_target_arr = np.array(test_target)                     # Flattening the test labels into a 1D array

        # Use LabelEncoder to encode the labels into label IDs
        label_encoder = LabelEncoder()
        label_encoder.fit(train_target_arr)  # Fit on training labels

        # Transform labels to numeric IDs
        train_target_encoded = label_encoder.transform(train_target_arr)
        test_target_encoded = label_encoder.transform(test_target_arr)

        return (train_data.tolist(), train_target_encoded), (test_data.tolist(), test_target_encoded), num_classes, target_names, class_type


    elif name == 'arxiv':

        import os

        class_type = 'multi-label'

        data_path = os.path.join(DATASET_DIR, 'arxiv')

        xtrain, ytrain, xtest, ytest, target_names, num_classes = fetch_arxiv(data_path=data_path, test_size=TEST_SIZE, seed=seed)

        return (xtrain, ytrain), (xtest, ytest), num_classes, target_names, class_type
    
    else:
        raise ValueError("Unsupported dataset:", name)


# ------------------------------------------------------------------------------------------------------------------------------------------------


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



def show_class_distribution(labels, target_names, class_type, dataset_name, display_mode='text'):
    """
    Visualize the class distribution and compute class weights for single-label or multi-label datasets.
    Supports graphical display or text-based summary for remote sessions.

    Parameters:
    - labels: The label matrix (numpy array or csr_matrix for multi-label) or 1D array for single-label.
    - target_names: A list of class names corresponding to the labels.
    - class_type: A string, either 'single-label' or 'multi-label', to specify the classification type.
    - dataset_name: A string representing the name of the dataset.
    - display_mode: A string, 'both', 'text', or 'graph'. Controls whether to display a graph, text, or both.

    Returns:
    - class_weights: A list of computed weights for each class, useful for loss functions.
    """
    # Handle sparse matrix for multi-label case
    if isinstance(labels, csr_matrix):
        labels = labels.toarray()

    # Calculate class counts differently for single-label vs multi-label
    if class_type == 'single-label':
        # For single-label, count occurrences of each class
        unique_labels, class_counts = np.unique(labels, return_counts=True)
    elif class_type == 'multi-label':
        # For multi-label, sum occurrences across the columns
        class_counts = np.sum(labels, axis=0)
        unique_labels = np.arange(len(class_counts))
    else:
        raise ValueError(f"Unsupported class_type: {class_type}")

    # Total number of samples
    total_samples = labels.shape[0]

    # Calculate class weights (inverse frequency)
    class_weights = [total_samples / (len(class_counts) * count) if count > 0 else 0 for count in class_counts]

    # Normalize weights
    max_weight = max(class_weights) if max(class_weights) > 0 else 1
    class_weights = [w / max_weight for w in class_weights]

    # Display graphical output if requested
    if display_mode in ('both', 'graph'):
        plt.figure(figsize=(14, 8))
        plt.bar(target_names, class_counts, color='blue', alpha=0.7)
        plt.xlabel('Classes', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title(f'Class Distribution in {dataset_name} ({len(target_names)} Classes)', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        try:
            plt.show()
        except:
            print("Unable to display the graph (e.g., no GUI backend available). Switching to text-only output.")

    # Display text-based summary if requested
    if display_mode in ('both', 'text'):
        print(f"\n\tClass Distribution and Weights in {dataset_name}:")
        for idx, (class_name, count, weight) in enumerate(zip(target_names, class_counts, class_weights)):
            print(f"{idx:2d}: {class_name:<20} Count: {count:5d}, Weight: {weight:.4f}")

    #print("\tclass weights:\n", class_weights)

    return class_weights




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


def check_empty_docs(data, name):
    """
    Check for empty docs (strings) in a list of data and print details for debugging.

    Args:
        data: List of strings (e.g., train_data or test_data).
        name: Name of the dataset (e.g., "Train", "Test").
    """
    empty_indices = [i for i, doc in enumerate(data) if not doc.strip()]
    if empty_indices:
        print(f"[WARNING] {name} dataset contains {len(empty_indices)} empty strings (docs).")
        for idx in empty_indices[:10]:  # Print details for up to 10 empty rows
            print(f"Empty String at Index {idx}: Original Document: '{data[idx]}'")
    else:
        print(f"[INFO] No empty strings (docs) found in {name} dataset.")



def spot_check_documents(documents, vectorizer, lc_tokenizer, vectorized_data, num_docs=5, debug=False):
    """
    Spot-check random documents in the dataset for their TF-IDF calculations.

    Args:
        documents: List of original documents (strings).
        vectorizer: Fitted LCTFIDFVectorizer object.
        lc_tokenizer: Custom tokenizer object.
        vectorized_data: Sparse matrix of TF-IDF features.
        num_docs: Number of random documents to check.
        debug: Whether to enable debug-level logging.
    """

    print(f"spot_check_documents()... num_docs: {num_docs}, debug: {debug}")

    vocab = vectorizer.vocabulary_
    idf_values = vectorizer.idf_
    reverse_vocab = {idx: token for token, idx in vocab.items()}
    
    tokenizer = lc_tokenizer.tokenizer

    # Dynamically identify the pad token from the tokenizer
    pad_token = tokenizer.pad_token if tokenizer.pad_token else "[PAD]"
    pad_token_id = vocab.get(pad_token, None)

    print("\n[INFO] Spot-checking random documents...")
    
    if debug:
        print("documents:", type(documents), len(documents))
        print("documents[0]:", type(documents[0]), documents[0])
        print("vectorized_data:", type(vectorized_data))
        print(f"[DEBUG] Vocabulary size: {len(vocab)}, IDF array size: {len(idf_values)}\n")

    # Randomly select `num_docs` indices from the document list
    doc_indices = random.sample(range(len(documents)), min(num_docs, len(documents)))

    for doc_id in doc_indices:
        doc = documents[doc_id]

        if debug:
            print(f"[INFO] Document {doc_id}:")
            print(f"Original Text: {doc}\n")

        # Tokenize the document
        tokens = lc_tokenizer(doc)
        if debug:
            print(f"Tokens: {tokens}\n")

        vectorized_row = vectorized_data[doc_id]
        mismatches = []

        for token in tokens:
            if token in vocab:
                idx = vocab[token]
                tfidf_value = vectorized_row[0, idx]
                expected_idf = idf_values[idx]

                # Check for proper TF-IDF values
                if token == pad_token:
                    if expected_idf != 0 or tfidf_value != 0:
                        print(f"[ERROR] PAD token '{pad_token}' has incorrect values: IDF={expected_idf}, TF-IDF={tfidf_value}.")
                else:
                    if tfidf_value == 0:
                        print(f"[DEBUG] Token '{token}' in tokenizer vocabulary: {token in tokenizer.get_vocab()}")
                        print(f"[ERROR] Token '{token}' has IDF {expected_idf} but TF-IDF is 0.")
            else:
                print(f"[ERROR] Token '{token}' not in vectorizer vocabulary.")

    print("---finished spot-checking documents---")



def spot_check_documents_old(documents, vectorizer, tokenizer, vectorized_data, num_docs=5, debug=False):
    """
    Spot-check random documents in the dataset for their TF-IDF calculations.

    Args:
        documents: List of original documents (strings).
        vectorizer: Fitted LCTFIDFVectorizer object.
        tokenizer: Custom tokenizer object.
        vectorized_data: Sparse matrix of TF-IDF features.
        num_docs: Number of random documents to check.
    """
    vocab = vectorizer.vocabulary_
    idf_values = vectorizer.idf_
    reverse_vocab = {idx: token for token, idx in vocab.items()}
    
    print("\n[INFO] Spot-checking random documents...")
    
    if (debug):
        print("documents:", type(documents), len(documents))
        print("documents[0]:", type(documents[0]), documents[0])

        print("vectorized_data:", type(vectorized_data))
        print(f"[DEBUG] Vocabulary size: {len(vocab)}, IDF array size: {len(idf_values)}\n")

    # Randomly select `num_docs` indices from the document list
    doc_indices = random.sample(range(len(documents)), min(num_docs, len(documents)))

    for doc_id in doc_indices:
        doc = documents[doc_id]

        if (debug):
            print(f"[INFO] Document {doc_id}:")
            print(f"Original Text: {doc}\n")

        tokens = tokenizer(doc)
        if (debug):
            print(f"Tokens: {tokens}\n")

        vectorized_row = vectorized_data[doc_id]
        mismatches = []

        for token in tokens:
            if token in vocab:
                idx = vocab[token]
                if idx < len(idf_values):
                    tfidf_value = vectorized_row[0, idx]
                    expected_idf = idf_values[idx]
                    if tfidf_value == 0:
                        print(f"[DEBUG] Token '{token}' in tokenizer vocabulary: {token in tokenizer.get_vocab()}")
                        print(f"[ERROR] Token '{token}' has IDF {expected_idf} but TF-IDF is 0.")
                else:
                    print(f"[WARNING] Token '{token}' has out-of-bounds index {idx}.")
            else:
                print(f"[ERROR] Token '{token}' not in vectorizer vocabulary.")

    print("---finished spot checking docs---")




# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------



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
        
        if (model_name is None) or (model_path is None):
            print("model_name not provided, using default (BERT) tokenizer...")
            self.model_name, self.model_path = get_model_identifier('bert')
        
        # Debugging information
        print("LCTokenizer initialized with the following parameters:")
        print(f"  Model name: {self.model_name}")
        print(f"  Model path: {self.model_path}")
        print(f"  Lowercase: {self.lowercase}")
        print(f"  Remove special tokens: {self.remove_special_tokens}")
        print(f"  Padding: {self.padding}")
        print(f"  Truncation: {self.truncation}")
        
        #print("creating tokenizer using HF AutoTokenizer...")

        # instantiate the tokenizer
        self.tokenizer, self.vocab_size, self.max_length = self._instantiate_tokenizer()
        
        self.filtered = False


    def _instantiate_tokenizer(self, vocab_file=None):
        
        print(f'instantiating new tokenizer from model: {self.model_name} and path: {self.model_path}...')
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=self.model_path
            )
    
        # Use an existing token as the padding token
        if tokenizer.pad_token is None:
            print(f"Tokenizer has no pad token. Reusing 'eos_token' ({tokenizer.eos_token_id}).")
            tokenizer.pad_token = tokenizer.eos_token

        # Print tokenizer details
        vocab_size = len(tokenizer.get_vocab())
        print("vocab_size:", vocab_size)

        # Compute max_length from tokenizer
        max_length = tokenizer.model_max_length
        print(f"max_length: {max_length}")

        # Handle excessive or default max_length values
        if max_length > MAX_TOKEN_LENGTH:
            print(f"Invalid max_length ({max_length}) detected. Adjusting to {MAX_TOKEN_LENGTH}.")
            max_length = MAX_TOKEN_LENGTH

        return tokenizer, vocab_size, max_length
    

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

        text = self.normalize_text(text)        # normalize text

        tokens = self.tokenizer.tokenize(
            text,
            add_special_tokens=True,
            return_token_type_ids=True,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
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


    def filter_tokens(self, texts, dataset_name):
        """
        Compute the dataset vocabulary as token IDs that align with the tokenizer's vocabulary.

        Args:
            texts (list of str): Texts to compute the token set.
            dataset_name (str): Name of the dataset for saving the filtered vocabulary.

        Returns:
            tuple: (relevant_tokens, relevant_token_ids, mismatches)
                - relevant_tokens: List of tokens in the dataset that are in the tokenizer vocabulary.
                - relevant_token_ids: List of token IDs in the dataset that are in the tokenizer vocabulary.
                - mismatches: Tokens found in the dataset but not in the tokenizer vocabulary.
        """
        print(f"Computing dataset token list for dataset {dataset_name}...")
        print(f"max_length: {self.max_length}, padding: {self.padding}, truncation: {self.truncation}")

        # Initialize sets and variables
        dataset_vocab_ids = set()
        dataset_tokens = set()
        mismatches = []

        # Tokenizer vocabulary
        tokenizer_vocab = self.get_vocab()
        tokenizer_vocab_set = set(tokenizer_vocab.keys())

        # Tokenize each document with the same parameters used during input preparation
        for text in tqdm(texts, desc="Tokenizing documents..."):
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            input_ids = tokens['input_ids']
            decoded_tokens = [self.tokenizer.convert_ids_to_tokens(tok_id) for tok_id in input_ids]

            # Check for tokens not in vocabulary
            for tok_id, tok in zip(input_ids, decoded_tokens):
                if tok in tokenizer_vocab_set:
                    dataset_tokens.add(tok)
                    dataset_vocab_ids.add(tok_id)
                else:
                    print("WARNING: Token not in tokenizer vocabulary:", tok)
                    mismatches.append((tok, tok_id))

        # Build the filtered vocabulary
        relevant_token_ids = sorted(dataset_vocab_ids)
        relevant_tokens = [
            self.tokenizer.convert_ids_to_tokens(token_id) for token_id in relevant_token_ids
        ]

        # Ensure special tokens retain their original IDs
        special_tokens = {
            "pad_token": (self.tokenizer.pad_token, self.tokenizer.pad_token_id),
            "cls_token": (self.tokenizer.cls_token, self.tokenizer.cls_token_id),
            "sep_token": (self.tokenizer.sep_token, self.tokenizer.sep_token_id),
            "mask_token": (self.tokenizer.mask_token, self.tokenizer.mask_token_id),
            "unk_token": (self.tokenizer.unk_token, self.tokenizer.unk_token_id),
        }

        for key, (token, token_id) in special_tokens.items():
            if token and token not in relevant_tokens:
                if token is not None:
                    relevant_tokens.append(token)
                    relevant_token_ids.append(token_id)
                else:
                    print(f"[INFO] Special token '{key}' not found in the default tokenizer vocab, not putting in the filtered vocabulary.")

        # Create the filtered vocabulary dictionary
        filtered_vocab = {token: idx for idx, token in enumerate(relevant_tokens)}
        print("filtered_vocab::", type(filtered_vocab), len(filtered_vocab)
              )
        
        tokenizer_name = self.tokenizer.__class__.__name__
        print("tokenizer_name:", tokenizer_name)

        # Reset the tokenizer with the filtered vocabulary
        vocab_file = f"{OUT_DIR}{dataset_name}.{tokenizer_name}.filtered_vocab.json"
        with open(vocab_file, "w") as vf:
            json.dump(filtered_vocab, vf)
        print(f"Filtered vocabulary saved to: {vocab_file}")
        
        # 
        # TODO: update self.tokenizer vocabulary with the limited, filtered vocabulary
        # although this looks difficult to do, see 
        # https://stackoverflow.com/questions/69531811/using-hugginface-transformers-and-tokenizers-with-a-fixed-vocabulary
        #

        return relevant_tokens, relevant_token_ids, mismatches


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

    def __init__(self, lc_tokenizer, max_length=MAX_TOKEN_LENGTH, lowercase=False, debug=False):
        """
        Custom TF-IDF Vectorizer that aligns its vocabulary with the Hugging Face tokenizer.

        Args:
            tokenizer: Hugging Face tokenizer object.
            lowercase: Whether to convert text to lowercase.
            debug: Whether to enable debugging messages.
        """
        self.lc_tokenizer = lc_tokenizer
        self.max_length = max_length
        self.lowercase = lowercase
        self.debug = debug
        self.vocabulary_ = {token: idx for token, idx in lc_tokenizer.get_vocab().items()}

        self.pad_token = lc_tokenizer.tokenizer.pad_token  # Dynamically retrieve pad token
        if self.pad_token is None:
            raise ValueError("Pad token is not defined in the tokenizer.")

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
        max_length = self.lc_tokenizer.max_length
        num_documents = len(tokenized_documents)

        if (max_length != self.max_length):
            print(f"WARNING: Max length mismatch between tokenizer ({max_length}) and vectorizer ({self.max_length}).")

        for doc_idx, tokenized_text in enumerate(tokenized_documents):

            if isinstance(tokenized_text, str):
                tokens = tokenized_text.split()  # Split pre-tokenized text
            else:
                raise ValueError("Tokenized documents must be strings of tokenized text.")

            # Check sequence length
            if len(tokens) > max_length:
                print(f"[ERROR] Document {doc_idx} exceeds max length ({len(tokens)} > {max_length}).")
                print(f"[DEBUG] Document: {tokenized_documents[doc_idx][:200]}...")                             # Print a truncated version of the doc
                print(f"[DEBUG] Tokens: {tokens[:50]}...")                                                      # Print a sample of the tokens
                raise ValueError("Document exceeds max length.")
            
            # Filter out [PAD] and blank tokens
            tokens = [token for token in tokens if token.strip() and token != self.pad_token]

            unique_tokens = set(tokens)        
            unmatched_tokens = []

            for token in unique_tokens:
                if token in self.vocabulary_:
                    term_doc_counts[token] += 1
                else:
                    print(f"[WARNING] Unmatched token '{token}' in document {doc_idx}.")
                    unmatched_tokens.append(token)
            
        missing_tokens = [token for token in self.vocabulary_ if token not in term_doc_counts]
        if (self.debug):
            if missing_tokens:
                print(f"[WARNING] Tokens in vocabulary with no document counts: {missing_tokens[:10]}")

        num_documents = len(tokenized_documents)

        # Compute IDF values
        self.idf_ = np.zeros(len(self.vocabulary_), dtype=np.float64)
        for token, idx in self.vocabulary_.items():
            doc_count = term_doc_counts.get(token, 0)
            if token == self.pad_token:
                self.idf_[idx] = 0                          # Exclude pad_token from TF-IDF calculations
            else:
                self.idf_[idx] = np.log((1 + num_documents) / (1 + doc_count)) + 1
                if self.debug and self.idf_[idx] == 0:
                    print(f"[WARNING] IDF for token '{token}' is 0 during fit. "
                        f"Document count: {doc_count}, Total docs: {num_documents}.")

        if self.debug:
            # Debug: Check if special tokens are present
            special_tokens = self.lc_tokenizer.tokenizer.all_special_tokens
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
        rows, cols, data = [], [], []
        empty_rows_details = []  # To collect details of empty rows

        for row_idx, tokenized_doc in enumerate(tokenized_documents):

            if isinstance(tokenized_doc, str):
                tokens = tokenized_doc.split()             # Split pre-tokenized text
            else:
                raise ValueError("Tokenized documents must be strings of tokenized text.")

            # Filter out pad_token and blank tokens
            tokens = [token for token in tokens if token.strip() and token != self.pad_token]
            term_freq = defaultdict(int)
            
            if len(tokens) == 0:
                print(f"[WARNING] Document {row_idx} contains no valid tokens after tokenization.")

            term_freq = defaultdict(int)
            unmatched_tokens = []

            for token in tokens:
                if token in self.vocabulary_:
                    term_freq[token] += 1
                else:
                    print("[WARNING] Unmatched token '{token}' in document {row_idx}.")
                    unmatched_tokens.append(token)

            # Handle empty rows (no matched tokens)
            if not term_freq:
                if original_documents is not None:
                    empty_rows_details.append((row_idx, original_documents[row_idx], tokens, unmatched_tokens))
                else:
                    empty_rows_details.append((row_idx, None, tokens, unmatched_tokens))

            # Calculate TF-IDF for tokens in the vocabulary
            total_tokens = len(tokens)
            if total_tokens == 0:
                continue  # Skip documents with no tokens

            for token, freq in term_freq.items():
                col_idx = self.vocabulary_[token]
                tf = freq / total_tokens  # Term frequency
                tfidf = tf * self.idf_[col_idx]  # TF-IDF value

                """
                if self.debug:
                    print(f"[DEBUG] Row {row_idx}, Token '{token}', TF: {tf}, IDF: {self.idf_[col_idx]}, TF-IDF: {tfidf}")
                """

                rows.append(row_idx)
                cols.append(col_idx)
                data.append(tfidf)

        # Construct sparse matrix
        matrix = csr_matrix((data, (rows, cols)), shape=(len(tokenized_documents), len(self.vocabulary_)))

        if self.debug:
            # Debugging: Check for empty rows in the matrix
            empty_rows = matrix.sum(axis=1).A1 == 0
            for row_idx, original_doc, tokens, unmatched_tokens in empty_rows_details:
                if empty_rows[row_idx]:
                    print(f"[WARNING] Row {row_idx} in TF-IDF matrix is empty.")
                    print(f"[INFO] Original document: {original_doc}")
                    print(f"[INFO] Tokens: {tokens}")
                    print(f"[INFO] Unmatched tokens: {unmatched_tokens}")

            # Debugging: Special tokens' TF-IDF values
            special_tokens = self.lc_tokenizer.tokenizer.all_special_tokens
            for token in special_tokens:
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    print(f"[DEBUG] Special token '{token}' - IDF: {self.idf_[idx]}")

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

    #print("lc_tokenizer:\n", lc_tokenizer)
    
    tokenized_train = [" ".join(lc_tokenizer(text)) for text in texts_train]
    tokenized_val = [" ".join(lc_tokenizer(text)) for text in texts_val]
    tokenized_test = [" ".join(lc_tokenizer(text)) for text in texts_test]

    # Debugging: Preprocessed data
    print("tokenized_train:", type(tokenized_train), len(tokenized_train))
    print(f"tokenized_train[0]: {tokenized_train[0]}")
    
    tokenizer_vocab = lc_tokenizer.tokenizer.get_vocab()

    # Debugging: Check preprocessed tokens and unmatched tokens
    print("[DEBUG] Checking tokenized tokens and their presence in vocabulary...")
    for i, doc in enumerate(tokenized_train[:5]):  # Sample first 5 documents
        tokens = doc.split()
        unmatched = [token for token in tokens if token not in lc_tokenizer.get_vocab()]
        print(f"[INFO]: Document {i}: Tokens: {tokens[:20]}...")  # Print a subset of tokens
        if unmatched:
            print(f"[WARNING]: Document {i} has unmatched tokens: {unmatched[:10]}")  # Show a few unmatched tokens

    vectorizer = LCTFIDFVectorizer(
        lc_tokenizer=lc_tokenizer, 
        max_length=lc_tokenizer.max_length,
        debug=debug
        )

    Xtr = vectorizer.fit_transform(
        X=tokenized_train,
        original_documents=texts_train
        )
    
    Xval = vectorizer.transform(
        X=tokenized_val,
        original_documents=texts_val
        )
    
    Xte = vectorizer.transform(
        X=tokenized_test,
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

