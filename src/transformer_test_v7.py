import numpy as np
import os
import argparse
from time import time
from tqdm import tqdm

import pandas as pd
import torch
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel, RobertaTokenizerFast, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

from util.metrics import evaluation_nn, evaluation_ml
from util.common import get_embedding_type, initialize_testing

from embedding.pretrained import *
#from data.dataset import *

from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1
from sklearn.datasets import get_data_home, fetch_20newsgroups

from embedding.supervised import get_supervised_embeddings

from scipy.sparse import csr_matrix


os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore')


#
# we assume everything runs from bin directory
#
PICKLE_DIR = '../pickles/'
OUT_DIR = '../out/'
DATASET_DIR = '../datasets/'
VECTOR_CACHE = '../.vector_cache'

# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 16
DEFAULT_GPU_BATCH_SIZE = 64
MPS_BATCH_SIZE = 16

PATIENCE = 2                            # # of loops before early stopping
EPOCHS = 6
LR = 1e-6

SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news", "ohsumed"]


# Define a default model mapping (optional) to avoid invalid identifiers
MODEL_MAP = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "xlnet": "xlnet-base-cased",
    "gpt2": "gpt2",
    "llama": "meta-llama/Llama-2-7b-chat-hf"  # Example for a possible LLaMA identifier
}


TEST_SIZE = 0.1
VAL_SIZE = 0.1

RANDOM_SEED = 42

MC_THRESHOLD = 0.5          # Multi-class threshold



# Get the full model identifier and load from local directory
def get_model_identifier(pretrained, cache_dir="../.vector_cache"):

    model_name = MODEL_MAP.get(pretrained, pretrained)
    model_path = os.path.join(cache_dir, pretrained)

    return model_name, model_path



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

        data_path = '../datasets/rcv1' 
        devel = fetch_RCV1(subset='train', data_path=data_path)
        test = fetch_RCV1(subset='test', data_path=data_path)

        train_data = devel.data      # Sparse matrix of token counts
        train_target = devel.target  # Multi-label binarized format

        test_data = test.data        # Sparse matrix of token counts
        test_target = test.target    # Multi-label binarized format

        class_type = 'multi-label'
                
        target_names = devel.target_names
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



class TransformerWCEClassifier(nn.Module):

    def __init__(self, transformer_model, num_classes, device, wce_matrix=None, use_supervised=False, combination="concat"):
        """
        A Transformer-based classifier with optional WCE integration.
        
        Args:
            transformer_model: The pre-trained transformer model (e.g., BERT).
            num_classes: Number of classes for classification.
            wce_matrix: Precomputed WCE matrix (Tensor) with shape [vocab_size, num_classes].
            use_supervised: Whether to use WCE embeddings.
            combination: Method to integrate WCE embeddings ("add" or "concat").
        """
        super(TransformerWCEClassifier, self).__init__()

        print(f'TransformerWCEClassifier:__init__...num_classes: {num_classes}, use_supervised: {use_supervised}, combination: {combination}, device: {device}')

        self.transformer = transformer_model
        self.transformer.to(device)
        self.hidden_size = self.transformer.config.hidden_size
        self.num_classes = num_classes
        self.use_supervised = use_supervised
        self.combination = combination
        self.device = device

        # Convert WCEs into a learnable parameter
        if wce_matrix is not None:
            self.wce_embedding = nn.Parameter(torch.tensor(wce_matrix, dtype=torch.float32).to(device), requires_grad=True)
        else:
            self.wce_embedding = None

        # Define a projection layer if using "add" combination
        if self.use_supervised and self.combination == "add":
            self.wce_projection = nn.Linear(num_classes, self.hidden_size)

        # Dynamically calculate classifier input size
        if self.use_supervised and self.combination == "concat":
            self.classifier_input_size = self.hidden_size + self.num_classes
        else:
            self.classifier_input_size = self.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

        # LayerNorm for feature normalization (only needed if supervised embeddings are used)
        if self.use_supervised:
            self.layer_norm = nn.LayerNorm(self.hidden_size + (num_classes if combination == "concat" else 0))

        # Learnable scaling parameter for WCE contribution
        self.wce_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32).to(device))  # Initial weight for WCE


    def forward(self, input_ids, attention_mask, token_ids=None):
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = transformer_output.last_hidden_state

        # Mean pooling across sequence
        attention_mask_expanded = attention_mask.unsqueeze(-1)
        token_embeddings = token_embeddings * attention_mask_expanded
        pooled_token_output = token_embeddings.sum(dim=1) / attention_mask_expanded.sum(dim=1)

        # WCE integration
        if self.use_supervised and self.wce_embedding is not None and token_ids is not None:
            # Lookup WCE embeddings and perform mean pooling
            wce_embeddings = self.wce_embedding[token_ids]
            wce_embeddings = wce_embeddings * attention_mask_expanded
            pooled_wce_output = wce_embeddings.sum(dim=1) / attention_mask_expanded.sum(dim=1)

            # Scale WCE contributions
            pooled_wce_output = self.wce_weight * pooled_wce_output

            # Combine token and WCE embeddings
            if self.combination == "concat":
                pooled_output = torch.cat((pooled_token_output, pooled_wce_output), dim=1)
            elif self.combination == "add":
                pooled_wce_output = self.wce_projection(pooled_wce_output)
                pooled_output = pooled_token_output + pooled_wce_output

            # Dynamically adjust LayerNorm
            layer_norm = nn.LayerNorm(pooled_output.size(-1)).to(self.device)  # Create LayerNorm for the correct size
            pooled_output = layer_norm(pooled_output)

            # Apply normalization
            pooled_output = self.layer_norm(pooled_output)
        else:
            pooled_output = pooled_token_output
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits




def train(model, train_loader, val_loader, device, class_type='single-label', epochs=3, learning_rate=2e-5, patience=3):
    """
    Trains the Transformer-based model with optional WCE integration.

    Args:
    - model: Transformer-based classifier (e.g., TransformerWithWCEClassifier).
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - device: Device to run the model on (CPU/GPU).
    - class_type: "single-label" or "multi-label".
    - epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.
    - patience: Early stopping patience.

    Returns:
    None
    """

    print(f'training model...epochs: {epochs}, learning_rate: {learning_rate}, patience: {patience}, device: {device}, class_type: {class_type}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print("optimizer:\n", optimizer)

    # Use appropriate loss function
    criterion = nn.BCEWithLogitsLoss() if class_type == "multi-label" else nn.CrossEntropyLoss()
    print("criterion:\n", criterion)

    model.to(device)

    best_macro_val_f1 = 0.0
    patience_counter = 0
    final_loss = None  # To track the final loss

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        #for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
            optimizer.zero_grad()

            # Load batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_ids = batch.get('token_ids', None)

            if token_ids is not None:
                token_ids = token_ids.to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_ids=token_ids)
            loss = criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        final_loss = avg_loss  # Update the final loss with the last epoch's loss
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader)}")

        # Validate after each epoch
        val_macro_f1, val_micro_f1, acc = validate(model, val_loader, device, class_type=class_type)
        if class_type == "multi-label":
            print(f"Validation Macro F1: {val_macro_f1:.4f}, Validation Micro F1: {val_micro_f1:.4f}")
        else:
            print(f"Validation Macro F1: {val_macro_f1:.4f}, Validation Micro F1: {val_micro_f1:.4f}, accuracy: {acc:.4f}")

        # Early stopping
        if val_macro_f1 > best_macro_val_f1:
            best_macro_val_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), "../out/best_model.pth")
            print("saved model to ../out/best_model.pth...")
        else:
            patience_counter += 1
            print(f"No improvement in validation F1 for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            return epoch + 1, final_loss
        
    return epochs, final_loss



def validate(model, val_loader, device, class_type="single-label"):
    """
    Validates the Transformer-based model during training.

    Args:
    - model: Transformer-based classifier.
    - val_loader: DataLoader for validation data.
    - device: Device to run the model on (CPU/GPU).
    - class_type: "single-label" or "multi-label".

    Returns:
    tuple: Macro F1 score, Micro F1 score, and Accuracy on validation data.
    """
    
    print(f'validate...class_type: {class_type}')

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        #for batch in val_loader:
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_ids = batch.get('token_ids', None)

            if token_ids is not None:
                token_ids = token_ids.to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_ids=token_ids)
            print(f"\nBatch {batch_idx}: Raw logits:")
            print(logits[:5].cpu().numpy())  # Print logits for the first 5 samples in the batch

            if class_type == "multi-label":
                # Apply sigmoid to convert logits to probabilities
                probabilities = torch.sigmoid(logits)
                print(f"Batch {batch_idx}: Sigmoid probabilities (first 5 samples):")
                print(probabilities[:5].cpu().numpy())

                # Convert probabilities to binary predictions
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()           # Binary predictions for multi-label
                print(f"Batch {batch_idx}: Predictions (first 5 samples):")
                print(preds[:5])

                all_preds.append(preds)
            else:
                preds = torch.argmax(logits, dim=1).cpu().numpy()                   # Class indices for single-label
                all_preds.extend(preds)                                             # Use extend to handle 1D predictions

            # Ground truth labels
            print(f"Batch {batch_idx}: Ground truth labels (first 5 samples):")
            print(labels[:5].cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())

    if class_type == "multi-label":
        all_preds = np.vstack(all_preds)                                # Multi-label predictions are 2D
        all_labels = np.vstack(all_labels).astype(int)                  # Ensure labels are integers
    else:
        all_preds = np.array(all_preds)                                 # Single-label predictions are 1D
        all_labels = np.array(all_labels).astype(int)                   # Ensure labels are integers

    print("all_preds:", type(all_preds), all_preds.shape)
    print("all_preds[0]:", type(all_preds[0]), all_preds[0].shape, all_preds[0])
    print("all_labels:", type(all_labels), all_labels.shape)
    print("all_labels[0]:", type(all_labels[0]), all_labels[0].shape, all_labels[0])
    
    # Ensure consistent format for f1_score
    if class_type == "multi-label":
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        acc = None  # Accuracy is not applicable for multi-label classification
    else:
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        acc = accuracy_score(all_labels, all_preds)

    return macro_f1, micro_f1, acc




def evaluate(model, test_loader, device, logfile, tinit, dimensions, target_names, class_type, epochs, final_loss):
    """
    Evaluates the Transformer-based model with optional WCE integration.

    Args:
    - model: Trained Transformer-based classifier.
    - test_loader: DataLoader for test data.
    - device: Device to run the model on (CPU/GPU).

    Returns:
    dict: Evaluation metrics (accuracy, macro/micro F1 scores).
    """

    print(f'\n\tevaluating...')
    print(f'class_type: {class_type}, epochs: {epochs}, final_loss: {final_loss}, device: {device}, target_names: {target_names}, dimensions: {dimensions}')

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_ids = batch.get('token_ids', None)

            if token_ids is not None:
                token_ids = token_ids.to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_ids=token_ids)

            if class_type == "multi-label":
                preds = (torch.sigmoid(logits) > MC_THRESHOLD).int().cpu().numpy()  # Binary predictions for multi-label
                all_preds.append(preds)
            else:
                preds = torch.argmax(logits, dim=1).cpu().numpy()  # Class indices for single-label
                all_preds.extend(preds)  # Use extend to handle 1D predictions

            all_labels.extend(labels.cpu().numpy())

    if class_type == "multi-label":
        all_preds = np.vstack(all_preds)                            # Multi-label predictions are 2D
        all_labels = np.vstack(all_labels).astype(int)              # Ensure labels are ints
    else:
        all_preds = np.array(all_preds)                             # Single-label predictions are 1D
        all_labels = np.array(all_labels).astype(int)                # Ensure labels are ints

    print("all_preds:", type(all_preds), all_preds.shape)
    print("all_preds[0]:", type(all_preds[0]), all_preds[0].shape, all_preds[0])
    print("all_labels:", type(all_labels), all_labels.shape)
    print("all_labels[0]:", type(all_labels[0]), all_labels[0].shape, all_labels[0])
    
    # Metrics computation
    if class_type == "multi-label":
        # Multi-label metrics
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        print(f"Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")

    else:
        # Single-label metrics
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")
    
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    macrof1, microf1, acc, h_loss, precision, recall, j_index = evaluation_nn(all_labels, all_preds, classification_type=class_type)
    print("\n--Layer Cake Metrics--")
    print(f"Macro-F1 = {macrof1:.4f}, Micro-F1 = {microf1:.4f}, Accuracy = {acc:.4f}, H-loss = {h_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, Jaccard = {j_index:.4f}")
    
    tend = time() - tinit

    measure_prefix = 'final'
    epoch = epochs
    
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=macrof1, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=microf1, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-loss', value=final_loss, timelapse=tend)

    logfile.insert(dimensions=dimensions, epoch=epoch, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure='te-precision', value=precision, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure='te-recall', value=recall, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure='te-jacard-index', value=j_index, timelapse=tend)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1
    }


# Dataset preparation
class TextDataset(Dataset):

    def __init__(self, encodings, labels, vectorizer_vocab, tokenizer, max_length, class_type="single-label"):
        """
        Dataset class for single-label and multi-label classification.

        Args:
            encodings (dict): Encoded input features (tokenized text).
            labels (list or ndarray): Target labels.
            vectorizer_vocab (dict): Vocabulary mapping for the vectorizer.
            tokenizer (PreTrainedTokenizerBase): Hugging Face tokenizer.
            max_length (int): Maximum sequence length for tokenization.
            class_type (str): "single-label" or "multi-label".
        """

        print(f'TextDataset:__init__...class_type: {class_type}')

        self.encodings = encodings
        self.labels = labels
        self.vectorizer_vocab = vectorizer_vocab
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_type = class_type

    def __getitem__(self, idx):

        item = {key: val[idx] for key, val in self.encodings.items()}

        # Convert labels to appropriate type
        if self.class_type == "multi-label":
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)                  # Keep as float for BCEWithLogitsLoss
        else:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        # Generate token_ids for WCE lookup
        tokens = self.tokenizer.convert_ids_to_tokens(self.encodings['input_ids'][idx])
        token_ids = [
            self.vectorizer_vocab.get(token, 0)  # Default to 0 for unknown tokens
            for token in tokens[:self.max_length]
        ]
        token_ids += [0] * (self.max_length - len(token_ids))  # Padding
        item['token_ids'] = torch.tensor(token_ids, dtype=torch.long)

        return item

    def __len__(self):
        return len(self.labels)




class TextDatasetOld(Dataset):

    def __init__(self, encodings, labels, vectorizer_vocab, tokenizer, max_length):
        self.encodings = encodings
        self.labels = labels
        self.vectorizer_vocab = vectorizer_vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        # Generate token_ids for WCE lookup
        tokens = self.tokenizer.convert_ids_to_tokens(self.encodings['input_ids'][idx])
        token_ids = [
            self.vectorizer_vocab.get(token, 0)  # Default to 0 for unknown tokens
            for token in tokens[:self.max_length]
        ]
        token_ids += [0] * (self.max_length - len(token_ids))  # Padding
        item['token_ids'] = torch.tensor(token_ids, dtype=torch.long)

        return item

    def __len__(self):
        return len(self.labels)



def vectorize(texts_train, texts_val, texts_test, tokenizer, vtype='tfidf'):
    """
    Vectorizes text data using TfidfVectorizer or CountVectorizer, aligning the vocabulary with the Hugging Face tokenizer.

    Args:
        texts_train (list): List of training texts (already preprocessed for truncation).
        texts_val (list): List of validation texts (already preprocessed for truncation).
        texts_test (list): List of test texts (already preprocessed for truncation).
        tokenizer: Hugging Face tokenizer instance with a vocabulary.
        vtype (str): Vectorization type ('tfidf' or 'count').

    Returns:
        vectorizer: The fitted vectorizer instance.
        Xtr: Sparse matrix of vectorized training data.
        Xval: Sparse matrix of vectorized validation data.
        Xte: Sparse matrix of vectorized test data.
    """
    print(f'vectorize(), vtype: {vtype}, tokenizer max_length: {tokenizer.model_max_length}')

    # Retrieve vocabulary from the tokenizer
    vocab = tokenizer.get_vocab()
    print("vocab:", type(vocab), len(vocab))

    # Initialize vectorizer based on type
    if vtype == 'tfidf':
        vectorizer = TfidfVectorizer(
            min_df=5,  # Ignore terms with document frequency < 5
            lowercase=False,  # Tokenizer already handles case sensitivity
            sublinear_tf=True,  # Use sublinear term frequency scaling
            vocabulary=vocab  # Align with tokenizer's vocabulary
        )
    elif vtype == 'count':
        vectorizer = CountVectorizer(
            min_df=5,
            lowercase=False,
            vocabulary=vocab
        )
    else:
        raise ValueError(f"Unsupported vectorization type: {vtype}. Choose 'tfidf' or 'count'.")

    # Fit and transform the training data
    Xtr = vectorizer.fit_transform(texts_train)

    # Transform the validation and test data
    Xval = vectorizer.transform(texts_val)
    Xte = vectorizer.transform(texts_test)

    # Sort indices for efficiency
    Xtr.sort_indices()
    Xval.sort_indices()
    Xte.sort_indices()

    # Ensure the output is a sparse matrix (redundant safety check)
    Xtr = csr_matrix(Xtr) if not isinstance(Xtr, csr_matrix) else Xtr
    Xval = csr_matrix(Xval) if not isinstance(Xval, csr_matrix) else Xval
    Xte = csr_matrix(Xte) if not isinstance(Xte, csr_matrix) else Xte

    print("Vectorization complete. Shapes - Train:", Xtr.shape, ", Val:", Xval.shape, ", Test:", Xte.shape)
    return vectorizer, Xtr, Xval, Xte





def vectorize_old(texts_train, texts_val, texts_test, tokenizer):

        print("vectorizing dataset text...")
        vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True, tokenizer=tokenizer.tokenize)

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




def classify(opt, device, batch_size=MPS_BATCH_SIZE, lr=LR, epochs=EPOCHS, patience=PATIENCE):
    """
    Fine-tunes the BERT or RoBERTa model and tests it.
    
    Args:
    - opt: Argument object containing dataset information.
    - device: Device to run the model on (CPU/GPU).
    - batch_size: Batch size for training.
    - epochs: Number of training epochs.
    - max_length: Maximum token length for BERT tokenization.
    - model_name: Pretrained model name (default: 'bert-base-uncased').
    
    Returns:
    None
    """

    # initialize logging and other system run variables
    already_modelled, logfile, method_name, pretrained, embeddings, emb_type, emb_path, lm_type, mode, system = initialize_testing(args)

    # check to see if model params have been computed already
    if (already_modelled):
        print(f'--- model {method_name} with embeddings {embeddings}, pretrained == {pretrained}, tunable == {args.tunable}, and wc_supervised == {args.supervised} for {args.dataset} already calculated, run with --force option to override. ---')
        exit(0)

    print("dataset:", opt.dataset)
    print("pretrained:", opt.pretrained)
    print("vtype:", opt.vtype)

    embedding_type = get_embedding_type(opt.pretrained)
    print("embedding_type:", embedding_type)
    print("embeddings:", embeddings)    
    print("embedding_path:", emb_path)

    torch.manual_seed(args.seed)

    # Load dataset and print class information
    (train_data, train_target), (test_data, labels_test), num_classes, target_names, class_type = load_dataset(args.dataset)

    print(f'DATA: {args.dataset}, class_type: {class_type}, num_classes: {num_classes}')
    print("train_data:", type(train_data), len(train_data))
    print("train_data[0]:", type(train_data[0]), train_data[0])
    print("train_target:", type(train_target), len(train_target))
    print("train_target[0]:", type(train_target[0]), train_target[0].shape, train_target[0])

    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), len(labels_test))
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])

    print("target_names:", target_names)

    # Split train into train and validation
    texts_train, texts_val, labels_train, labels_val = train_test_split(train_data, train_target, test_size=VAL_SIZE, random_state=RANDOM_SEED)

    print(f'AFTER SPLITTING: train_data: {len(texts_train)}, val_data: {len(texts_val)}, test_data: {len(test_data)}')

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
    
    print("SETTING UP MODEL...")
    model_name, model_path = get_model_identifier(args.pretrained)
    print("model_name:", model_name)
    print("model_path:", model_path)

    start = time()                      # start the clock

    print("tokenizing texts...")

    # Tokenizer initialization
    #tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=VECTOR_CACHE + '/BERT')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
    print("tokenizer:\n", tokenizer)

    # Dynamically get the maximum length
    max_length = tokenizer.model_max_length
    print("Maximum sequence length:", max_length)

    # Tokenize inputs
    train_encodings = tokenizer(texts_train, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    val_encodings = tokenizer(texts_val, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    test_encodings = tokenizer(test_data, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    vectorizer, Xtr, Xval, Xte = vectorize(texts_train, texts_val, test_data, tokenizer)
    print("vectorizer:\n", vectorizer)

    print("Xtr:", type(Xtr), Xtr.shape)
    #print("Xtr[0]:", type(Xtr[0]), Xtr[0].shape, Xtr[0])    
    print("Xval:", type(Xval), Xval.shape)
    #print("Xval[0]:", type(Xval[0]), Xval[0].shape, Xval[0])
    print("Xte:", type(Xte), Xte.shape)
    #print("Xte[0]:", type(Xte[0]), Xte[0].shape, Xte[0])

    print("vectorizer.vocabulary_:", type(vectorizer.vocabulary_), len(vectorizer.vocabulary_))
    vec_vocab_size = len(vectorizer.vocabulary_)
    print("vec_vocab_size:", vec_vocab_size)

    if (class_type in ['single-label', 'singlelabel']):

        print("single label, converting target labels to to one-hot")

        label_binarizer = LabelBinarizer()        
        one_hot_labels_train = label_binarizer.fit_transform(labels_train)
        one_hot_labels_val = label_binarizer.transform(labels_val)
        one_hot_labels_test = label_binarizer.transform(labels_test)

    # Compute supervised WCEs
    wce_train, wce_val, wce_test = None, None, None
    if opt.supervised:
        print("Computing supervised embeddings...")
        WCE = get_supervised_embeddings(
            X=Xtr,
            y=one_hot_labels_train,
            method=opt.supervised_method,
            max_label_space=opt.max_label_space,
            dozscore=(not opt.nozscore),
            transformers=True
        )
        num_missing_rows = vec_vocab_size - WCE.shape[0]
        WCE = np.vstack((WCE, np.zeros((num_missing_rows, WCE.shape[1]))))
        wce_train = torch.from_numpy(WCE).float()
        wce_val = torch.zeros_like(wce_train)  # Placeholder for validation
        wce_test = torch.zeros_like(wce_train)  # Placeholder for test

    if (wce_train is not None):
        print("wce_train:", type(wce_train), wce_train.shape)
        #print("wce_val:", type(wce_val), wce_val.shape)
        #print("wce_test:", type(wce_test), wce_test.shape)
    else:
        print("wce_train: None")

    max_length = tokenizer.model_max_length
    print("Maximum sequence length:", max_length)

    train_dataset = TextDataset(train_encodings, labels_train, vectorizer.vocabulary_, tokenizer, max_length, class_type=class_type)
    val_dataset = TextDataset(val_encodings, labels_val, vectorizer.vocabulary_, tokenizer, max_length, class_type=class_type)
    test_dataset = TextDataset(test_encodings, labels_test, vectorizer.vocabulary_, tokenizer, max_length, class_type=class_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    transformer_model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
    transformer_model.to(device)
    print("transformer_model:\n", transformer_model)

    # Get embedding size from the model
    embedding_size = transformer_model.config.hidden_size
    print(f"Embedding size: {embedding_size}")

    # Inspect the embedding layer
    embedding_layer = transformer_model.embeddings.word_embeddings
    #print(f"Embedding layer dimensions: {embedding_layer.weight.shape}")

    dimensions = embedding_layer.weight.shape       # for logging
    print("dimensions:", dimensions)

    model = TransformerWCEClassifier(
        transformer_model=transformer_model,
        num_classes=num_classes,
        device=device,
        wce_matrix=wce_train,                       # Precomputed WCE matrix
        use_supervised=opt.supervised,
        combination="concat"                        # Use "concat" or "add" as needed
    ).to(device)
        
    print("TransformerWCEModel:\n", model)

    # Train the model
    num_epochs, final_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_type=class_type,
        epochs=epochs,
        learning_rate=lr,
        patience=patience
    )

    # Evaluate the model
    evaluation_metrics = evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        logfile=logfile,
        tinit=start,
        dimensions=dimensions,
        target_names=target_names,
        class_type=class_type,
        epochs=num_epochs,
        final_loss=final_loss
    )
    
    print("Evaluation Metrics:", evaluation_metrics)
    print(f'Test time = {time() - start}')
    print("\n\t--- model training and evaluation complete---\n")



    
if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Text Classification with BERT.")
    
    parser.add_argument('--dataset', required=True, type=str, default='20newsgroups', help='Dataset to use: 20newsgroups or bbc-news.')
    
    parser.add_argument('--log_file', type=str, default='../log/lc_nn_test.test', help='Path to log file')

    parser.add_argument('--vtype', type=str, default='tfidf', metavar='N', 
                        help=f'dataset base vectorization strategy, in [tfidf, count]')

    parser.add_argument('--dropprob', type=float, default=0.5, metavar='[0.0, 1.0]',
                        help='dropout probability (default: 0.5)')
    
    parser.add_argument('--droptype', type=str, default='sup', metavar='DROPTYPE',
                        help=f'chooses the type of dropout to apply after the embedding layer. Default is "sup" which '
                             f'only applies to word-class embeddings (if present). Other options include "none" which '
                             f'does not apply dropout (same as "sup" with no supervised embeddings), "full" which '
                             f'applies dropout to the entire embedding, or "learn" that applies dropout only to the '
                             f'learnable embedding.')
    
    parser.add_argument('--static', action='store_true', help='keep the underlying pretrained model static (ie no unfrozen layers)')

    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train neural model.')

    parser.add_argument('--patience', type=int, default=PATIENCE, metavar='int',
                        help='patience for early-stop (default: 2)')
    
    parser.add_argument('--seed', type=int, default=1, metavar='int',
                        help='random seed (default: 1)')
    
    parser.add_argument('--learnable', type=int, default=0, metavar='int',
                        help='dimension of the learnable embeddings (default 0)')

    parser.add_argument('--tunable', action='store_true', default=False,
                        help='pretrained embeddings are tunable from the beginning (default False, i.e., static)')
    
    parser.add_argument('--weight_decay', type=float, default=0, metavar='float',
                        help='weight decay (default: 0)')
    
    parser.add_argument('--hidden', type=int, default=512, metavar='int',
                        help='hidden lstm size (default: 512)')
    
        
    parser.add_argument('--pretrained', type=str, default=None, metavar='str',
                        help='pretrained embeddings, one of "bert", "roberta", "xlnet", "gpt2", or "llama" (default None)')

    parser.add_argument('--bert-path', type=str, default=VECTOR_CACHE+'/BERT',
                        metavar='PATH',
                        help=f'directory to BERT pretrained vectors, used only with --pretrained bert')
    
    parser.add_argument('--roberta-path', type=str, default=VECTOR_CACHE+'/RoBERTa',
                        metavar='PATH',
                        help=f'directory to RoBERTa pretrained vectors, used only with --pretrained roberta')

    parser.add_argument('--distilbert-path', type=str, default=VECTOR_CACHE+'/DistilBERT',
                        metavar='PATH',
                        help=f'directory to DistilBERT pretrained vectors, used only with --pretrained distilbert')
    
    parser.add_argument('--xlnet-path', type=str, default=VECTOR_CACHE+'/XLNet',
                        metavar='PATH',
                        help=f'directory to XLNet pretrained vectors, used only with --pretrained xlnet')
    
    parser.add_argument('--gpt2-path', type=str, default=VECTOR_CACHE+'/GPT2',
                        metavar='PATH',
                        help=f'directory to GPT2 pretrained vectors, used only with --pretrained gpt2')

    parser.add_argument('--llama-path', type=str, default=VECTOR_CACHE+'/LlaMa',
                        metavar='PATH',
                        help=f'directory to LlaMa pretrained vectors, used only with --pretrained llama')

    
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='use supervised embeddings')
    
    parser.add_argument('--nozscore', action='store_true', default=False,
                        help='disables z-scoring form the computation of WCE')
    
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
    
    parser.add_argument('--max-label-space', type=int, default=300, metavar='int',
                        help='larger dimension allowed for the feature-label embedding (if larger, then PCA with this '
                             'number of components is applied (default 300)')


    args = parser.parse_args()
    print("args:", args)
        
    # Setup device prioritizing CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        args.batch_size = DEFAULT_GPU_BATCH_SIZE
    elif torch.backends.mps.is_available():
        args.device = torch.device("mps")
        args.batch_size = MPS_BATCH_SIZE
    else:
        args.device = torch.device("cpu")
        args.batch_size = DEFAULT_CPU_BATCH_SIZE
    print(f'running on {args.device}')
    print("batch_size:", args.batch_size)

    args.net = 'hf.class.ff'
    print("args.net:", args.net)

    classify(
        args, 
        device=args.device, 
        batch_size=args.batch_size, 
        lr=LR,
        epochs=args.epochs,
        patience=args.patience
    )
        