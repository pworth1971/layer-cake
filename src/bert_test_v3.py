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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelBinarizer

from util.metrics import evaluation_nn, evaluation_ml
from util.common import get_embedding_type, initialize_testing

from embedding.pretrained import *
from data.dataset import *

from embedding.supervised import get_supervised_embeddings

from scipy.sparse import csr_matrix



import warnings
warnings.filterwarnings('ignore')


#
# we assume everything runs from bin directory
#
PICKLE_DIR = '../pickles/'
OUT_DIR = '../out/'
DATASET_DIR = '../datasets/'
VECTOR_CACHE = '../.vector_cache'


# Define a default model mapping (optional) to avoid invalid identifiers
MODEL_MAP = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "xlnet": "xlnet-base-cased",
    "gpt2": "gpt2",
    "llama": "meta-llama/Llama-2-7b-chat-hf"  # Example for a possible LLaMA identifier
}

BERT_MODEL      = 'bert-base-uncased'               # dimension = 768
LLAMA_MODEL     = 'meta-llama/Llama-2-7b-hf'        # dimension = 4096
ROBERTA_MODEL   = 'roberta-base'                    # dimension = 768
#LLAMA_MODEL = 'meta-llama/Llama-2-13b-hf'

MAX_LENGTH = 512  # Max sequence length for the transformer models

# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 16
DEFAULT_GPU_BATCH_SIZE = 64
MPS_BATCH_SIZE = 16

PATIENCE = 2                            # # of loops before early stopping
EPOCHS = 6
LR = 1e-6

SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news", "ohsumed"]

TEST_SIZE = 0.1
VAL_SIZE = 0.1

RANDOM_SEED = 42

MC_THRESHOLD = 0.5          # Multi-class threshold




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
        self.hidden_size = self.transformer.config.hidden_size
        self.num_classes = num_classes
        self.use_supervised = use_supervised
        self.combination = combination  # How to combine WCE and token embeddings: "add" or "concat"

        self.device = device

        # Store WCE matrix for lookup if provided
        self.wce_matrix = wce_matrix            # [vocab_size, num_classes] (or None)
        if (wce_matrix is not None):
            wce_matrix.to(self.device)
        print("self.wce_matrix:", type(self.wce_matrix), wce_matrix.shape)

        # Dynamically calculate input size for classifier
        if self.use_supervised and self.combination == "concat":
            classifier_input_size = self.hidden_size + self.num_classes
        else:
            classifier_input_size = self.hidden_size
        print("classifier_input_size:", classifier_input_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

        self.to(self.device)  # Move the entire model to the specified device


    def forward(self, input_ids, attention_mask, token_ids=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input IDs for the transformer (batch_size, seq_length).
            attention_mask: Attention masks for the transformer (batch_size, seq_length).
            token_ids: Token IDs for WCE lookup (optional).
        
        Returns:
            logits: Classification logits (batch_size, num_classes).
        """

        # Move all inputs to the model's device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if token_ids is not None:
            token_ids = token_ids.to(self.device)
            print("token_ids:", type(token_ids), token_ids.shape)
        else:
            print("** warning: token_ids is None **")

        # Transformer forward pass
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = transformer_output.last_hidden_state  # [batch_size, seq_length, hidden_size]

        # Mean pooling across the sequence
        attention_mask_expanded = attention_mask.unsqueeze(-1)
        token_embeddings = token_embeddings * attention_mask_expanded
        pooled_token_output = token_embeddings.sum(dim=1) / attention_mask_expanded.sum(dim=1)
        print("pooled_token_output:", type(pooled_token_output), pooled_token_output.shape)

        # Integrate WCE embeddings
        if self.use_supervised and self.wce_matrix is not None and token_ids is not None:

            # Lookup WCE embeddings
            wce_embeddings = self.wce_matrix[token_ids]  # [batch_size, seq_length, num_classes]

            # Mask WCE embeddings
            wce_embeddings = wce_embeddings * attention_mask_expanded

            # Mean pooling for WCE embeddings
            pooled_wce_output = wce_embeddings.sum(dim=1) / attention_mask_expanded.sum(dim=1)  # [batch_size, num_classes]
            print("pooled_wce_output:", type(pooled_wce_output), pooled_wce_output.shape)
            
            if self.combination == "concat":
                # Concatenate token embeddings with WCE embeddings
                pooled_output = torch.cat((pooled_token_output, pooled_wce_output), dim=1)  # [batch_size, hidden_size + num_classes]
            elif self.combination == "add":
                # Add token embeddings and WCE embeddings
                pooled_output = pooled_token_output + pooled_wce_output  # [batch_size, hidden_size]
            else:
                raise ValueError(f"Invalid combination method: {self.combination}")
        else:
            pooled_output = pooled_token_output  # No WCE integration
        print("pooled_output:", type(pooled_output), pooled_output.shape)

        # Classification
        logits = self.classifier(pooled_output)  # Shape: [batch_size, num_classes]
        
        return logits



class TransformerWithWCEClassifierCLS(nn.Module):
    def __init__(self, transformer_model, num_classes, wce_dim=100, use_supervised=False):
        super(TransformerWithWCEClassifierCLS, self).__init__()
        self.transformer = transformer_model
        self.hidden_size = self.transformer.config.hidden_size
        self.num_classes = num_classes
        self.use_supervised = use_supervised
        self.wce_dim = wce_dim

        # Add a linear layer for projecting WCEs if needed
        if self.use_supervised:
            self.wce_projector = nn.Linear(self.wce_dim, self.hidden_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2 if use_supervised else self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, input_ids, attention_mask, wce=None):
        # Transformer forward pass
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = transformer_output.last_hidden_state[:, 0, :]  # CLS token embedding

        if self.use_supervised and wce is not None:
            # Project WCEs to match hidden_size
            wce = self.wce_projector(wce)
            # Concatenate WCEs with CLS token embedding
            pooled_output = torch.cat((pooled_output, wce), dim=1)

        # Classification
        logits = self.classifier(pooled_output)
        return logits


# Initialize the model
def initialize_model(model_name, num_classes, use_supervised=False, wce_dim=100):

    transformer_model = MODEL_MAP.get(model_name, model_name)
    tokenizer = BertTokenizerFast.from_pretrained(transformer_model)
    model = BertModel.from_pretrained(transformer_model)

    classifier = TransformerWCEClassifier(model, num_classes, wce_dim, use_supervised)
    
    return classifier, tokenizer





class BertClassifier(nn.Module):
    def __init__(self, num_classes, wce_matrix=None, model_name='bert-base-uncased', dropout=0.5):
        """
        A BERT-based classifier with optional WCE embedding integration.
        
        Args:
            num_classes (int): Number of classes in the dataset.
            wce_matrix (torch.Tensor): Precomputed WCE matrix.
            model_name (str): Pretrained BERT model name.
            dropout (float): Dropout rate.
        """
        super(BertClassifier, self).__init__()
        print(f'BertClassifier:__init__...num_classes: {num_classes}, model_name: {model_name}, dropout: {dropout}')
        
        # Load BERT model
        self.bert = BertModel.from_pretrained(model_name, cache_dir=VECTOR_CACHE+'BERT')

        # Dynamically determine BERT's hidden size
        self.hidden_size = self.bert.config.hidden_size

        # WCE embedding layer
        if wce_matrix is not None:
            self.wce_embedding = nn.Embedding.from_pretrained(wce_matrix, freeze=False)
            self.wce_projection = nn.Sequential(
                nn.Linear(wce_matrix.size(1), self.hidden_size),  # Project to hidden size
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),   # Add another layer for better representation
                nn.LayerNorm(self.hidden_size)                  # Normalize the final output
            )
        else:
            self.wce_embedding = None
            self.wce_projection = None

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Final classifier
        self.classifier = nn.Linear(self.hidden_size, num_classes)


    def forward(self, input_ids, attention_mask, token_ids=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs for BERT.
            attention_mask (torch.Tensor): Attention masks for BERT.
            token_ids (torch.Tensor): Token IDs for WCE embeddings (optional).
        
        Returns:
            logits (torch.Tensor): Classification logits.
        """
        # BERT forward pass
        _, bert_pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        # WCE embeddings (if provided)
        if self.wce_embedding is not None and token_ids is not None:
            wce_embeds = self.wce_embedding(token_ids)             # [batch_size, seq_length, wce_dim]
            wce_embeds = self.wce_projection(wce_embeds)           # Project to hidden size
            wce_embeds = wce_embeds.mean(dim=1)                    # Pool across sequence length
            combined_output = bert_pooled_output + wce_embeds      # Combine with BERT output
        else:
            combined_output = bert_pooled_output

        # Apply dropout and pass through classifier
        dropout_output = self.dropout(combined_output)
        logits = self.classifier(dropout_output)

        return logits





class BertDataset(Dataset):

    def __init__(self, texts, targets, tokenizer, max_length, vectorizer_vocab=None):

        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.vectorizer_vocab = vectorizer_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        target = self.targets[index]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )

        # Generate token IDs for WCE
        token_ids = None
        if self.vectorizer_vocab:
            token_ids = [
                self.vectorizer_vocab.get(token, 0)  # Use 0 for unknown tokens
                for token in self.tokenizer.tokenize(text)
            ]
            token_ids = token_ids[:self.max_length]  # Truncate
            token_ids += [0] * (self.max_length - len(token_ids))  # Pad

        # Debug: Print values
        """
        print(f"Index: {index}, Text: {text}")
        print(f"Token IDs: {token_ids}")
        print(f"Input IDs: {inputs['input_ids']}")
        print(f"Mask: {inputs['attention_mask']}")
        print(f"Target: {target}")
        """
        return {
            'ids': torch.tensor(inputs["input_ids"], dtype=torch.long),
            'mask': torch.tensor(inputs["attention_mask"], dtype=torch.long),
            'token_ids': torch.tensor(token_ids, dtype=torch.long) if token_ids else None,
            'target': torch.tensor(target, dtype=torch.long),
        }






# Training script
def train_model(model, tokenizer, train_dataloader, val_dataloader, device, epochs=3, lr=2e-5):
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            wce = batch.get('wce', None)
            if wce is not None:
                wce = wce.to(device)

            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, wce=wce)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

        # Validation step
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
                wce = batch.get('wce', None)
                if wce is not None:
                    wce = wce.to(device)

                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, wce=wce)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print(classification_report(all_labels, all_preds, digits=4))


def validate(model, val_loader, device):
    """
    Validates the Transformer-based model during training.

    Args:
    - model: Transformer-based classifier.
    - val_loader: DataLoader for validation data.
    - device: Device to run the model on (CPU/GPU).

    Returns:
    float: Macro F1 score on validation data.
    """
    #print("\n\tvalidating...")

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_ids = batch.get('token_ids', None)

            if token_ids is not None:
                token_ids = token_ids.to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_ids=token_ids)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate macro F1 score
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return macro_f1


def train(model, train_loader, val_loader, device, epochs=3, learning_rate=2e-5, patience=3):
    """
    Trains the Transformer-based model with optional WCE integration.

    Args:
    - model: Transformer-based classifier (e.g., TransformerWithWCEClassifier).
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - device: Device to run the model on (CPU/GPU).
    - epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.
    - patience: Early stopping patience.

    Returns:
    None
    """

    print("\n\ttraining...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
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

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader)}")

        # Validate after each epoch
        val_f1 = validate(model, val_loader, device)
        print(f"Validation F1 Score: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "../out/best_model.pth")
            print("saved model to ../out/best_model.pth...")
        else:
            patience_counter += 1
            print(f"No improvement in validation F1 for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


def evaluate(model, test_loader, device):
    """
    Evaluates the Transformer-based model with optional WCE integration.

    Args:
    - model: Trained Transformer-based classifier.
    - test_loader: DataLoader for test data.
    - device: Device to run the model on (CPU/GPU).

    Returns:
    dict: Evaluation metrics (accuracy, macro/micro F1 scores).
    """

    print("\n\tevaluating...")
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
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')

    print(f'Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}')

    """
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    """

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1
    }



def train_old(model, train_data, val_data, tokenizer, vectorizer, max_length, learning_rate, epochs, device, batch_size, patience=PATIENCE):
    """
    Train the model using the given training and validation data with early stopping.
    """
    print("\n\ttraining...")

    train_texts, train_targets = train_data
    val_texts, val_targets = val_data

    # Create datasets
    train_dataset = BertDataset(train_texts, train_targets, tokenizer, max_length, vectorizer.vocabulary_)
    val_dataset = BertDataset(val_texts, val_targets, tokenizer, max_length, vectorizer.vocabulary_)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch_num in range(epochs):
        model.train()
        total_acc_train = 0
        total_loss_train = 0

        # Collect predictions and labels for F1 score
        all_preds_train = []
        all_labels_train = []

        for train_batch in tqdm(train_dataloader, desc=f"Epoch {epoch_num + 1}/{epochs}"):
            ids = train_batch['ids'].to(device)
            mask = train_batch['mask'].to(device)
            token_ids = train_batch['token_ids'].to(device) if 'token_ids' in train_batch else None
            targets = train_batch['target'].to(device)

            targets = targets.long()

            optimizer.zero_grad()
            outputs = model(ids, mask, token_ids=token_ids)

            loss = criterion(outputs, targets)
            total_loss_train += loss.item()

            preds = outputs.argmax(dim=1)
            acc = (preds == targets).sum().item()
            total_acc_train += acc

            # Collect predictions and labels for F1
            all_preds_train.extend(preds.cpu().numpy())
            all_labels_train.extend(targets.cpu().numpy())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Compute F1 scores
        train_macro_f1 = f1_score(all_labels_train, all_preds_train, average='macro')
        train_micro_f1 = f1_score(all_labels_train, all_preds_train, average='micro')

        model.eval()
        total_acc_val = 0
        total_loss_val = 0

        # Collect predictions and labels for F1 score in validation
        all_preds_val = []
        all_labels_val = []

        with torch.no_grad():
            for val_batch in val_dataloader:
                ids = val_batch['ids'].to(device)
                mask = val_batch['mask'].to(device)
                token_ids = val_batch['token_ids'].to(device) if 'token_ids' in val_batch else None
                targets = val_batch['target'].to(device)

                targets = targets.long()
                outputs = model(ids, mask, token_ids=token_ids)

                loss = criterion(outputs, targets)
                total_loss_val += loss.item()

                preds = outputs.argmax(dim=1)
                acc = (preds == targets).sum().item()
                total_acc_val += acc

                # Collect predictions and labels for F1
                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(targets.cpu().numpy())

        # Compute F1 scores
        val_macro_f1 = f1_score(all_labels_val, all_preds_val, average='macro')
        val_micro_f1 = f1_score(all_labels_val, all_preds_val, average='micro')

        print(f"Epoch {epoch_num + 1}/{epochs} | "
              f"Train Loss: {total_loss_train / len(train_dataset):.3f} | "
              f"Train Accuracy: {total_acc_train / len(train_dataset):.3f} | "
              f"Train Macro F1: {train_macro_f1:.4f} | Train Micro F1: {train_micro_f1:.4f} | "
              f"Val Loss: {total_loss_val / len(val_dataset):.3f} | "
              f"Val Accuracy: {total_acc_val / len(val_dataset):.3f} | "
              f"Val Macro F1: {val_macro_f1:.4f} | Val Micro F1: {val_micro_f1:.4f}")

        # Early stopping logic
        if total_loss_val / len(val_dataset) < best_val_loss:
            best_val_loss = total_loss_val / len(val_dataset)
            epochs_without_improvement = 0
            print("Validation loss improved. Saving model...")
            torch.save(model.state_dict(), '../out/best_model.pth')  # Save the best model
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break



def evaluate_old(model, test_data, tokenizer, vectorizer, max_length, batch_size, device):
    """
    Evaluate the model using the given test data and vectorizer vocabulary.
    
    Args:
        model: The trained model to evaluate.
        test_data: Tuple (texts, targets) for evaluation.
        tokenizer: Tokenizer for BERT.
        vectorizer: vectorizer used to generate WCE token IDs (optional).
        max_length: Maximum token length for BERT tokenization.
        batch_size: Batch size for evaluation.
        device: Device to run the evaluation (CPU/GPU).
    """

    print("\n\tevaluating...")

    test_texts, test_targets = test_data

    # Create test dataset and dataloader
    test_dataset = BertDataset(test_texts, test_targets, tokenizer, max_length, vectorizer_vocab=vectorizer.vocabulary_)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    total_acc_test = 0
    total_loss_test = 0
    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, desc="Evaluating model...", leave=True):    
            ids = test_batch['ids'].to(device)
            mask = test_batch['mask'].to(device)
            token_ids = test_batch['token_ids'].to(device) if 'token_ids' in test_batch else None
            targets = test_batch['target'].to(device)

            targets = targets.long()

            outputs = model(ids, mask, token_ids=token_ids)
            
            loss = criterion(outputs, targets)
            total_loss_test += loss.item()

            preds = outputs.argmax(dim=1)
            acc = (preds == targets).sum().item()
            total_acc_test += acc

            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # Compute F1 scores
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')

    print(f"Test Loss: {total_loss_test / len(test_dataset):.3f} | "
          f"Test Accuracy: {total_acc_test / len(test_dataset):.3f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")

    return {
        "loss": total_loss_test / len(test_dataset),
        "accuracy": total_acc_test / len(test_dataset),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }


def classify_old(opt, device, batch_size=MPS_BATCH_SIZE, epochs=EPOCHS, max_length=MAX_LENGTH, model_name='bert-base-uncased'):
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

    print("\n\tclassifying...")

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

    """
    print("train_data:", type(train_data), len(train_data))
    print("train_data[0]:", type(train_data[0]), train_data[0])
    print("train_target:", type(train_target), len(train_target))
    print("train_target[0]:", type(train_target[0]), train_target[0].shape, train_target[0])

    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), len(labels_test))
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])
    """

    print("class_type:", class_type)
    print("num_classes:", num_classes)
    print("target_names:", target_names)

    # Split train into train and validation
    texts_train, texts_val, labels_train, labels_val = train_test_split(train_data, train_target, test_size=VAL_SIZE, random_state=RANDOM_SEED)

    print("texts_train:", type(texts_train), len(texts_train))
    #print("texts_train[0]:", type(texts_train[0]), texts_train[0])
    print("labels_train:", type(labels_train), len(labels_train))
    #print("labels_traint[0]:", type(labels_train[0]), labels_train[0].shape, labels_train[0])

    print("texts_val:", type(texts_val), len(texts_val))
    #print("texts_val[0]:", type(texts_val[0]), texts_val[0])
    print("labels_val:", type(labels_val), len(labels_val))
    #print("labels_val[0]:", type(labels_val[0]), labels_val[0].shape, labels_val[0])

    print("test_data:", type(test_data), len(test_data))
    #print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), len(labels_test))
    #print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])


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

    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names))
    # Set the pad_token_id in the model configuration
    hf_model.config.pad_token_id = tokenizer.pad_token_id
    hf_model.to(device)
    print("model:\n", hf_model)
    """

    
    """    
    tokenizer = None

    # Load the appropriate tokenizer
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name, cache_dir=VECTOR_CACHE + '/RoBERTa')
    elif 'bert' in model_name:
        tokenizer = BertTokenizerFast.from_pretrained(model_name, cache_dir=VECTOR_CACHE + '/BERT')
    else:
        print("Invalid model name. Please provide a valid BERT or RoBERTa model.")
        return
    """

    print("model_name:", model_name)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=VECTOR_CACHE + '/BERT')
    tokenizer.do_lower_case = True  # Ensure lowercase matches vectorizer's behavior
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token if missing
    print("tokenizer:\n", tokenizer)    


    def vectorize(texts_train, texts_val, texts_test, tokenizer):

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

    # Supervised embeddings (WCEs)
    wce_matrix = None
    
    if opt.supervised:
        print(f'computing supervised embeddings...')
        Xtr = Xtr
        Ytr = one_hot_labels_train
        print("\tXtr:", type(Xtr), Xtr.shape)
        print("\tYtr:", type(Ytr), Ytr.shape)

        WCE = get_supervised_embeddings(Xtr, Ytr, method=opt.supervised_method,
                                         max_label_space=opt.max_label_space,
                                         dozscore=(not opt.nozscore),
                                         transformers=True)

        # Adjust WCE matrix size
        num_missing_rows = vec_vocab_size - WCE.shape[0]
        WCE = np.vstack((WCE, np.zeros((num_missing_rows, WCE.shape[1]))))
        wce_matrix = torch.from_numpy(WCE).float()
        print('\t[supervised-matrix]', wce_matrix.shape)

    model = BertClassifier(num_classes=num_classes, wce_matrix=wce_matrix, model_name=model_name, dropout=opt.dropprob)    
    print("model:\n", model)
    
    # Train model
    train(
        model=model,
        train_data=(train_data, train_target),
        val_data=(texts_val, labels_val),
        tokenizer=tokenizer,
        vectorizer=vectorizer,
        max_length=MAX_LENGTH,
        learning_rate=LR,
        epochs=epochs,
        device=device,
        batch_size=MPS_BATCH_SIZE,
        patience=PATIENCE
    )

    # Evaluate the model on test data
    evaluation_metrics = evaluate(
        model=model,
        test_data=(test_data, labels_test),
        tokenizer=tokenizer,
        vectorizer=vectorizer,
        max_length=MAX_LENGTH,
        batch_size=MPS_BATCH_SIZE,
        device=device
    )

    print("Evaluation Metrics:", evaluation_metrics)
    

    """
    datapath = DATASET_DIR + 'bbc-news/BBC News Train.csv'

    df = pd.read_csv(datapath)
    print(df.head())

    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                        [int(.8*len(df)), int(.9*len(df))])

    labels = {'business':0,
          'entertainment':1,
          'sport':2,
          'tech':3,
          'politics':4
          }

    print("labels:", type(labels), len(labels), labels)

    print("train, val, test:")
    print(len(df_train),len(df_val), len(df_test))

    train_orig(model, df_train, df_val, learning_rate=LR, epochs=EPOCHS, device=device, batch_size=MPS_BATCH_SIZE)

    evaluate_orig(model, df_test, batch_size=MPS_BATCH_SIZE, device=device)
    """




def classify(opt, device, batch_size=MPS_BATCH_SIZE, epochs=EPOCHS, max_length=MAX_LENGTH, model_name='bert-base-uncased'):
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

    """
    print("train_data:", type(train_data), len(train_data))
    print("train_data[0]:", type(train_data[0]), train_data[0])
    print("train_target:", type(train_target), len(train_target))
    print("train_target[0]:", type(train_target[0]), train_target[0].shape, train_target[0])

    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), len(labels_test))
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])
    """

    print("class_type:", class_type)
    print("num_classes:", num_classes)
    print("target_names:", target_names)

    # Split train into train and validation
    texts_train, texts_val, labels_train, labels_val = train_test_split(train_data, train_target, test_size=VAL_SIZE, random_state=RANDOM_SEED)

    print("texts_train:", type(texts_train), len(texts_train))
    #print("texts_train[0]:", type(texts_train[0]), texts_train[0])
    print("labels_train:", type(labels_train), len(labels_train))
    #print("labels_traint[0]:", type(labels_train[0]), labels_train[0].shape, labels_train[0])

    print("texts_val:", type(texts_val), len(texts_val))
    #print("texts_val[0]:", type(texts_val[0]), texts_val[0])
    print("labels_val:", type(labels_val), len(labels_val))
    #print("labels_val[0]:", type(labels_val[0]), labels_val[0].shape, labels_val[0])

    print("test_data:", type(test_data), len(test_data))
    #print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), len(labels_test))
    #print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])

    print("model_name:", model_name)

    print("tokenizing texts...")

    # Tokenizer initialization
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=VECTOR_CACHE + '/BERT')
    print("tokenizer:\n", tokenizer)

    # Tokenize inputs
    train_encodings = tokenizer(texts_train, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    val_encodings = tokenizer(texts_val, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    test_encodings = tokenizer(test_data, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    def vectorize(texts_train, texts_val, texts_test, tokenizer):

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

    # Dataset preparation
    class TextDataset(Dataset):
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

    max_length = tokenizer.model_max_length
    print("Maximum sequence length:", max_length)

    train_dataset = TextDataset(train_encodings, labels_train, vectorizer.vocabulary_, tokenizer, max_length)
    val_dataset = TextDataset(val_encodings, labels_val, vectorizer.vocabulary_, tokenizer, max_length)
    test_dataset = TextDataset(test_encodings, labels_test, vectorizer.vocabulary_, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    transformer_model = BertModel.from_pretrained(model_name, cache_dir=VECTOR_CACHE + '/BERT')
    transformer_model.to(device)
    print("transformer_model:\n", transformer_model)

    model = TransformerWCEClassifier(
        transformer_model=transformer_model,
        num_classes=num_classes,
        device=device,
        wce_matrix=wce_train,
        use_supervised=opt.supervised,
        combination='concat'
    ).to(device)
    print("TransformerWCEModel:\n", model)

    # Train the model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=LR,
        patience=PATIENCE
    )

    # Evaluate the model
    evaluation_metrics = evaluate(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    print("Evaluation Metrics:", evaluation_metrics)



    
if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Text Classification with BERT.")
    
    parser.add_argument('--dataset', required=True, type=str, default='20newsgroups', help='Dataset to use: 20newsgroups or bbc-news.')
    
    parser.add_argument('--learner', required=True, type=str, default='ft', 
                        help='Choose the learner: nn, ft, svm, lr, nb.')

    parser.add_argument('--net', type=str, default='lstm', metavar='str',
                        help=f'net, one in (CNN, LSTM, ATTN)')
    
    parser.add_argument('--dropprob', type=float, default=0.5, metavar='[0.0, 1.0]',
                        help='dropout probability (default: 0.5)')
    
    parser.add_argument('--droptype', type=str, default='sup', metavar='DROPTYPE',
                        help=f'chooses the type of dropout to apply after the embedding layer. Default is "sup" which '
                             f'only applies to word-class embeddings (if present). Other options include "none" which '
                             f'does not apply dropout (same as "sup" with no supervised embeddings), "full" which '
                             f'applies dropout to the entire embedding, or "learn" that applies dropout only to the '
                             f'learnable embedding.')
    
    parser.add_argument('--pretrained', type=str, default=None, metavar='str',
                        help='pretrained embeddings, one of "bert", "roberta", "xlnet", "gpt2", or "llama" (default None)')

    parser.add_argument('--vtype', type=str, default='tfidf', metavar='N', 
                        help=f'dataset base vectorization strategy, in [tfidf, count]')

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
    
    parser.add_argument('--embedding-dir', type=str, default=VECTOR_CACHE, metavar='str',
                        help=f'path where to load and save document embeddings')
    
    parser.add_argument('--bert-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to BERT pretrained vectors, used only with --pretrained bert')
    
    parser.add_argument('--roberta-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to RoBERTa pretrained vectors, used only with --pretrained roberta')

    parser.add_argument('--static', action='store_true', help='keep the underlying pretrained model static (ie no unfrozen layers)')

    parser.add_argument('--optimc', action='store_true', help='Optimize classifier with GridSearchCV.')

    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs, used with --learner nn.')

    parser.add_argument('--log-file', type=str, default='../log/lc_nn_test.test', metavar='str',
                        help='path to the log logger output file')
    
    parser.add_argument('--cm', action='store_true', help='Generate confusion matrix.')
    
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
    
        
    # Check for CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda")

        # Number of GPUs available
        num_gpus = torch.cuda.device_count()
        print('Number of GPUs:', num_gpus)

        num_replicas = torch.cuda.device_count()
        print(f'Using {num_replicas} GPU(s)')
        
        # If using multiple GPUs, use DataParallel or DistributedDataParallel
        """
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)    
        """

    # Check for MPS availability (for Apple Silicon)
    elif torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("mps")

        num_replicas = 1  # Use CPU or single GPU
        
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"          # disable memory upper limit

    # Default to CPU if neither CUDA nor MPS is available
    else:
        print("Neither CUDA nor MPS is available, using CPU")
        device = torch.device("cpu")
        
        num_replicas = 1  # Use CPU or single GPU
        
    print(f"Using device: {device}")

    if (args.pretrained == 'bert'):
        model_name = BERT_MODEL
        model_path = args.bert_path
    elif (args.pretrained == 'roberta'):
        model_name = ROBERTA_MODEL
        model_path = args.roberta_path

    print("model_name:", model_name)
    print("model_path:", model_path)

    print("args.epoch:", args.epochs)
    print("args.learner:", args.learner)

    start = time()                      # start the clock

    if (args.learner == 'nn'):
    
        classify(
            args, 
            device, 
            batch_size=MPS_BATCH_SIZE, 
            epochs=args.epochs,
            max_length=MAX_LENGTH,
            model_name=model_name
        )

    else:
        print(f"Invalid learner '{args.learner}'")

    print(f'Test time = {time() - start}')
        