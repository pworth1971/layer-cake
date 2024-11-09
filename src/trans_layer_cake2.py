import argparse
import os
import numpy as np
import pandas as pd

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

from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import transformers     # Hugging Face transformers
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from torch.optim import Adam
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR

from util.metrics import evaluation_nn




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

BATCH_SIZE = 32

MAX_LENGTH = 512  # Max sequence length for the transformer models

TEST_SIZE = 0.3
VAL_SIZE = 0.3

DATASET_DIR = "../datasets"
VECTOR_CACHE = "../.vector_cache"

RANDOM_SEED = 42



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
            random_state = 1
        )

        # reset indeces
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type
    
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

# Tokenize and build vocabulary with Hugging Face transformers
def tokenize_and_build_vocab(train_data, val_data, test_data, model_name="bert-base-uncased", max_length=MAX_LENGTH, cache_dir="../.vector_cache"):

    print("Tokenizing and building vocabulary...")

    # Load tokenizer from the local cache directory
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("tokenizer:", tokenizer)

    # Tokenize and encode sequences with padding and truncation
    def encode_texts(data):
        return [
            torch.tensor(tokenizer.encode(text, max_length=max_length, padding="max_length", truncation=True), dtype=torch.long)
            for text in data
        ]
    
    train_sequences = encode_texts(train_data)
    val_sequences = encode_texts(val_data)
    test_sequences = encode_texts(test_data)
    
    # Pad sequences to ensure uniform length
    train_padded = pad_sequence(train_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
    val_padded = pad_sequence(val_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
    test_padded = pad_sequence(test_sequences, batch_first=True, padding_value=tokenizer.pad_token_id)

    vocab_size = tokenizer.vocab_size

    return train_padded, val_padded, test_padded, vocab_size


def split_data(train_data, train_target, val_size=VAL_SIZE, random_seed=RANDOM_SEED):

    train_data, val_data, train_target, val_target = train_test_split(
        train_data, 
        train_target, 
        test_size=val_size, 
        random_state=random_seed, 
        shuffle=True
        #stratify=train_target
    )

    return (train_data, train_target), (val_data, val_target)


def stratified_split_data(train_data, train_target, val_size=VAL_SIZE, random_seed=RANDOM_SEED):

    train_data, val_data, train_target, val_target = train_test_split(
        train_data, train_target, test_size=val_size, random_state=random_seed, stratify=train_target
    )

    return (train_data, train_target), (val_data, val_target)


class TextClassifier(torch.nn.Module):
    def __init__(self, net_type, embedding_dim, hidden_size, num_classes, model_name=None, model_path="../.vector_cache"):
        super(TextClassifier, self).__init__()
        
        if model_name:
            self.embedding = AutoModel.from_pretrained(model_name, cache_dir=model_path)
            self.uses_transformer = True
            self.projection = torch.nn.Linear(self.embedding.config.hidden_size, hidden_size)  # Project from 768 to 512

            # Unfreeze all layers in the transformer model
            for param in self.embedding.parameters():
                param.requires_grad = True
        else:
            self.embedding = torch.nn.Embedding(embedding_dim, hidden_size)
            self.uses_transformer = False
        
        self.net_type = net_type
        if net_type == 'CNN':
            self.model = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        elif net_type == 'LSTM':
            self.model = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        elif net_type == 'ATTN':
            self.model = torch.nn.MultiheadAttention(hidden_size, num_heads=8)
        
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        # Transformer embedding extraction with attention mask
        if self.uses_transformer:
            embedding_output = self.embedding(x, attention_mask=attention_mask)
            # Use pooled output for classification directly
            x = self.projection(embedding_output.pooler_output)  
        else:
            x = self.embedding(x)
        
        if self.net_type == 'CNN':
            x = x.transpose(1, 2)  # For Conv1D compatibility
            x = torch.nn.functional.relu(self.model(x))
        elif self.net_type == 'LSTM':
            x, _ = self.model(x)  # Pass directly to LSTM
        elif self.net_type == 'ATTN':
            x, _ = self.model(x, x, x)
        
        return self.fc(x)  # Classification output






# Wrapper for TransformerTextClassifier with tokenizer integration
class TransformerTextClassifier(nn.Module):
    def __init__(self, net_type, embedding_dim, hidden_size, num_classes, model_name, model_path):
        super(TransformerTextClassifier, self).__init__()
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            cache_dir=model_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
        
        # Store padding token ID for attention mask creation
        self.pad_token_id = self.tokenizer.pad_token_id

        self.net_type = net_type

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)





# Updated create_attention_mask function
def create_attention_mask(input_ids, pad_token_id):
    return (input_ids != pad_token_id).long()

# Training loop with attention mask
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10, patience=5):
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        total_correct = 0
        total_labels = 0            # for multi-label case to count total labels
        
        # Initialize a progress bar for this epoch
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch")

        for batch_data, batch_labels in progress_bar:

            # Move data to the device
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Create attention mask for padding tokens
            attention_mask = create_attention_mask(batch_data, model.pad_token_id).to(device)

            #print("forward pass...")

            # Forward pass
            outputs = model(batch_data, attention_mask=attention_mask)
            
            # Apply label smoothing
            alpha = 0.1
            smoothed_labels = batch_labels * (1 - alpha) + alpha / model.model.config.num_labels

            # Compute loss with focal loss if applicable, else use BCE
            loss = loss_fn(outputs, smoothed_labels)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute loss (using Focal Loss here as an alternative)
            #loss = focal_loss(outputs, smoothed_labels)  # Replace with `loss_fn(outputs, smoothed_labels)` if using BCE
            #total_loss += loss.item()

            """
            loss = loss_fn(outputs, batch_labels)
            total_loss += loss.item()
            """

            """
            print("outputs:", outputs)
            print("batch_labels:", batch_labels)
            print("loss:", loss)
            print("total_loss:", total_loss)
            """
            
            PRED_THRSHOLD = 0.3

            # Calculate accuracy
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):  # Single-label
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == batch_labels).sum().item()
            else:  # Multi-label
                predicted = (outputs > PRED_THRSHOLD).float()
                total_correct += (predicted == batch_labels).sum().item()
                #print("total_correct:", total_correct)
                total_labels += batch_labels.numel()  # Total number of labels

            # Update progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate overall training accuracy
        if isinstance(loss_fn, torch.nn.CrossEntropyLoss):  # Single-label
            train_accuracy = total_correct / len(train_loader.dataset)
        else:  # Multi-label
            train_accuracy = total_correct / total_labels

        val_accuracy, val_macro_f1, val_micro_f1, f1_by_class = evaluate_model(
            model, val_loader, class_type, loss_fn, device
        )
        print(f"Epoch {epoch + 1}: Train Loss = {total_loss:.4f}, Train Acc = {train_accuracy:.4f}, Val Acc = {val_accuracy:.4f}, Val Macro-F1 = {val_macro_f1:.4f}, Val Micro-F1 = {val_micro_f1:.4f}")
        
        # Early stopping based on validation macro F1 score
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break



# Function to print confusion matrix to stdout
def print_confusion_matrix(conf_matrix, labels, title):
    print(f"\n{title}")
    print(" " * 10 + "Predicted")
    print(" " * 8 + " ".join(f"{label:>5}" for label in labels))
    print("True")
    for i, row in enumerate(conf_matrix):
        row_str = " ".join(f"{val:>5}" for val in row)
        print(f"{labels[i]:>5}  {row_str}")





# Evaluation function for single-label and multi-label cases
def evaluate_model(model, data_loader, class_type, loss_fn, device):

    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0
    total_correct = 0
    total_labels = 0  # For multi-label case

    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            attention_mask = create_attention_mask(batch_data, model.pad_token_id).to(device)

            # Forward pass with attention mask
            outputs = model(batch_data, attention_mask=attention_mask)
            loss = loss_fn(outputs, batch_labels.float() if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss) else batch_labels)
            total_loss += loss.item()

            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):  # Single-label
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == batch_labels).sum().item()
                all_labels.extend(batch_labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
            else:  # Multi-label
                predicted = (outputs > 0.5).float()
                total_correct += (predicted == batch_labels).sum().item()
                total_labels += batch_labels.numel()
                all_labels.extend(batch_labels.cpu().numpy().tolist())
                all_preds.extend(predicted.cpu().numpy().tolist())

    # Calculate overall metrics
    if isinstance(loss_fn, torch.nn.CrossEntropyLoss):  # Single-label
        accuracy = total_correct / len(data_loader.dataset)

        # Confusion matrix for single-label
        #conf_matrix = confusion_matrix(all_labels, all_preds)
        #print_confusion_matrix(conf_matrix, target_names, "Overall Confusion Matrix")
    else:  # Multi-label
        accuracy = total_correct / total_labels

        # Calculate confusion matrix for each label separately
        """
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        for i, label in enumerate(target_names):
            conf_matrix = confusion_matrix(all_labels[:, i], all_preds[:, i])
            print_confusion_matrix(conf_matrix, ["0", "1"], f"Confusion Matrix for {label}")
        """

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=1)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=1)

    f1_by_class = f1_score(all_labels, all_preds, average=None, zero_division=1)

    """
    # Display F1 scores by class
    print("F1 Scores by Class (Validation):")
    for i, f1 in enumerate(f1_by_class):
        print(f"  Class {i}: F1 Score = {f1:.4f}")
    """

    # Calculate precision, recall, f1, and support by class
    precision, recall, f1_by_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=1
    )

    # Display F1 scores by class along with precision, recall, and support
    """
    print("Class-wise Metrics (Validation):")
    for i in range(len(f1_by_class)):
        print(f"  Class {i}: Precision = {precision[i]:.4f}, Recall = {recall[i]:.4f}, F1 Score = {f1_by_class[i]:.4f}, Support = {support[i]}")
    """

    macrof1, microf1, acc, h_loss, precision, recall, j_index = evaluation_nn(all_labels, all_preds, classification_type=class_type)
    print("\n--Layer Cake Metrics--")
    print(f"Macro-F1 = {macrof1:.4f}, Micro-F1 = {microf1:.4f}, Accuracy = {acc:.4f}, H-loss = {h_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, Jaccard = {j_index:.4f}")

    return accuracy, macro_f1, micro_f1, f1_by_class






# Convert datasets to TensorDatasets and DataLoaders
def create_singlelabel_dataloader(data, target, batch_size=BATCH_SIZE, shuffle=True):

    tensor_data = data  # Already a padded tensor
    tensor_target = torch.tensor(target, dtype=torch.long)
    dataset = TensorDataset(tensor_data, tensor_target)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Helper function to prepare DataLoader inputs for multi-label classification
def create_multilabel_dataloader(data, target, batch_size=32, shuffle=True):
    # Ensure `data` is a tensor without copying if it already is
    if isinstance(data, torch.Tensor):
        data = data.clone().detach()
    else:
        data = torch.tensor(data)
    
    # Convert the target to float if using multi-label classification
    if isinstance(target, torch.Tensor):
        target = target.float()
    else:
        target = torch.tensor(target, dtype=torch.float32)
    
    dataset = TensorDataset(data, target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)





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


def calculate_f1_per_class(y_true, y_pred):
    """
    y_true and y_pred should be tensors of shape (num_samples, num_classes)
    where each entry is either 0 or 1, representing the presence of a label.
    """
    # Convert tensors to numpy for sklearn compatibility
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Calculate F1 score for each class
    f1_scores = f1_score(y_true, y_pred, average=None)  # average=None gives F1 for each class
    
    # Display F1 scores by class
    for i, score in enumerate(f1_scores):
        print(f"Class {i}: F1 Score = {score:.4f}")

    return f1_scores


# Helper function to calculate class weights for multi-label classification
def calculate_class_weights(train_target):
    # Convert train_target to a torch Tensor if it's not already
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32)
    
    # Sum occurrences for each class
    total_counts = torch.sum(train_target_tensor, dim=0)
    pos_weights = (len(train_target) - total_counts) / total_counts
    return pos_weights

# Define Focal Loss function
def focal_loss(logits, targets, alpha=0.25, gamma=2):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)  # prevents NaN
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()



# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Text Classification with Transformer Models.")
    
    # Dataset and learner arguments
    parser.add_argument('--dataset', required=True, type=str, choices=['20newsgroups', 'reuters21578', 'bbc-news', 'rcv1'], help='Dataset to use')
    parser.add_argument('--learner', required=True, type=str, choices=['nn', 'svm', 'lr', 'nb'], help='Choose the learner')
    parser.add_argument('--net', type=str, default='lstm', choices=['CNN', 'LSTM', 'ATTN'], help='Network architecture')
    
    # Training parameters
    parser.add_argument('--dropprob', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--pretrained', type=str, choices=['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2', 'llama'], help='Pretrained embeddings')
    parser.add_argument('--vtype', type=str, default='tfidf', choices=['tfidf', 'count'], help='Vectorization strategy')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--supervised', action='store_true', help='Use supervised embeddings')
    parser.add_argument('--learnable', type=int, default=0, help='Dimension of learnable embeddings')
    parser.add_argument('--tunable', action='store_true', help='Unfreeze pretrained embeddings')
    parser.add_argument('--dist', action='store_true', default=False, help='show class distribution plots')
    parser.add_argument('--hidden', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--static', action='store_true', help='Keep pretrained model static (freeze layers)')
    parser.add_argument('--log_file', type=str, default='../log/lc_nn_test.test', help='Path to log file')
    parser.add_argument('--cm', action='store_true', help='Generate confusion matrix')

    return parser.parse_args()



def main_old():

    print("\n\ttrans layer cake")

    args = parse_args()
    print("args:", args)
    
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

    # Split off validation data from training data using specified validation percentage
    (train_data, train_target), (val_data, val_target) = split_data(train_data, train_target, val_size=VAL_SIZE)

    # Get the full model identifier and cache directory path for tokenizer/model
    model_name, model_path = get_model_identifier(args.pretrained)
    print("model_name:", model_name)
    print("model_path:", model_path)
    
    # Tokenize and pad sequences
    train_data, val_data, test_data, vocab_size = tokenize_and_build_vocab(train_data, val_data, test_data, model_name=model_name, max_length=MAX_LENGTH, cache_dir=model_path)
    print(f"vocab_size: {vocab_size}")

    if (args.dist):
        # Show distribution by class for train, validation, and test datasets
        if (class_type in ['multi-label', 'multilabel']):
            train_distribution = show_multilabel_class_distribution(train_target, target_names, "Train Dataset")
            val_distribution = show_multilabel_class_distribution(val_target, target_names, "Validation Dataset")
            test_distribution = show_multilabel_class_distribution(test_target, target_names, "Test Dataset")
        else:
            train_distribution = show_class_distribution(train_target, target_names, "Train Dataset")
            val_distribution = show_class_distribution(val_target, target_names, "Validation Dataset")
            test_distribution = show_class_distribution(test_target, target_names, "Test Dataset")

    # Create DataLoaders depending upon the type of classification problem we are 
    # working with, as a function of the requirements for the underlying model, optimizer, etc
    if class_type == 'single-label':
        train_loader = create_singlelabel_dataloader(train_data, train_target, batch_size=BATCH_SIZE)
        val_loader = create_singlelabel_dataloader(val_data, val_target, batch_size=BATCH_SIZE)
        test_loader = create_singlelabel_dataloader(test_data, test_target, batch_size=BATCH_SIZE)    
    else:
        train_loader = create_multilabel_dataloader(train_data, train_target, batch_size=BATCH_SIZE)
        val_loader = create_multilabel_dataloader(val_data, val_target, batch_size=BATCH_SIZE)
        test_loader = create_multilabel_dataloader(test_data, test_target, batch_size=BATCH_SIZE)
        
    # Initialize the model
    """
    model = TextClassifier(
        net_type=args.net, 
        embedding_dim=vocab_size, 
        hidden_size=args.hidden, 
        num_classes=num_classes, 
        model_name=model_name,
        model_path=model_path,
    ).to(device)
    """

    model = TransformerTextClassifier(
        net_type=args.net, 
        embedding_dim=vocab_size, 
        hidden_size=args.hidden, 
        num_classes=num_classes, 
        model_name=model_name,
        model_path=model_path,
    ).to(device)

    # Apply Xavier uniform initialization
    model.apply(init_weights)
    print("model:", model)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("optimizer:", optimizer)

    # Define loss function with class weights for multi-label case
    if class_type == 'single-label':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        """
        pos_weights = calculate_class_weights(train_target).to(device)
        print("pos_weights:", pos_weights)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        """
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
    print("loss_fn:", loss_fn)

    # Train and evaluate
    print(f"Starting training on {device} for {args.epochs} epochs with learning rate {args.lr}")
    train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=args.epochs, patience=args.patience)
    
    # Final evaluation on test set
    test_accuracy, test_macro_f1, test_micro_f1, test_f1_by_class = evaluate_model(model, test_loader, class_type, loss_fn, device)
    print("\n--Final Test Metrics--")
    print(f"Test Accuracy = {test_accuracy:.4f}, Test Macro-F1 = {test_macro_f1:.4f}, Test Micro-F1 = {test_micro_f1:.4f}")




# Modify: Dataset class to handle single-label and multi-label formats
class LCDataset(Dataset):
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
    

class LCDatasetOld(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        # Squeeze to remove extra dimensions and prepare correct format
        item = {key: val.squeeze(0) for key, val in encoding.items()}  # Remove extra dimension
        item["labels"] = torch.tensor(labels, dtype=torch.float)
        return item


# Modify: Enhanced metrics function to handle single-label and multi-label
def compute_metrics(pred, threshold=0.5, class_type='multi-label'):
    labels = pred.label_ids
    
    if class_type == 'single-label':
        preds = np.argmax(pred.predictions, axis=1)
    else:
        preds = pred.predictions > threshold  # Adjust threshold for multi-label classification

    f1_micro = f1_score(labels, preds, average='micro', zero_division=1)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=1)
    
    # Additional metrics for multi-label, optional for single-label
    hamming = hamming_loss(labels, preds) if class_type == 'multi-label' else None
    jaccard = jaccard_score(labels, preds, average='samples', zero_division=1) if class_type == 'multi-label' else None
    accuracy = accuracy_score(labels, preds)
    precision_micro = precision_score(labels, preds, average='micro', zero_division=1)
    recall_micro = recall_score(labels, preds, average='micro', zero_division=1)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'hamming_loss': hamming,
        'jaccard_index': jaccard,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'accuracy': accuracy
    }


# Metrics function
def compute_metricsOld(pred, threshold=0.5):
    
    labels = pred.label_ids
    
    preds = pred.predictions > threshold  # Adjust threshold for multi-label classification
    
    f1_micro = f1_score(labels, preds, average='micro', zero_division=1)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=1)

    return {'f1_micro': f1_micro, 'f1_macro': f1_macro}



# Main
if __name__ == "__main__":

    print("\n\ttrans layer cake2")

    args = parse_args()
    print("args:", args)
    
    device = get_device()
    print("device:", device)
    
    torch.manual_seed(args.seed)

    # Load the Reuters dataset
    """
    train_data = fetch_reuters21578(subset='train')
    test_data = fetch_reuters21578(subset='test')

    texts_train, labels_train = train_data.data, train_data.target
    texts_test, labels_test = test_data.data, test_data.target
    """

    """
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
    """

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

    # Get the full model identifier and cache directory path for tokenizer/model
    model_name, model_path = get_model_identifier(args.pretrained)
    print("model_name:", model_name)
    print("model_path:", model_path)

    #MODEL_NAME = 'bert-base-uncased'  # Change this to 'roberta-base' or 'distilbert-base-uncased' if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names))

    model.to(device)
    print("model:", model)

    # Split train into train and validation
    texts_train, texts_val, labels_train, labels_val = train_test_split(train_data, train_target, test_size=0.2, random_state=42)

    print("texts_train:", type(texts_train), len(texts_train))
    print("texts_train[0]:", type(texts_train[0]), texts_train[0])
    print("labels_train:", type(labels_train), len(labels_train))
    print("labels_traint[0]:", type(labels_train[0]), labels_train[0].shape, labels_train[0])

    print("texts_val:", type(texts_val), len(texts_val))
    print("texts_val[0]:", type(texts_val[0]), texts_val[0])
    print("labels_val:", type(labels_val), len(labels_val))
    print("labels_val[0]:", type(labels_val[0]), labels_val[0].shape, labels_val[0])

    # Prepare datasets
    train_dataset = LCDataset(texts_train, labels_train, tokenizer, class_type=class_type)
    val_dataset = LCDataset(texts_val, labels_val, tokenizer, class_type=class_type)
    test_dataset = LCDataset(test_data, test_target, tokenizer, class_type=class_type)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='../out',
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir='../log',
        run_name='layer_cake'
    )

    # Modify: Trainer setup with `class_type` passed to `compute_metrics`
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: compute_metrics(pred, class_type=class_type),
    )

    # Train and evaluate
    trainer.train()
    results = trainer.evaluate(test_dataset)
    print("Evaluation Results:", results)


    """    
    # Prepare datasets
    train_dataset = LCDataset(texts_train, labels_train, tokenizer)
    val_dataset = LCDataset(texts_val, labels_val, tokenizer)
    test_dataset = LCDataset(test_data, test_target, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='../out',
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir='../log/',
        run_name='layer_cake',
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Training and evaluation
    trainer.train()
    results = trainer.evaluate(test_dataset)
    print(results)
    """

    # Generate detailed classification report on test set
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