#import os
import re
#import string

import fasttext

import numpy as np
import pandas as pd
from tqdm import tqdm

#from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
#from sklearn.feature_extraction.text import TfidfVectorizer

#from textacy.preprocess import preprocess_text

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import scipy.sparse as sp

import util.metrics as metrics



def create_embedding(emb_file, word_index, embed_size, max_features):
    if emb_file.endswith('bin'):
        embeddings_index = fasttext.load_model(emb_file)
    else:
        embeddings_index = pd.read_table(emb_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE, usecols=range(embed_size + 1))
    
    nb_words = min(max_features, len(word_index))
    mean, std = embeddings_index.values.mean(), embeddings_index.values.std()
    embedding_matrix = np.random.normal(mean, std, (nb_words, embed_size))

    for word, i in word_index.items():
        if i >= nb_words:
            continue
        if emb_file.endswith('bin') and embeddings_index.get_word_id(word) != -1:
            embedding_matrix[i] = embeddings_index.get_word_vector(word)
        elif word in embeddings_index.index:
            embedding_matrix[i] = embeddings_index.loc[word].values

    return torch.tensor(embedding_matrix, dtype=torch.float32)



class Attention(nn.Module):
    def __init__(self, input_dim, context_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, context_dim, bias=True)
        self.u = nn.Parameter(torch.randn(context_dim))
        self.softmax = nn.Softmax(dim=1)  # Softmax over the time steps (sequence length)

    def forward(self, x):
        # x is of shape (batch_size, sequence_length, hidden_size)
        u_t = torch.tanh(self.W(x)) @ self.u  # Compute attention scores
        # u_t shape: (batch_size, sequence_length)

        at = self.softmax(u_t)  # Apply softmax to get attention weights
        # at shape: (batch_size, sequence_length)

        # Weighted sum of the inputs, along the time axis (dim=1)
        weighted_sum = torch.sum(x * at.unsqueeze(-1), dim=1)
        # weighted_sum shape: (batch_size, hidden_size)
        return weighted_sum


class ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, context_dim=100):
        super(ClassificationModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2, context_dim)  # hidden_size * 2 for bidirectional LSTM
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_size * 2)
        attn_out = self.attention(lstm_out)  # attn_out shape: (batch_size, hidden_size * 2)

        # Pass the attention output to the linear layer
        out = self.fc(attn_out)  # out shape: (batch_size, output_size)
        return torch.sigmoid(out)


import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.is_sparse = sp.issparse(self.texts)  # Check if the input matrix is sparse

    def __len__(self):
        if self.is_sparse:
            return self.texts.shape[0]
        else:
            return len(self.texts)
    
    def __getitem__(self, idx):
        if self.is_sparse:
            text_row = self.texts[idx]
            # Convert sparse row to dense array
            text = text_row.toarray().squeeze()
        else:
            text = self.texts[idx]

        label = self.labels[idx]
        # No need for tokenization since the text is already vectorized
        tokens_tensor = torch.tensor(text, dtype=torch.float32)
        return tokens_tensor, torch.tensor(label, dtype=torch.float32)
"""

class TextClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TextClassificationModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)  # Bidirectional, hence hidden_size * 2
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # hidden_size * 2 for BiLSTM output

    def forward(self, x):
        # x is the vectorized input (shape: [batch_size, input_size])
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x.unsqueeze(1))  # lstm_out shape: [batch_size, sequence_length, hidden_size * 2]
        
        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # shape: [batch_size, sequence_length, 1]
        
        # Apply attention weights to LSTM output
        attn_output = torch.sum(attn_weights * lstm_out, dim=1)  # shape: [batch_size, hidden_size * 2]
        
        # Pass the result through the fully connected layer
        output = self.fc(attn_output)  # shape: [batch_size, num_classes]
        
        return torch.sigmoid(output)




class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts                                      # This is the vectorized text data (sparse or dense)
        print("CustomDataset: texts:", type(self.texts), self.texts.shape)
        
        self.labels = labels                                    # These are the labels (targets) for classification
        print("CustomDataset: labels:", type(self.labels), self.labels.shape)

        self.is_sparse = sp.issparse(self.texts)                # Check if the input matrix is sparse
        print("CustomDataset: is_sparse:", self.is_sparse)
    
    def __len__(self):
        # For sparse matrices, use .shape[0] to get the number of rows (documents)
        return self.texts.shape[0] if self.is_sparse else len(self.texts)

    def __getitem__(self, idx):
        if self.is_sparse:
            # Convert sparse row to dense array
            text = self.texts[idx].toarray().squeeze()
        else:
            text = self.texts[idx]  # Already dense data

        label = self.labels[idx]

        # Return the text (as float tensor) and label (as float tensor)
        return torch.tensor(text, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20):

    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Get the predicted class by taking the argmax of the logits
                predictions = torch.argmax(outputs, dim=1)

                # Append true and predicted labels for evaluation
                y_true.extend(labels.cpu().numpy())   # Ground truth
                y_pred.extend(predictions.cpu().numpy())  # Predictions

                loss = criterion(outputs, labels)
                val_loss += loss.item()

        scheduler.step(val_loss)

        # Evaluate the model using the evaluation_nn function
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = metrics.evaluation_nn(
            np.array(y_true), np.array(y_pred), classification_type='single-label')

        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Eval Metrics - Macro F1: {Mf1:.4f}, Micro F1: {mf1:.4f}, Accuracy: {accuracy:.4f}")
        #print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Hamming Loss: {h_loss:.4f}, Jaccard Index: {j_index:.4f}")



from data.lc_dataset import LCDataset, loadpt_data


def main(max_features=60000, maxlen=250, embed_size=300, batch_size=256, epochs=20, lr=1e-3, hidden_size=64):
    """
    train = pd.read_csv('train.csv').sample(frac=1)
    test = pd.read_csv("test.csv")
    
    train['comment_text'] = train['comment_text'].fillna("_na_").apply(text_cleanup)
    test['comment_text'] = test['comment_text'].fillna("_na_").apply(text_cleanup)
    
    list_sentences_train = train['comment_text'].tolist()
    list_sentences_test = test['comment_text'].tolist()
    y = train[list_classes].values

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_t = vectorizer.fit_transform(list_sentences_train)
    X_te = vectorizer.transform(list_sentences_test)
    """

    # Setup device prioritizing CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        
    else:
        device = torch.device("cpu")

    print("device:", device)

    #
    # Load the dataset and the associated (pretrained) embedding structures
    # to be fed into the model
    #                                                          
    lcd = loadpt_data(
        dataset='20newsgroups',                                                 # Dataset name
        vtype='tfidf',                                                          # Vectorization type
        pretrained='fasttext',                                                  # pretrained embeddings type
        embedding_path='../.vector_cache/fastText',                             # path to pretrained embeddings
        emb_type='subword'                                                      # embedding type (word or token)
        )                                                

    print("loaded LCDataset object:", type(lcd))
    print("lcd:", lcd.show())

    #X_train, X_val, y_train, y_val = train_test_split(X_t, y, test_size=0.2, random_state=1337)
    X_train, X_val, y_train, y_val = train_test_split(lcd.Xtr_vectorized, lcd.ytr_encoded, test_size=0.2, random_state=1337)
    
    # Create custom dataset
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    input_size = X_train.shape[1]  # Number of features in the vectorized data
    num_classes = len(lcd.target_names)  # The number of target labels/classes
    model = TextClassificationModel(input_size, hidden_size, num_classes)

    print("lcd.class_type:", lcd.class_type)

    # Set up loss function and optimizer
    if (lcd.class_type in ['singlelabel', 'single-label']):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.3, min_lr=1e-7)

    # Device configuration (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=epochs)




if __name__ == "__main__":
    main()
