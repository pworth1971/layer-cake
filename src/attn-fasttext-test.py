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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        u_t = torch.tanh(self.W(x)) @ self.u
        at = self.softmax(u_t)
        weighted_sum = torch.sum(x * at.unsqueeze(-1), dim=1)
        return weighted_sum

class ToxicModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size, maxlen, context_dim=100):
        super(ToxicModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2, context_dim)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        out = self.fc(attn_out)
        return torch.sigmoid(out)



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10):
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
        val_roc_auc = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_roc_auc += roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())

        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val ROC AUC: {val_roc_auc:.4f}")


from data.lc_dataset import LCDataset, loadpt_data


def main(max_features=60000, maxlen=250, embed_size=300, batch_size=256, epochs=10, lr=1e-3, hidden_size=64):
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

    pretrained_vectors = lcd.lcr_model
    pretrained_vectors.show()

    #word_index = vectorizer.vocabulary_
    word_index = lcd.vocabulary
    #embedding_matrix = create_embedding(emb_file, word_index, embed_size, max_features)
    embedding_matrix = lcd.embedding_vocab_matrix
    print("embedding_matrix:", type(embedding_matrix), embedding_matrix.shape)

    # Convert embedding_matrix to a PyTorch tensor
    embedding_matrix_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
    print("embedding_matrix_tensor:", type(embedding_matrix_tensor), embedding_matrix_tensor.shape)

    #X_train, X_val, y_train, y_val = train_test_split(X_t, y, test_size=0.2, random_state=1337)
    X_train, X_val, y_train, y_val = train_test_split(lcd.Xtr_vectorized, lcd.ytr_encoded, test_size=0.2, random_state=1337)
    
    train_dataset = CustomDataset(X_train, y_train, lcd.vectorizer, maxlen)
    val_dataset = CustomDataset(X_val, y_val, lcd.vectorizer, maxlen)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = lcd.lcr_model.device
    print("device:", device)
    
    #model = ToxicModel(embedding_matrix, hidden_size, len(list_classes), maxlen)
    model = ToxicModel(embedding_matrix_tensor, hidden_size, len(lcd.target_names), maxlen)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.3, min_lr=1e-7)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=epochs)


if __name__ == "__main__":
    main()
