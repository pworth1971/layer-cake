import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from transformers import LlamaTokenizer, LlamaModel
import torch
import os
from tqdm import tqdm
import pandas as pd
import pickle

DATASET_DIR = '../datasets/'
MAX_VOCAB_SIZE = 5000
llama_model_name = 'meta-llama/Llama-2-7b-hf'                    # dimension = 4096
PICKLE_DIR = '../pickles/'

dataset = 'bbc_news'
#dataset = '20newsgroups'

#
# tokens for LLAMA model access, must be requested from huggingface
#
from huggingface_hub import login

HF_TOKEN = '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'
HF_TOKEN2 = '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'


if (torch.backends.mps.is_built()):
    print("MPS is available")
    device = torch.device("mps")
else:
    print("MPS is not available")
    device = torch.device("cpu")
    
#
# BBC News 
#
if (dataset == 'bbc_news'):
    
    # Load the BBC News dataset
    print(f'\n\tloading BBC News dataset from {DATASET_DIR}...')

    for dirname, _, filenames in os.walk(DATASET_DIR + 'bbc-news'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    # Load datasets
    train_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Train.csv')
    test_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Test.csv')

    print("train_set:", train_set.shape)
    print("test_set:", test_set.shape)    

    print("train_set columns:", train_set.columns)
    #print("train_set:\n", train_set.head())

    #train_set['Category'].value_counts().plot(kind='bar', title='Category distribution in training set')
    #train_set['Category'].value_counts()
    print("Unique Categories:\n", train_set['Category'].unique())
    numCats = len(train_set['Category'].unique())
    print("# of categories:", numCats)

    X_raw = train_set['Text'].tolist()
    y = np.array(train_set['Category'])

    target_names = train_set['Category'].unique()
#
# 20 Newsgroups
#
elif (dataset == '20newsgroups'):
    
    print(f'\n\tloading 20 Newsgroups dataset...')        

    # Load the 20 newsgroups dataset
    newsgroups_data = fetch_20newsgroups(subset='all')
    X_raw, y = newsgroups_data.data, newsgroups_data.target

    target_names = newsgroups_data.target_names

print("X_raw:", type(X_raw), len(X_raw))
print("y", type(y), len(y))
print("target_names:", target_names)

# Split the dataset into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

print("X_train_raw:", type(X_train_raw), len(X_train_raw))
print("X_test_raw:", type(X_test_raw), len(X_test_raw))
print("y_train:", type(y_train), len(y_train))
print("y_test:", type(y_test), len(y_test))


# Load the pre-trained LLaMA model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)

# Ensure padding token is available
if tokenizer.pad_token is None:
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Reuse the end-of-sequence token for padding
    
model = LlamaModel.from_pretrained(llama_model_name).to(device)
print("model: ", model)


# Custom tokenizer function using LLaMA tokenizer
def llama_tokenizer(text):
    tokens = tokenizer.tokenize(text)
    return tokens

# Create a TF-IDF vectorizer with the custom tokenizer
tfidf_vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE, tokenizer=llama_tokenizer)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test_raw).toarray()

print("X_train_tfidf:", type(X_train_tfidf), X_train_tfidf.shape)
print("X_test_tfidf:", type(X_test_tfidf), X_test_tfidf.shape)


# Check if embeddings are already saved, if not, compute and save them
embeddings_file = f'{PICKLE_DIR}/{dataset}_llama_vocab_embeddings_{MAX_VOCAB_SIZE}.pkl'
print("embeddings_file:", embeddings_file)

if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        llama_vocab_embeddings = pickle.load(f)
else:
    # Create a vocabulary list of LLaMA encoded tokens based on the vectorizer vocabulary
    llama_vocab_embeddings = {}
    for token in tqdm(tfidf_vectorizer.get_feature_names_out(), desc="encoding Vocabulary using LlaMa pretrained embeddings..."):
        input_ids = tokenizer.encode(token, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model(input_ids)
        llama_vocab_embeddings[token] = output.last_hidden_state.mean(dim=1).cpu().numpy()

    with open(embeddings_file, "wb") as f:
        pickle.dump(llama_vocab_embeddings, f)
        
print("llama_vocab_embeddings:", type(llama_vocab_embeddings), len(llama_vocab_embeddings))



# Project the TF-IDF vectors into the LLaMA embedding space
def project_tfidf_to_llama(tfidf_vectors, vocab_embeddings, vocab):
    print("projecting tfidf vectorized data to llama embeddings...")
          
    print("tfidf_vectors:", type(tfidf_vectors), tfidf_vectors.shape)
    print("vocab_embeddings:", type(vocab_embeddings), len(vocab_embeddings))
    print("vocab:", type(vocab), vocab.shape)
    
    embedded_vectors = np.zeros((tfidf_vectors.shape[0], list(vocab_embeddings.values())[0].shape[1]))
    print("embedded_vectors:", type(embedded_vectors), embedded_vectors.shape)
    
    for i, doc in enumerate(tfidf_vectors):
        for j, token in enumerate(vocab):
            if token in vocab_embeddings:
                embedded_vectors[i] += doc[j] * vocab_embeddings[token].squeeze()
    return embedded_vectors


# Project the training and testing sets
vocab = tfidf_vectorizer.get_feature_names_out()
print("vocab:", type(vocab), vocab.shape)

X_train_projected = project_tfidf_to_llama(X_train_tfidf, llama_vocab_embeddings, vocab)
X_test_projected = project_tfidf_to_llama(X_test_tfidf, llama_vocab_embeddings, vocab)
print("X_train_projected:", type(X_train_projected), X_train_projected.shape)
print("X_test_projected:", type(X_test_projected), X_test_projected.shape)

# Train an SVM classifier on the projected features
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_projected, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_projected)

# Print classification report
print(classification_report(y_test, y_pred, target_names=target_names))

