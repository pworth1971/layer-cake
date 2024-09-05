import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from transformers import LlamaTokenizer, LlamaModel

import os
from tqdm import tqdm
import pandas as pd
import pickle

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


# ---------------------------------------------------------------------------------------------------
#
# tokens for LLAMA model access, must be requested from huggingface
# 
# must login to huggingface first 
# (see https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login) 
#
from huggingface_hub import login

HF_TOKEN = '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'
HF_TOKEN2 = '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'
# ---------------------------------------------------------------------------------------------------

import torch

# Check for CUDA availability
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device("cuda")

# Check for MPS availability (for Apple Silicon)
elif torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device("mps")

    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"          # disable memory upper limit

# Default to CPU if neither CUDA nor MPS is available
else:
    print("Neither CUDA nor MPS is available, using CPU")
    device = torch.device("cpu")

print(f"Using device: {device}")



# ---------------------------------------------------------------------------------------------------
# constants
#
DATASET_DIR = '../datasets/'
MAX_VOCAB_SIZE = 20000

llama_model_name = 'meta-llama/Llama-2-7b-hf'                    # dimension = 4096

PICKLE_DIR = '../pickles/'
TEST_SIZE = 0.2

#dataset = 'bbc_news'
dataset = '20newsgroups'
# 
# ---------------------------------------------------------------------------------------------------

stop_words = set(stopwords.words('english'))

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
    #test_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Test.csv')

    print("train_set:", train_set.shape)
    #print("test_set:", test_set.shape)    

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
#print("X_raw:", X_raw)
print("y", type(y), len(y))
print("target_names:", target_names)


print("removing stopwords...")

# Function to remove stopwords before tokenization
def remove_stopwords(texts):
    filtered_texts = []
    for text in texts:
        filtered_words = [word for word in text.split() if word.lower() not in stop_words]
        filtered_texts.append(" ".join(filtered_words))
    return filtered_texts

# Remove stopwords from the raw text
X_raw = remove_stopwords(X_raw)
print("X_raw:", type(X_raw), len(X_raw))

print("X_raw[0]:\n", X_raw[0])
print("y[0]:", y[0])

# Split the dataset into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=TEST_SIZE, random_state=42)

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


# Custom tokenizer function using LLaMA tokenizer (no need for stopwords filtering here)
def llama_tokenizer(text):
    tokens = tokenizer.tokenize(text)
    return tokens


# Create a TF-IDF vectorizer with the custom tokenizer
tfidf_vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE, tokenizer=llama_tokenizer)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test_raw).toarray()

print("X_train_tfidf:", type(X_train_tfidf), X_train_tfidf.shape)
print("X_train_tfidf[0]:", X_train_tfidf[0])
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
    for token in tqdm(tfidf_vectorizer.get_feature_names_out(), desc="encoding vocabulary using LlaMa (pretrained) embeddings..."):
        input_ids = tokenizer.encode(token, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model(input_ids)
        llama_vocab_embeddings[token] = output.last_hidden_state.mean(dim=1).cpu().numpy()

    with open(embeddings_file, "wb") as f:
        pickle.dump(llama_vocab_embeddings, f)
        
print("llama_vocab_embeddings:", type(llama_vocab_embeddings), len(llama_vocab_embeddings))

from itertools import islice

print("llama_vocab_embeddings (first three elements):\n:")
# Print the first 3 elements
for key, value in islice(llama_vocab_embeddings.items(), 3):
    print(f'{key}, {value}\n')
    
print("X_train_tfidf[0]\n:", X_train_tfidf[0])

print("\n\tApproach I: converting dataset into LlaMa embedding space (--solo)...")

# Project the TF-IDF vectors into the LLaMA embedding space
def llama_weighted_average_vectorization(tfidf_vectors, vocab_embeddings, vocab):
    print("converting tfidf vectorized data into llama embedding space...")
          
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

# convert the training and testing datasets
vocab = tfidf_vectorizer.get_feature_names_out()
print("vocab (get_feature_names_out):", type(vocab), vocab.shape)

vect_vocab = tfidf_vectorizer.vocabulary_
print("vect_vocab:", type(vect_vocab), len(vect_vocab))

# Use the tokenizer's vocabulary directly, lowercased for consistency
lower_vect_vocab = {k.lower(): v for k, v in tfidf_vectorizer.vocabulary_.items()}
print("lower_vect_vocab:", type(vect_vocab), len(vect_vocab))
        

print("encoding dataset using LlaMa embeddings (weighted average approach)...")
        
X_train_encoded_wa = llama_weighted_average_vectorization(X_train_tfidf, llama_vocab_embeddings, vocab)
X_test_encoded_wa = llama_weighted_average_vectorization(X_test_tfidf, llama_vocab_embeddings, vocab)
print("X_train_projected_wa:", type(X_train_encoded_wa), X_train_encoded_wa.shape)
print("X_test_projected_wa:", type(X_test_encoded_wa), X_test_encoded_wa.shape)

print("training SVM classifier...")

# Train an SVM classifier on the projected features
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_encoded_wa, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_encoded_wa)

# Print classification report
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


print("\n\tApproach II: projecting tfidf vectors into the LlaMa embedding space (vocabulary) using matrix multiplication (i.e. dot product)...")

# Function to convert llama_vocab_embeddings (dict) to a numpy matrix
def convert_dict_to_matrix(vocab_embeddings, vocab):
    
    print("converting dict to matrix...")
    
    # Assuming all embeddings have the same dimension and it's correctly 4096 as per the LLaMA model dimension
    embedding_dim = 4096
    embedding_matrix = np.zeros((len(vocab), embedding_dim))  # Shape (vocab_size, embedding_dim)

    print("embedding_dim:", embedding_dim)
    print("embedding_matrix:", type(embedding_matrix), embedding_matrix.shape)
    
    for i, token in enumerate(vocab):
        if token in vocab_embeddings:
            # Direct assignment of the embedding which is already in the correct shape (4096,)
            embedding_matrix[i, :] = vocab_embeddings[token]
        else:
            # Initialize missing tokens with zeros or a small random value
            embedding_matrix[i, :] = np.zeros(embedding_dim)

    return embedding_matrix

# Function to project the TF-IDF vectors into the LLaMA embedding space using matrix multiplication
def project_tfidf_to_llama(tfidf_vectors, embedding_matrix):
    return np.dot(tfidf_vectors, embedding_matrix)

print("building llama vocabulary matrix for dataset vocab...")

llama_vocab_matrix = convert_dict_to_matrix(llama_vocab_embeddings, vocab)
print("llama_vocab_matrix:", type(llama_vocab_matrix), llama_vocab_matrix.shape)
print("llama_vocab_matrix[0]:\n", llama_vocab_matrix[0])

print("-- before numpy.dot operation...")
print("X_train_tfidf:", type(X_train_tfidf), X_train_tfidf.shape)
print("X_train_tfidf[0]:\n", X_train_tfidf[0])

print("X_test_tfidf:", type(X_test_tfidf), X_test_tfidf.shape)
print("X_test_tfidf[0]:\n", X_test_tfidf[0])

# Project the training and testing sets
X_train_projected_dot = project_tfidf_to_llama(X_train_tfidf, llama_vocab_matrix)
X_test_projected_dot = project_tfidf_to_llama(X_test_tfidf, llama_vocab_matrix)

print("-- after numpy.dot product operation (input to SVM)...")
print("X_train_projected_dot:", type(X_train_projected_dot), X_train_projected_dot.shape)
print("X_train_projected_dot[0]:\n", X_train_projected_dot[0])

print("X_test_projected_dot:", type(X_test_projected_dot), X_test_projected_dot.shape)
print("X_test_projected_dot[0]:\n", X_test_projected_dot[0])

print("y_train:", type(y_train), y_train.shape)
print("y_train[0]:", y_train[0])

# Train an SVM classifier on the projected features
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_projected_dot, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_projected_dot)

# Print classification report
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

