import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from transformers import LlamaTokenizer, LlamaModel
import torch
from tqdm import tqdm


DATASET_DIR = '../../datasets/'

MAX_VOCAB_SIZE = 30000

llama_model_name = 'meta-llama/Llama-2-7b-hf'                    # dimension = 4096

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

X_raw = train_set['Text']
y = train_set['Category']
    
"""
# Load the 20 newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all')
X_raw, y = newsgroups_data.data, newsgroups_data.target
"""

print("X_raw:", type(X_raw), len(X_raw))
print("y", type(y), len(y))

# Split the dataset into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

print("X_train_raw:", type(X_train_raw), len(X_train_raw))
print("X_test_raw:", type(X_test_raw), len(X_test_raw))
print("y_train:", type(y_train), len(y_train))
print("y_test:", type(y_test), len(y_test))


# Convert raw documents to TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test_raw).toarray()

print("X_train_tfidf:", type(X_train_tfidf), X_train_tfidf.shape)
print("X_test_tfidf:", type(X_test_tfidf), X_test_tfidf.shape)

# Load the pre-trained LLaMA model and tokenizer
#llama_model_name = 'huggingface/llama-7b'  # Example model, adjust according to availability
tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)

# Ensure padding token is available
if tokenizer.pad_token is None:
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Reuse the end-of-sequence token for padding
    
model = LlamaModel.from_pretrained(llama_model_name).to(device)
print("model: ", model)

def encode_with_llama(texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with LLaMA"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))
    embeddings = torch.cat(embeddings).cpu().numpy()
    return embeddings

# Encode the training texts with LLaMA to get the LLaMA embedding space
X_train_llama_embeddings = encode_with_llama(X_train_raw)
print("X_train_llama_embeddings:", type(X_train_llama_embeddings), X_train_llama_embeddings.shape)

# Transpose the LLaMA embeddings to create a projection matrix
llama_projection_matrix = X_train_llama_embeddings.T
print("llama_projection_matrix:", type(llama_projection_matrix), llama_projection_matrix.shape)

# Project TF-IDF vectors into the LLaMA embedding space using a dot product
def project_tfidf_to_llama(tfidf_vectors, projection_matrix):
    return np.dot(tfidf_vectors, projection_matrix)

# Project the training and testing sets
X_train_projected = project_tfidf_to_llama(X_train_tfidf, llama_projection_matrix)
X_test_projected = project_tfidf_to_llama(X_test_tfidf, llama_projection_matrix)

print("X_train_projected:", type(X_train_projected), X_train_projected.shape)
print("X_test_projected:", type(X_test_projected), X_test_projected.shape)

# Train an SVM classifier on the projected features
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_projected, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_projected)

# Print classification report
print(classification_report(y_test, y_pred, target_names=newsgroups_data.target_names))