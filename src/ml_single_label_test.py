import pandas as pd
import string
import numpy as np
import os
import pickle
import argparse

from scipy.sparse import issparse, csr_matrix

from matplotlib import pyplot as plt
import plotly.offline as pyo
import plotly.graph_objs as go

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report
from sklearn.metrics import confusion_matrix, make_scorer, recall_score, hamming_loss, jaccard_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, LlamaTokenizer

import warnings
warnings.filterwarnings('ignore')

#
# we assume everything runs from bin directory
#
PICKLE_DIR = '../pickles/'
OUT_DIR = '../out/'
DATASET_DIR = '../datasets/'
VECTOR_CACHE = '../.vector_cache'
MAX_VOCAB_SIZE = 25000

NUM_JOBS = -1          # important to manage CUDA memory allocation
#NUM_JOBS = 40          # for rcv1 dataset which has 101 classes, too many to support in parallel


#dataset_available = {'reuters21578', '20newsgroups', 'ohsumed', 'rcv1'}
dataset_available = {'20newsgroups', 'bbc-news'}


# Ensure stopwords are downloaded
nltk.download('stopwords')

# ------------------------------------------------------------------------------------------------------------------------
# Utility functions for preprocessing data
# ------------------------------------------------------------------------------------------------------------------------
def missing_values(df):
    """
    Calculate the percentage of missing values for each column in a DataFrame.
    
    Args:
    df (pd.DataFrame): The input DataFrame to analyze.
    
    Returns:
    pd.DataFrame: A DataFrame containing the total count and percentage of missing values for each column.
    """
    # Calculate total missing values and their percentage
    total = df.isnull().sum()
    percent = (total / len(df) * 100)
    
    # Create a DataFrame with the results
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    # Sort the DataFrame by percentage of missing values (descending)
    missing_data = missing_data.sort_values('Percent', ascending=False)
    
    # Filter out columns with no missing values
    missing_data = missing_data[missing_data['Total'] > 0]
    
    print("Columns with missing values:")
    print(missing_data)
    
    return missing_data


def remove_punctuation(x):
    punctuationfree="".join([i for i in x if i not in string.punctuation])
    return punctuationfree


# Function to lemmatize text with memory optimization
def lemmatization(texts, chunk_size=1000):
    lmtzr = WordNetLemmatizer()
    
    num_chunks = len(texts) // chunk_size + 1
    #print(f"Number of chunks: {num_chunks}")
    for i in range(num_chunks):
        chunk = texts[i*chunk_size:(i+1)*chunk_size]
        texts[i*chunk_size:(i+1)*chunk_size] = [' '.join([lmtzr.lemmatize(word) for word in text.split()]) for text in chunk]
    
    return texts


# ------------------------------------------------------------------------------------------------------------------------

def preprocessDataset(train_text):
    
    #print("preprocessing...")
    
    # Ensure input is string
    train_text = str(train_text)
    
    # Word tokenization using NLTK's word_tokenize
    tokenized_train_set = word_tokenize(train_text.lower())
    
    # Stop word removal
    stop_words = set(stopwords.words('english'))
    stopwordremove = [i for i in tokenized_train_set if i not in stop_words]
    
    # Join words into sentence
    stopwordremove_text = ' '.join(stopwordremove)
    
    # Remove numbers
    numberremove_text = ''.join(c for c in stopwordremove_text if not c.isdigit())
    
    # Stemming using NLTK's PorterStemmer
    stemmer = PorterStemmer()
    stem_input = word_tokenize(numberremove_text)
    stem_text = ' '.join([stemmer.stem(word) for word in stem_input])
    
    # Lemmatization using NLTK's WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    lem_input = word_tokenize(stem_text)
    lem_text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in lem_input])
    
    return lem_text

# ------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------
# load_20newsgroups()
#
# Define the load_20newsgroups function
# ------------------------------------------------------------------------------------------------------------------------
def load_20newsgroups(vectorizer_type='tfidf', embedding_type='word', pretrained=None):
    """
    Load and preprocess the 20 Newsgroups dataset, returning X and y sparse matrices.

    Parameters:
    - vectorizer_type: 'tfidf' or 'count', determines which vectorizer to use for tokenization.
    - embedding_type: 'word' for word-based embeddings (GloVe, Word2Vec, fastText) or 'token' for token-based models (LLaMA, BERT).

    Returns:
    - X: Sparse matrix of features (tokenized text aligned with the chosen vectorizer and embedding type).
    - y: Sparse array of target labels.
    """

    print("Loading 20 newsgroups dataset...")

    # Fetch the 20 newsgroups dataset
    X_raw, y = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), return_X_y=True)

    # Initialize the vectorizer based on the type
    if embedding_type == 'word':
        print("Using word-level vectorization...")
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE, stop_words='english')  # Adjust max_features as needed
        elif vectorizer_type == 'count':
            vectorizer = CountVectorizer(max_features=MAX_VOCAB_SIZE, stop_words='english')  # Adjust max_features as needed
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")
        
        # Fit and transform the text data to obtain tokenized features
        X_vectorized = vectorizer.fit_transform(X_raw)

    elif embedding_type == 'token':
        print("Using token-level vectorization...")
        if pretrained == 'bert':
            print("Using token-level vectorization with BERT embeddings...")
            tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')             # Replace with correct LLaMA model
        elif pretrained == 'llama': 
            print("Using token-level vectorization with BERT or LLaMa embeddings...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')                          # BERT tokenizer
        else:
            raise ValueError("Invalid embedding type. Use pretrained = 'bert' or pretrained = 'llama' for token embeddings.")

        def tokenize(text):
            tokens = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            return tokens['input_ids']
        
        X_vectorized = [tokenize(text) for text in X_raw]
        X_vectorized = csr_matrix(np.vstack(X_vectorized))  # Convert list of arrays to a sparse matrix
    
    else:
        raise ValueError("Invalid embedding type. Use 'word' for word embeddings or 'token' for BERT/LLaMA embeddings.")
    
    print("Text vectorization completed.")
    
    # Ensure X_vectorized is a sparse matrix
    if not isinstance(X_vectorized, csr_matrix):
        X_vectorized = csr_matrix(X_vectorized)
    
    # Encode the labels
    label_encoder = LabelEncoder()
    y_sparse = csr_matrix(label_encoder.fit_transform(y)).T  # Transpose to match the expected shape

    print("Labels encoded and converted to sparse format.")

    # Return X (features) and y (target labels) as sparse arrays
    return X_vectorized, y_sparse



# ------------------------------------------------------------------------------------------------------------------------
# load_bbc_news()
# ------------------------------------------------------------------------------------------------------------------------
def load_bbc_news(vectorizer_type='tfidf', embedding_type='word', pretrained=None):
    """
    Load and preprocess the BBC News dataset and return X, Y sparse arrays.
    
    Parameters:
    - vectorizer_type: 'tfidf' or 'count', determines which vectorizer to use for tokenization.
    - embedding_type: 'word' for word-based embeddings (GloVe, Word2Vec, fastText) or 'token' for token-based models (BERT, LLaMa).
    
    Returns:
    - X: Sparse array of features (tokenized text aligned with the chosen vectorizer and embedding type).
    - Y: Sparse array of target labels.
    """
    
    print("Loading BBC News dataset...")

    for dirname, _, filenames in os.walk(DATASET_DIR + 'bbc-news'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    # Load datasets
    train_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Train.csv')
    test_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Test.csv')

    print("train_set:", train_set.shape)
    print("test_set:", test_set.shape)

    # Combine train and test sets
    #df = pd.concat([train_set, test_set], ignore_index=True)
    df = train_set

    target_category = df['Category'].unique()
    print("target categories:", target_category)

    df['categoryId'] = df['Category'].factorize()[0]
    
    #category = df[["Category", "categoryId"]].drop_duplicates().sort_values('categoryId')
    #print("after de-duping:", category)

    #print(df.groupby('Category').categoryId.count())

    print("preprocessing...")

    # Preprocess the text
    #df['Text'] = df['Text'].apply(preprocessDataset)
    
    # Choose the vectorization and tokenization strategy based on embedding type
    if embedding_type == 'word':
        print("Using word-level vectorization...")
        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE, stop_words=stopwords.words("english"))  # Adjust max_features as needed
        elif vectorizer_type == 'count':
            vectorizer = CountVectorizer(max_features=MAX_VOCAB_SIZE, stop_words=stopwords.words("english"))  # Adjust max_features as needed
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")
        
        # Fit and transform the text data to obtain tokenized features
        X_vectorized = vectorizer.fit_transform(df['Text'])
    
    elif embedding_type == 'token':

        if pretrained == 'bert':
            print("Using token-level vectorization with BERT embeddings...")
            tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')             # Replace with correct LLaMA model
        elif pretrained == 'llama': 
            print("Using token-level vectorization with BERT or LLaMa embeddings...")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')                          # BERT tokenizer
        else:
            raise ValueError("Invalid embedding type. Use pretrained = 'bert' or pretrained = 'llama' for token embeddings.")

        def tokenize(text):
            tokens = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            return tokens['input_ids']
        
        X_vectorized = df['Text'].apply(tokenize).values
        X_vectorized = csr_matrix(np.vstack(X_vectorized))  # Convert list of arrays to a sparse matrix
    
    else:
        raise ValueError("Invalid embedding type. Use 'word' for word embeddings or 'token' for BERT/LLaMa embeddings.")
    
    # Ensure X_vectorized is a sparse matrix (in case of word-based embeddings)
    if not isinstance(X_vectorized, csr_matrix):
        X_vectorized = csr_matrix(X_vectorized)
    
    # Encode the labels
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(df['Category'])
    
    # Convert Y to a sparse matrix
    Y_sparse = csr_matrix(Y).T  # Transpose to match the expected shape

    # Return X (features) and Y (target labels) as sparse arrays
    return X_vectorized, Y_sparse

# ------------------------------------------------------------------------------------------------------------------------

def load_data(dataset='20newsgroups', pretrained=None):

    print(f"Loading data set {dataset}, pretrained is {pretrained}...")

    if (pretrained == 'llama' or pretrained == 'bert'):
        embedding_type = 'token'
    else:
        embedding_type = 'word'

    if (dataset == '20newsgroups'): 
        X, y = load_20newsgroups(embedding_type=embedding_type)
        return X, y
    elif (dataset == 'bbc-news'):
        X, y = load_bbc_news(embedding_type=embedding_type)
        return X, y
    else:
        print(f"Dataset '{dataset}' not available.")
        return None


# ------------------------------------------------------------------------------------------------------------------------
# Save X and y sparse matrices to pickle
# ------------------------------------------------------------------------------------------------------------------------
def save_to_pickle(X, y, pickle_file):
    print(f"Saving X and y to pickle file: {pickle_file}")
    with open(pickle_file, 'wb') as f:
        # Save the sparse matrices as a tuple
        pickle.dump((X, y), f)

# ------------------------------------------------------------------------------------------------------------------------
# Load X and y sparse matrices from pickle
# ------------------------------------------------------------------------------------------------------------------------
def load_from_pickle(pickle_file):
    print(f"Loading X and y from pickle file: {pickle_file}")
    with open(pickle_file, 'rb') as f:
        X, y = pickle.load(f)
    return X, y



# --------------------------------------------------------------------------------------------------------------
# Core processing function
# --------------------------------------------------------------------------------------------------------------
def classify(dataset='20newsgrouops', args=None):
    
    if (args is None):
        print("No arguments passed.")
        return
    
    if (args.dataset not in dataset_available):
        print(f"Dataset '{args.dataset}' not available.")
        return
    
    pickle_file_name=f'{dataset}_{args.pretrained}_tokenized.pickle'

    print(f"Classifying {dataset}...")

    #print(f"Loading data set {dataset}...")

    pickle_file = PICKLE_DIR + pickle_file_name                         # Define the path to the pickle file

    if os.path.exists(pickle_file):  # if the pickle file exists
        print(f"Loading tokenized data from '{pickle_file}'...")
        X, y = load_from_pickle(pickle_file)
    else:
        print(f"'{pickle_file}' not found, retrieving and preprocessing dataset {dataset}...")
        X, y = load_data(dataset=dataset, pretrained=args.pretrained)

        save_to_pickle(X, y, pickle_file)                               # Save the tokenized matrices to a pickle file

    print("Tokenized data loaded.")
 
    print("X:", type(X), X.shape)
    print("y:", type(y), y.shape)

    print("train_test_sp;it...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size = 0.25, 
        random_state = 44,
        shuffle=True 
        )

    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
    
    print("y_train:", type(y_train), y_train.shape)
    print("y_test:", type(y_test), y_test.shape)

    run_model(X_train, X_test, y_train, y_test, args)

# -------------------------------------------------------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------------------------------------------------------
# run_model()
# -------------------------------------------------------------------------------------------------------------------------------------------------
def run_model(X_train, X_test, y_train, y_test, args):
    
    print("Running model...")

    print('X_train:', type(X_train), X_train.shape)
    print('X_test:', type(X_test), X_test.shape)
             
    # Generate embeddings
    X_train, X_test = gen_embeddings(
        X_train=X_train, 
        X_test=X_test, 
        dataset=args.dataset, 
        embeddings=args.pretrained, 
        mode=args.mode, 
        ngram_size=args.ngram_size
        )
        
    # Support Vector Machine Classifier
    if (args.learner == 'svm'):
        run_svm_model(X_train, X_test, y_train, y_test, args)

    # Logistic Regression Classifier
    elif (args.learner == 'lr'):
        run_lr_model(X_train, X_test, y_train, y_test, args)

    # Naive Bayes (MultinomialNB) Classifier
    elif (args.learner == 'nb'):
        run_nb_model(X_train, X_test, y_train, y_test, args)

    elif (args.learner == 'dt'):
        print("Decision Tree Classifier")
        dt = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('dt', DecisionTreeClassifier())
            ])

        dt.fit(X_train, y_train)

        test_predict = dt.predict(X_test)

        train_accuracy = round(dt.score(X_train, y_train)*100)
        test_accuracy =round(accuracy_score(test_predict, y_test)*100)

        print("Decision Tree Train Accuracy Score : {}% ".format(train_accuracy ))
        print("Decision Tree Test Accuracy Score  : {}% ".format(test_accuracy ))
        print(classification_report(y_true=test_predict, y_pred=y_test, digits=4))

    elif (args.learner == 'rf'):

        print("Random Forest Classifier")
        rfc = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('rfc', RandomForestClassifier(n_estimators=100))
            ])

        rfc.fit(X_train, y_train)

        test_predict = rfc.predict(X_test)

        train_accuracy = round(rfc.score(X_train, y_train)*100)
        test_accuracy =round(accuracy_score(test_predict, y_test)*100)

        print("K-Nearest Neighbour Train Accuracy Score : {}% ".format(train_accuracy ))
        print("K-Nearest Neighbour Test Accuracy Score  : {}% ".format(test_accuracy ))
        print(classification_report(y_true=test_predict, y_pred=y_test, digits=4))

    else:
        print(f"Invalid learner '{args.learner}'")
        return


# ---------------------------------------------------------------------------------------------------------------------
# run_svm_model()
# ---------------------------------------------------------------------------------------------------------------------
def run_svm_model(X_train, X_test, y_train, y_test, args):

    if (not args.optimc):
        
        print("Training default Support Vector Machine model...")
        
        """
        default_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('lr', LinearSVC(max_iter=1000))
        ])

        default_pipeline.fit(X_train, y_train)
        y_pred_default = default_pipeline.predict(X_test)
        """

        svc = LinearSVC(max_iter=1000)
        svc.fit(X_train, y_train)
        y_pred_default = svc.predict(X_test)
        
        print("\nDefault Support Vector Mechine Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_default, digits=4))

    elif (args.optimc):                             # Optimize Support Vector Machine with GridSearchCV

        print("Optimizing Support Vector Machine model with GridSearchCV...")

        """
        # Define the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svm', LinearSVC(max_iter=1000))
        ])

        # Define the parameter grid
        param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],     # Unigrams, bigrams, or trigrams
            'tfidf__use_idf': [True, False],                    # Whether to use IDF
            'tfidf__sublinear_tf': [True, False],               # Sublinear term frequency
            'svm__penalty': ['l1', 'l2'],                       # Regularization method
            'svm__loss': ['hinge', 'squared_hinge'],            # Loss function
            'svm__multi_class': ['ovr', 'crammer_singer'],      # Multi-class strategy
            'svm__class_weight': [None, 'balanced'],            # Class weights
            'svm__C': np.logspace(-3, 3, 7)                     # Regularization parameter   
        }
        """
        param_grid = {
            'penalty': ['l1', 'l2'],                       # Regularization method
            'loss': ['hinge', 'squared_hinge'],            # Loss function
            'multi_class': ['ovr', 'crammer_singer'],      # Multi-class strategy
            'class_weight': [None, 'balanced'],            # Class weights
            'C': np.logspace(-3, 3, 7)                     # Regularization parameter   
        }
        
        print("param_grid:", param_grid)

        cross_validation = StratifiedKFold()

        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro'),
            'precision_score': make_scorer(precision_score, average='micro'),
            'hamming_loss': make_scorer(hamming_loss),
            'jaccard_score': make_scorer(jaccard_score, average='micro')
            }

        grid_search = GridSearchCV(
            n_jobs=-1, 
            #estimator=pipeline,
            estimator=LinearSVC(max_iter=1000),
            refit='f1_score',
            param_grid=param_grid,
            cv=cross_validation,
            #scoring=scoring
            scoring=scorers,
            return_train_score=True         # ensure train scores are calculated
            )

        # Fit the model
        grid_search.fit(X_train, y_train.toarray())

        print('Best parameters: {}'.format(grid_search.best_params_))
        print("best_estimator:", grid_search.best_estimator_)
        print('Best score: {}'.format(grid_search.best_score_))
        #print("cv_results_:", grid_search.cv_results_)

        results = grid_search.cv_results_

        if (args.plot):

            print("Plotting the results...")

            # Define the metrics we want to plot
            metrics_to_plot = ['accuracy_score', 'f1_score', 'recall_score', 'precision_score', 'hamming_loss']

            # Iterate over each metric to create a separate plot
            for metric in metrics_to_plot:
                traces = []

                print(f"Plotting {metric}...")

                for sample in ["train", "test"]:

                    key_mean = f"mean_{sample}_{metric}"
                    key_std = f"std_{sample}_{metric}"

                    print(f"Plotting {key_mean}...")
                    print(f"Plotting {key_std}...")

                    # Directly use the keys without conditional check
                    sample_score_mean = np.nan_to_num(np.array(results[key_mean]) * 100)  # Convert to percentage and handle NaN
                    sample_score_std = np.nan_to_num(np.array(results[key_std]) * 100)  # Convert to percentage and handle NaN

                    x_axis = np.linspace(0, 100, len(sample_score_mean))

                    # Create the trace for Plotly
                    traces.append(
                        go.Scatter(
                            x=x_axis,
                            y=sample_score_mean,
                            mode='lines+markers',
                            name=f"{metric} ({sample})",
                            line=dict(dash='dash' if sample == 'train' else 'solid'),
                            error_y=dict(
                                type='data',
                                array=sample_score_std,
                                visible=True
                            ),
                            hoverinfo='x+y+name'
                        )
                    )

                # Define the layout of the plot
                layout = go.Layout(
                    title={'text': f"Training and Test Scores for {metric.capitalize()}",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    xaxis=dict(title="Training Sample Percentage (%)"),
                    yaxis=dict(title="Score (%)", range=[0, 100]),
                    hovermode='closest'
                )

                # Create the figure
                fig = go.Figure(data=traces, layout=layout)

                # Write the plot to an HTML file
                filename = f'{OUT_DIR}training_test_scores_{metric}.html'
                pyo.plot(fig, filename=filename)

                print(f"Saved plot for {metric} as {filename}")

        # Extract the best estimator from the GridSearchCV
        best_model = grid_search.best_estimator_

        # Predict on the test set using the best model
        y_pred_best = best_model.predict(X_test)

        print("Accuracy best score:", metrics.accuracy_score(y_test, y_pred_best))
        print(classification_report(y_true=y_test, y_pred=y_pred_best, digits=4))


# ---------------------------------------------------------------------------------------------------------------------
# run_lr_model()
# ---------------------------------------------------------------------------------------------------------------------
def run_lr_model(X_train, X_test, y_train, y_test, args):

    # Default Logistic Regression Model
    print("Training default Logistic Regression model...")
    
    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
    
    if (not args.optimc):
        
        print("Optimization not requested, training default Logistic Regression model...")
        """
        default_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('lr', LogisticRegression(max_iter=1000))
        ])
        
        default_pipeline.fit(X_train, y_train)
        y_pred_default = default_pipeline.predict(X_test)
        """

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_default = lr.predict(X_test)
        
        print("\nDefault Logistic Regression Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_default, digits=4))

    elif (args.optimc):
        
        # Optimize Logistic Regression with GridSearchCV
        print("Optimizing Logistic Regression model with GridSearchCV...")

        # Define the pipeline
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('lr', LogisticRegression(max_iter=1000))
        ])

        # Define the parameter grid
        param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],     # Unigrams, bigrams, or trigrams
            'tfidf__use_idf': [True, False],                    # Whether to use IDF
            'tfidf__sublinear_tf': [True, False],               # Sublinear term frequency
            'lr__C': [0.01, 0.1, 1, 10, 100],                   # Inverse of regularization strength
            'lr__penalty': ['l2'],                              # Regularization method (L2 Ridge)
            'lr__solver': ['liblinear', 'lbfgs']                # Solver types
        }
        """

        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],                                           # Inverse of regularization strength
            'penalty': ['l1', 'l2'],                                                # Regularization method (L2 Ridge)
            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']            # Solver types
        }
        
        print("param_grid:", param_grid)

        # Define scorers
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro'),
            'precision_score': make_scorer(precision_score, average='micro')
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            n_jobs=-1,
            #estimator=pipeline,
            estimator =  LogisticRegression(max_iter=1000),
            param_grid=param_grid,
            scoring=scorers,
            refit='f1_score',  # Optimize on F1 Score
            cv=StratifiedKFold(n_splits=5),
            return_train_score=True
        )

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Display the best parameters
        print('Best parameters found by GridSearchCV:')
        print(grid_search.best_params_)

        # Evaluate on the test set
        y_pred_optimized = grid_search.best_estimator_.predict(X_test)

        print("\nOptimized Logistic Regression Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_optimized):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_optimized, digits=4))

    if (args.cm):
        # Optionally, plot confusion matrix for the optimized model
        create_confusion_matrix(
            y_test, 
            y_pred_optimized, 
            title='Confusion Matrix for Optimized Logistic Regression Model',
            file_name=OUT_DIR+'bbc_news_logistic_regression_confusion_matrix.png',
            debug=False
        )



# ---------------------------------------------------------------------------------------------------------------------
# run_nb_model()
# ---------------------------------------------------------------------------------------------------------------------
def run_nb_model(X_train, X_test, y_train, y_test, args):

    print("Building default Naive Bayes Classifier...")

    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
    
    if (not args.optimc):
            
        print("Optimization not requested, training default Naive Bayes model...")
        """
        nb = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
            ])
        """
        
        nb = MultinomialNB()
        nb.fit(X_train,y_train)
        test_predict = nb.predict(X_test)

        train_accuracy = round(nb.score(X_train,y_train)*100)
        test_accuracy =round(accuracy_score(test_predict, y_test)*100)

        print("Naive Bayes Train Accuracy Score : {}% ".format(train_accuracy ))
        print("Naive Bayes Test Accuracy Score  : {}% ".format(test_accuracy ))
        print(classification_report(y_true=test_predict, y_pred=y_test, digits=4))

    elif (args.optimc):

        print("Optimizing the model using GridSearchCV...")

        """
        # Define a pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('nb', MultinomialNB())
        ])

        # Define the parameter grid
        param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],         # Unigrams, bigrams, or trigrams
            'tfidf__use_idf': [True, False],                        # Whether to use IDF
            'tfidf__sublinear_tf': [True, False],                   # Sublinear term frequency
            'nb__alpha': [0.1, 0.5, 1.0, 1.5, 2.0],                 # Smoothing parameter for Naive Bayes
        }
        """
        
        # Define the parameter grid
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],                 # Smoothing parameter for Naive Bayes
        }
        
        print("param_grid:", param_grid)

        # Define scorers
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro'),
            'precision_score': make_scorer(precision_score, average='micro'),
            'hamming_loss': make_scorer(hamming_loss),
            'jaccard_score': make_scorer(jaccard_score, average='micro')
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            n_jobs=-1,
            #estimator=pipeline
            estimator=MultinomialNB(),
            param_grid=param_grid,
            scoring=scorers,
            refit='f1_score',                           # Optimize on F1 Score
            cv=StratifiedKFold(n_splits=5),
            return_train_score=True
        )

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Display the best parameters
        print('Best parameters found by GridSearchCV:')
        print(grid_search.best_params_)

        # Evaluate on the test set
        y_pred = grid_search.best_estimator_.predict(X_test)

        print("\nBest Estimator's Test Set Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred, average='micro'):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred, average='micro'):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='micro'):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred, digits=4))

        if (args.cm):
            # Optionally, plot confusion matrix
            create_confusion_matrix(
                y_test, 
                y_pred, 
                title='Confusion Matrix for Optimized Naive Bayes Model',
                file_name=OUT_DIR+'bbc_news_naive_bayes_confusion_matrix.png',
                debug=False
            )


# ------------------------------------------------------------------------------------------------------------------------
# BERT Embeddings functions
# ------------------------------------------------------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten()
        }

def get_bert_embeddings(texts, model, tokenizer, device, batch_size=32, max_len=256):
    dataset = TextDataset(texts, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    embeddings = []
    model.eval()

    with torch.cuda.amp.autocast(), torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings

# -------------------------------------------------------------------------------------------------------------------
def cache_embeddings(train_embeddings, test_embeddings, cache_path):
    print("caching embeddings to:", cache_path)
    
    np.savez(cache_path, train=train_embeddings, test=test_embeddings)

def load_cached_embeddings(cache_path, debug=False):
    if (debug):
        print("looking for cached embeddings at ", cache_path)
    
    if os.path.exists(cache_path):
        
        if (debug):
            print("found cached embeddings, loading...")
        data = np.load(cache_path)
        return data['train'], data['test']
    
    if (debug):
        print("did not find cached embeddings, returning None...")
    return None, None
# -------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------
def todense(y):
    """Convert sparse matrix to dense format as needed."""
    return y.toarray() if issparse(y) else y


def tosparse(y):
    """Ensure matrix is in CSR format for efficient arithmetic operations."""
    return y if issparse(y) else csr_matrix(y)
# -------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------
# gen_embeddings()
#
# Embedding generation function with BERT tokenization
# -------------------------------------------------------------------------------------------------------------------
def gen_embeddings(X_train, X_test, dataset='20newsgroups', embeddings=None, mode='solo', ngram_size=1):
    print("generating embeddings...")
    print("dataset:", dataset)
    print("embeddings:", embeddings)
    print('X_train:', type(X_train), X_train.shape)
    print('X_test:', type(X_test), X_test.shape)
    
    if embeddings == 'bert':
        print("Using BERT pretrained embeddings...")

        BERT_MODEL = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        cache_path = f'../.vector_cache/{BERT_MODEL}_{dataset}.npz'
        train_embeddings, test_embeddings = load_cached_embeddings(cache_path, debug=True)

        # Tokenize text using BERT tokenizer and vectorize accordingly
        def tokenize_and_vectorize(texts, tokenizer):
            tokenized_texts = texts.apply(lambda x: tokenizer.tokenize(x))
            token_indices = tokenized_texts.apply(lambda tokens: tokenizer.convert_tokens_to_ids(tokens))
            
            # Initialize a zero matrix to store token counts (csr_matrix for efficiency)
            num_texts = len(texts)
            vocab_size = tokenizer.vocab_size
            token_matrix = csr_matrix((num_texts, vocab_size), dtype=np.float32)
            
            for i, tokens in enumerate(token_indices):
                for token in tokens:
                    token_matrix[i, token] += 1  # Increment the count for each token
            
            return token_matrix

        X_train_token_matrix = tokenize_and_vectorize(X_train, tokenizer)
        X_test_token_matrix = tokenize_and_vectorize(X_test, tokenizer)

        if train_embeddings is None or test_embeddings is None: 
            print("generating BERT embeddings")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            bert_model = BertModel.from_pretrained(BERT_MODEL).to(device)
            X_train_bert = get_bert_embeddings(X_train.tolist(), bert_model, tokenizer, device)
            X_test_bert = get_bert_embeddings(X_test.tolist(), bert_model, tokenizer, device)
            cache_embeddings(X_train_bert, X_test_bert, cache_path)
        else:
            X_train_bert = train_embeddings
            X_test_bert = test_embeddings

        print('X_train_bert:', type(X_train_bert), X_train_bert.shape)
        print('X_test_bert:', type(X_test_bert), X_test_bert.shape)

        if mode == 'solo':
            print("using just the BERT embeddings alone (solo)...")
            X_train = X_train_bert
            X_test = X_test_bert
        elif mode == 'cat':
            print("concatenating BERT embeddings with token matrix...")
            X_train_combined = np.hstack([X_train_token_matrix.toarray(), X_train_bert])
            X_test_combined = np.hstack([X_test_token_matrix.toarray(), X_test_bert])
            X_train = X_train_combined
            X_test = X_test_combined
        elif mode == 'dot':
            print("dot product (matrix multiplication) of BERT embeddings with token matrix...")
            X_train = X_train_token_matrix.dot(X_train_bert)
            X_test = X_test_token_matrix.dot(X_test_bert)

    return X_train, X_test



    
# --------------------------------------------------------------------------------------------------------------------------------------------

def create_confusion_matrix(y_test, y_pred, title, file_name=OUT_DIR+'svm_20newsgroups_confusion_matrix_best_model_table.png', debug=False):

    print("Creating confusion matrix...")

    # Assuming y_test and y_pred_best are already defined
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plotting the confusion matrix as a table with numbers
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the width and height of the figure

    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create the table with smaller font sizes and adjusted scale
    table = ax.table(
        cellText=conf_matrix,
        rowLabels=[f'Actual {i}' for i in range(conf_matrix.shape[0])],
        colLabels=[f'Predicted {i}' for i in range(conf_matrix.shape[1])],
        cellLoc='center',
        loc='center'
    )

    # Adjust the font size and layout
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Reduced font size for better fitting
    table.scale(1.2, 1.2)

    # Add a title with centered text
    plt.title(title, fontsize=16, pad=20)

    # Adjust layout to add more padding around the plot
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)  # Increase padding on the left

    # Save the plot to a file
    confusion_matrix_filename = file_name
    plt.savefig(confusion_matrix_filename, bbox_inches='tight')  # Ensure everything is saved in the output file
    plt.show()

    print(f"Confusion matrix saved as {confusion_matrix_filename}")

    accuracy = accuracy_score(y_test, y_pred)

    # Plain text explanation of the confusion matrix
    if debug:
        print("\nHow to read this confusion matrix:")
        print("------------------------------------------------------")
        print("The confusion matrix shows the performance of the classification model.")
        print("Each row of the matrix represents the actual classes, while each column represents the predicted classes.")
        print("Values on the diagonal (from top-left to bottom-right) represent correct predictions (true positives and true negatives).")
        print("Values outside the diagonal represent incorrect predictions (false positives and false negatives).")
        print("\nAccuracy Score: {:.2f}%".format(accuracy * 100))
        
        print("\nConfusion Matrix Values:")
        for i in range(len(conf_matrix)):
            print(f"Actual class {i}:")
            for j in range(len(conf_matrix[i])):
                print(f"  Predicted as class {j}: {conf_matrix[i][j]}")

# --------------------------------------------------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification Testing')
    
    parser.add_argument('--dataset', required=True, type=str, default='20newsgroups', metavar='N', help=f'dataset, one in {dataset_available}')
    
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str', help=f'path where to load the pickled dataset from')
    
    parser.add_argument('--log-file', type=str, default='../log/svm.test', metavar='N', help='path to the application log file')
    
    parser.add_argument('--learner', type=str, default='svm', metavar='N', help=f'learner (svm, lr, or nb)')
    
    parser.add_argument('--mode', type=str, default='solo', metavar='N', help=f'way to combine tfidf and pretrained embeddings, in [solo, cat, dot]')

    parser.add_argument('--cm', action='store_true', default=False, help=f'create confusion matrix')

    parser.add_argument('--plot', action='store_true', default=False, help=f'create plots of GridSearchCV metrics (if --optimc is True)')
                             
    parser.add_argument('--optimc', action='store_true', default=False, help='optimize the model using relevant models params')
    
    parser.add_argument('--ngram-size', type=int, default=2, metavar='int',
                        help='ngram parameter into vectorization routines (TFIDFVectorizer)')
    
    parser.add_argument('--pretrained', type=str, default=None, metavar='glove|word2vec|fasttext|bert|llama',
                        help='pretrained embeddings, use "glove", "word2vec", "fasttext", "bert", or "llama" (default None)')
    
    parser.add_argument('--embedding-dir', type=str, default='../.vector_cache', metavar='str',
                        help=f'path where to load and save BERT document embeddings')
    
    parser.add_argument('--word2vec-path', type=str, default=VECTOR_CACHE+'/GoogleNews-vectors-negative300.bin',
                        metavar='PATH',
                        help=f'path + filename to Word2Vec pretrained vectors (e.g. ../.vector_cache/GoogleNews-vectors-negative300.bin), used only '
                             f'with --pretrained word2vec')
    
    parser.add_argument('--glove-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to pretrained glove embeddings (glove.840B.300d.txt.pt file), used only with --pretrained glove')
    
    parser.add_argument('--fasttext-path', type=str, default=VECTOR_CACHE+'/crawl-300d-2M.vec',
                        metavar='PATH',
                        help=f'path + filename to fastText pretrained vectors (e.g. --fasttext-path ../.vector_cache/crawl-300d-2M.vec), used only '
                            f'with --pretrained fasttext')
    
    parser.add_argument('--bert-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to BERT pretrained vectors, used only with --pretrained bert')
    
    parser.add_argument('--llama-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to LLaMA pretrained vectors, used only with --pretrained llama')
    

    args = parser.parse_args()

    print("args:", type(args), args)

    classify(dataset=args.dataset, args=args)
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------