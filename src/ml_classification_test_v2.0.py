
import numpy as np
import os
import argparse
from time import time

import plotly.offline as pyo
import plotly.graph_objs as go

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report
from sklearn.metrics import make_scorer, recall_score, hamming_loss, jaccard_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

from util.common import initialize, SystemResources
from data.lc_dataset import LCDataset, load_data, save_to_pickle, load_from_pickle
from util.metrics import evaluation

import warnings
from ipykernel.pickleutil import class_type
warnings.filterwarnings('ignore')

#
# we assume everything runs from bin directory
#
PICKLE_DIR = '../pickles/'
VECTOR_CACHE = '../.vector_cache'


NUM_JOBS = -1          # important to manage CUDA memory allocation
#NUM_JOBS = 40          # for rcv1 dataset which has 101 classes, too many to support in parallel





# -------------------------------------------------------------------------------------------------------------------------------------------------
# run_model()
# -------------------------------------------------------------------------------------------------------------------------------------------------
def run_model(X_train, X_test, y_train, y_test, args):
    
    print("\tRunning model...")

    print('X_train:', type(X_train), X_train.shape)
    print('X_test:', type(X_test), X_test.shape)

    print('y_train:', type(y_train), y_train.shape)
    print('y_test:', type(y_test), y_test.shape)
    
    #print("y_train:", y_train)
    #print("y_test:", y_test)
        
    tinit = time()

    # Support Vector Machine Classifier
    if (args.learner == 'svm'):                                     
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = run_svm_model(X_train, X_test, y_train, y_test, args)
    
    # Logistic Regression Classifier
    elif (args.learner == 'lr'):                                  
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = run_lr_model(X_train, X_test, y_train, y_test, args)

    # Naive Bayes (MultinomialNB) Classifier
    elif (args.learner == 'nb'):                                  
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = run_nb_model(X_train, X_test, y_train, y_test, args)
    
    else:
        print(f"Invalid learner '{args.learner}'")
        return None

    formatted_string = f'Macro F1: {Mf1:.4f} Micro F1: {mf1:.4f} Acc: {accuracy:.4f} Hamming Loss: {h_loss:.4f} Precision: {precision:.4f} Recall: {recall:.4f} Jaccard Index: {j_index:.4f}'
    print(formatted_string)

    tend = time() - tinit

    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index, tend


# ---------------------------------------------------------------------------------------------------------------------
# run_svm_model()
# ---------------------------------------------------------------------------------------------------------------------
def run_svm_model(X_train, X_test, y_train, y_test, args, class_type='single-label'):

    if (not args.optimc):
        
        print("Training default Support Vector Machine model...")

        svc = LinearSVC(max_iter=1000)
        svc.fit(X_train, y_train)
        y_pred_default = svc.predict(X_test)
        
        #print("\nDefault Support Vector Mechine Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_default, digits=4))

        y_preds = y_pred_default

    elif (args.optimc):                             # Optimize Support Vector Machine with GridSearchCV

        print("Optimizing Support Vector Machine model with GridSearchCV...")

        param_grid = {
            'penalty': ['l1', 'l2'],                        # Regularization method
            'loss': ['hinge', 'squared_hinge'],             # Loss function
            'multi_class': ['ovr', 'crammer_singer'],       # Multi-class strategy
            'class_weight': [None, 'balanced'],             # Class weights
            'dual': [True, False],                          # Dual or primal formulation
            'C': np.logspace(-3, 3, 7)                      # Regularization parameter   
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
            estimator=LinearSVC(max_iter=1000),
            refit='f1_score',
            param_grid=param_grid,
            cv=cross_validation,
            scoring=scorers,
            return_train_score=True                         # ensure train scores are calculated
            )

        grid_search.fit(X_train, y_train)                   # Fit the model

        print('Best parameters: {}'.format(grid_search.best_params_))
        print("best_estimator:", grid_search.best_estimator_)
        print('Best score: {}'.format(grid_search.best_score_))
        #print("cv_results_:", grid_search.cv_results_)

        results = grid_search.cv_results_

        # Extract the best estimator from the GridSearchCV
        best_model = grid_search.best_estimator_

        # Predict on the test set using the best model
        y_pred_best = best_model.predict(X_test)

        print("Accuracy best score:", metrics.accuracy_score(y_test, y_pred_best))
        print(classification_report(y_true=y_test, y_pred=y_pred_best, digits=4))

        y_preds = y_pred_best

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

    Mf1, mf1, accuracy, h_loss, precision, recall, j_index = evaluation(y_test, y_preds, class_type)

    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index

# ---------------------------------------------------------------------------------------------------------------------
# run_lr_model()
# ---------------------------------------------------------------------------------------------------------------------
def run_lr_model(X_train, X_test, y_train, y_test, args, class_type='single-label'):

    # Default Logistic Regression Model
    print("Training default Logistic Regression model...")
    
    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
    
    if (not args.optimc):
        
        print("Optimization not requested, training default Logistic Regression model...")
    
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_default = lr.predict(X_test)
        
        #print("\nDefault Logistic Regression Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_default, digits=4))

        y_preds = y_pred_default

    elif (args.optimc):
        
        # Optimize Logistic Regression with GridSearchCV
        print("Optimizing Logistic Regression model with GridSearchCV...")

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
            estimator =  LogisticRegression(max_iter=1000),
            param_grid=param_grid,
            scoring=scorers,
            refit='f1_score',  # Optimize on F1 Score
            cv=StratifiedKFold(n_splits=5),
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)           # Fit the model

        # Display the best parameters
        print('Best parameters found by GridSearchCV:')
        print(grid_search.best_params_)

        # Evaluate on the test set
        y_pred_optimized = grid_search.best_estimator_.predict(X_test)

        #print("\nOptimized Logistic Regression Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_optimized):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_optimized, digits=4))

        y_preds = y_pred_optimized

    if (args.cm):
        # Optionally, plot confusion matrix for the optimized model
        create_confusion_matrix(
            y_test, 
            y_pred_optimized, 
            title='Confusion Matrix for Optimized Logistic Regression Model',
            file_name=OUT_DIR+'bbc_news_logistic_regression_confusion_matrix.png',
            debug=False
        )

    Mf1, mf1, accuracy, h_loss, precision, recall, j_index = evaluation(y_test, y_preds, class_type)

    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index


# ---------------------------------------------------------------------------------------------------------------------
# run_nb_model()
# ---------------------------------------------------------------------------------------------------------------------
def run_nb_model(X_train, X_test, y_train, y_test, args):

    print("Building default Naive Bayes Classifier...")

    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
    
    if (not args.optimc):
            
        print("Optimization not requested, training default Naive Bayes model...")
        
        nb = MultinomialNB()
        nb.fit(X_train,y_train)
        test_predict = nb.predict(X_test)

        train_accuracy = round(nb.score(X_train,y_train)*100)
        test_accuracy =round(accuracy_score(test_predict, y_test)*100)

        print("Naive Bayes Train Accuracy Score : {}% ".format(train_accuracy ))
        print("Naive Bayes Test Accuracy Score  : {}% ".format(test_accuracy ))
        print(classification_report(y_true=y_test, y_pred=test_predict, digits=4))

        y_preds = test_predict

    elif (args.optimc):

        print("Optimizing the model using GridSearchCV...")
        
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

        #print("\nBest Estimator's Test Set Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred, average='micro'):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred, average='micro'):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='micro'):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred, digits=4))

        y_preds = y_pred

        if (args.cm):
            # Optionally, plot confusion matrix
            create_confusion_matrix(
                y_test, 
                y_pred, 
                title='Confusion Matrix for Optimized Naive Bayes Model',
                file_name=OUT_DIR+'bbc_news_naive_bayes_confusion_matrix.png',
                debug=False
            )
    
    Mf1, mf1, accuracy, h_loss, precision, recall, j_index = evaluation(y_test, y_preds, class_type)

    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index





import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, LlamaTokenizer

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
# Function to create embeddings for a given text
# -------------------------------------------------------------------------------------------------------------------
def create_embedding(texts, model, embedding_dim):
    embeddings = np.zeros((len(texts), embedding_dim))
    for i, text in enumerate(texts):
        words = text.split()
        word_embeddings = [model[word] for word in words if word in model]
        if word_embeddings:
            embeddings[i] = np.mean(word_embeddings, axis=0)            # Average the word embeddings
        else:
            embeddings[i] = np.zeros(embedding_dim)                     # Use a zero vector if no words are in the model
    return embeddings
# -------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------
# Tokenize text using BERT tokenizer and vectorize accordingly
# -------------------------------------------------------------------------------------------------------------------
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
# -------------------------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------------------------
# gen_embeddings()
#
# Embedding generation function with BERT tokenization
# -------------------------------------------------------------------------------------------------------------------
def gen_embeddings(X_train, y_train, X_test, dataset='bbc-news', vocab=None, pretrained=None, pretrained_vectors=None, supervised=False, mode='solo'):
    
    print("\n\tgenerating embeddings...")
        
    print('X_train:', type(X_train), X_train.shape)
    print("y_train:", type(y_train), y_train.shape)
    print('X_test:', type(X_test), X_test.shape)
    
    print("dataset:", dataset)
    print("vocab:", type(vocab), len(vocab))
    print("pretrained:", pretrained)
    print("pretrained_vectors:", type(pretrained_vectors), pretrained_vectors.shape)
    print("supervised:", supervised)
    print("mode:", mode)

    # build vocabulary numpy array
    vocabulary = np.asarray(list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0])
    print("vocabulary:", type(vocabulary), vocabulary.shape)

    pretrained_embeddings = []                  # List to store pretrained embeddings

    if pretrained in ['word2vec', 'glove', 'fasttext', 'bert', 'llama']:
        
        print("setting up pretrained embeddings...")

        #P = pretrained_vectors.extract(vocabulary).numpy()
        print("pretrained_vectors: ", type(pretrained_vectors), {pretrained_vectors.shape})

        pretrained_embeddings.append(pretrained_vectors)
        #print(f'pretrained embeddings count after loading pretrained embeddings: {len(pretrained_embeddings[0])}')

    if supervised:
        
        print('setting up supervised embeddings...')

        """
        Xtr, _ = dataset.vectorize()
        Ytr = dataset.devel_labelmatrix
        """
        Xtr = X_train
        Ytr = y_train
        
        S = get_supervised_embeddings(Xtr, Ytr)
        print("supervised_embeddings:", type(S), {S.shape})

        pretrained_embeddings.append(S)
        #print(f'pretrained embeddings count after appending supervised: {len(pretrained_embeddings[1])}')

    embedding_matrix = np.hstack(pretrained_embeddings)
    print("after np.hstack():", type(embedding_matrix), {embedding_matrix.shape})

    if mode == 'solo':
        print("Using just the word embeddings alone (solo)...")
        # Directly return the embedding matrix for X_train and X_test
        X_train = embedding_matrix
        X_test = embedding_matrix                   # This assumes X_test has the same vocab and order as X_train
    elif mode == 'cat':
        print("Concatenating word embeddings with TF-IDF vectors...")
        X_train = np.hstack([X_train.toarray(), embedding_matrix])
        X_test = np.hstack([X_test.toarray(), embedding_matrix])
    elif mode == 'dot':
        print("Dot product (matrix multiplication) of word embeddings with TF-IDF vectors...")
        X_train = X_train.dot(embedding_matrix)
        X_test = X_test.dot(embedding_matrix)

    print("after embedding generation...")
    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
   
    """
    if pretrained == 'bert':
        print("Using BERT pretrained embeddings...")

        BERT_MODEL = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        cache_path = f'../.vector_cache/{BERT_MODEL}_{dataset}.npz'
        train_embeddings, test_embeddings = load_cached_embeddings(cache_path, debug=True)

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
        elif moode == 'tfidf':
            print("using just the TF-IDF vectors...")
            X_train = X_train
            X_test = X_test
    """

    return X_train, X_test



# --------------------------------------------------------------------------------------------------------------
# Core processing function
# --------------------------------------------------------------------------------------------------------------
def classify_data(dataset='20newsgrouops', pretrained_embeddings=None, embedding_path=None, supervised=False, method=None, args=None, logfile=None, system=None):
    
    print("\n\tclassify_data()...")
    
    #
    # load the dataset using appropriate tokenization method as dictated by pretrained embeddings
    #
    pickle_file_name=f'{dataset}_{args.pretrained}_tokenized.pickle'

    #print(f"Loading data set {dataset}...")

    pickle_file = PICKLE_DIR + pickle_file_name                                     # Define the path to the pickle file

    if os.path.exists(pickle_file):                                                 # if the pickle file exists
        print(f"Loading tokenized data from '{pickle_file}'...")
        X, y, embedding_matrix, vocab = load_from_pickle(pickle_file)
    else:
        print(f"'{pickle_file}' not found, loading {dataset}...")
        X, y, embedding_matrix, vocab = load_data(dataset=dataset, pretrained=args.pretrained, embedding_path=embedding_path)

        save_to_pickle(X, y, embedding_matrix, vocab, pickle_file)                              # Save the tokenized matrices to a pickle file

    print("Tokenized data loaded.")
 
    print("X:", type(X), X.shape)
    print("y:", type(y), y.shape)

    print("train_test_split...")

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

    #print("y_train:", y_train)
    #print("y_test:", y_test)

    y_train = y_train.toarray().ravel()             # Ensure y_train is a 1D array
    y_test = y_test.toarray().ravel()               # Ensure y_test is a 1D array

    print('y_train:', type(y_train), y_train.shape)
    print('y_test:', type(y_test), y_test.shape)

    emb_path = None

    if (pretrained_embeddings is not None):
        print("Using pretrained embeddings...")
        if (pretrained_embeddings == 'word2vec'):
            emb_path = args.word2vec_path
        elif (pretrained_embeddings == 'glove'):
            emb_path = args.glove_path
        elif (pretrained_embeddings == 'fasttext'):
            emb_path = args.fasttext_path
        elif (pretrained_embeddings == 'bert'):
            emb_path = args.bert_path
        elif (pretrained_embeddings == 'llama'):
            emb_path = args.llama_path

    print("embeddings:", pretrained_embeddings)
    print("emb_path:", emb_path)

    if (args.pretrained is None) and (args.supervised == False):        # no embeddings in this case
        sup_tend = 0
    else:                                                               # embeddings are present
        tinit = time()

        print("building the embeddings...")

        # Generate embeddings
        X_train, X_test = gen_embeddings(
            X_train=X_train,
            y_train=y_train, 
            X_test=X_test, 
            dataset=dataset,
            vocab=vocab, 
            pretrained=pretrained_embeddings,
            pretrained_vectors=embedding_matrix,
            supervised=args.supervised,
            mode=args.mode
            )
        
        sup_tend = time() - tinit

    Mf1, mf1, acc, h_loss, precision, recall, j_index, tend = run_model(X_train, X_test, y_train, y_test, args)

    tend += sup_tend

    logfile.add_layered_row(tunable=False, measure='final-te-macro-F1', value=Mf1, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='final-te-micro-F1', value=mf1, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='te-accuracy', value=acc, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='te-hamming-loss', value=h_loss, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='te-precision', value=precision, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='te-recall', value=recall, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='te-jacard-index', value=j_index, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())

# -------------------------------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    available_datasets = LCDataset.dataset_available

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification Testing')
    
    parser.add_argument('--dataset', required=True, type=str, default='20newsgroups', metavar='N', help=f'dataset, one in {available_datasets}')
    
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str', help=f'path where to load the pickled dataset from')
    
    parser.add_argument('--log-file', type=str, default='../log/ml_classify.test', metavar='N', help='path to the application log file')
    
    parser.add_argument('--learner', type=str, default='svm', metavar='N', help=f'learner (svm, lr, or nb)')
    
    parser.add_argument('--mode', type=str, default='solo', metavar='N', help=f'way to prepare the embeddings, in [tfidf, solo, cat, dot]')

    parser.add_argument('--supervised', action='store_true', default=False, help='use supervised embeddings')

    parser.add_argument('--cm', action='store_true', default=False, help=f'create confusion matrix')

    parser.add_argument('--plot', action='store_true', default=False, help=f'create plots of GridSearchCV metrics (if --optimc is True)')
                             
    parser.add_argument('--optimc', action='store_true', default=False, help='optimize the model using relevant models params')

    parser.add_argument('--force', action='store_true', default=False,
                    help='force the execution of the experiment even if a log already exists')
    
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

    print("\n\t-------------------------------------------------------- Layer Cake: ML baseline classification code: main() --------------------------------------------------------")

    print("args:", type(args), args)

    print("available_datasets:", available_datasets)

    # check dataset
    assert args.dataset in available_datasets, \
        f'unknown dataset {args.dataset}'

    # initialize log file and run params
    already_modelled, vtype, learner, pretrained, embeddings, emb_path, supervised, method_name, logfile = initialize(args)

    # check to see if model params have been computed already
    if (already_modelled) and not (args.force):
        print(f'Assertion warning: model {method_name} with embeddings {embeddings}, pretrained == {pretrained} and wc_supervised == {args.supervised} for {args.dataset} already calculated.')
        print("Run with --force option to override, returning...")
        exit(0)

    sys = SystemResources()
    print("system resources:", sys)

    classify_data(
        dataset=args.dataset, 
        pretrained_embeddings=embeddings,
        embedding_path=emb_path,
        supervised=supervised,
        method=method_name,
        args=args, 
        logfile=logfile, 
        system=SystemResources()
        )
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------