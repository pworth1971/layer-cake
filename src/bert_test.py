import numpy as np
import os
import argparse

import plotly.offline as pyo
import plotly.graph_objs as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, hamming_loss, jaccard_score
from sklearn.metrics import classification_report, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from data.lc_dataset import LCDataset

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from transformers import BertTokenizerFast, BertModel, RobertaTokenizerFast, RobertaModel

from tqdm import tqdm


import warnings
from click.core import batch
warnings.filterwarnings('ignore')



#
# we assume everything runs from bin directory
#
PICKLE_DIR = '../pickles/'
OUT_DIR = '../out/'
DATASET_DIR = '../datasets/'
VECTOR_CACHE = '../.vector_cache'

MAX_VOCAB_SIZE = 10000                                      # max feature size for TF-IDF vectorization

BERT_MODEL      = 'bert-base-uncased'               # dimension = 768
LLAMA_MODEL     = 'meta-llama/Llama-2-7b-hf'        # dimension = 4096
ROBERTA_MODEL   = 'roberta-base'                    # dimension = 768
#LLAMA_MODEL = 'meta-llama/Llama-2-13b-hf'

TOKEN_TOKENIZER_MAX_LENGTH = 512

# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 16
DEFAULT_GPU_BATCH_SIZE = 64
MPS_BATCH_SIZE = 16


EPOCHS = 30

NUM_UNFROZEN_MODEL_LAYERS = 2

#dataset_available = {'reuters21578', '20newsgroups', 'ohsumed', 'rcv1'}
dataset_available = {'20newsgroups', 'bbc-news'}


import warnings
warnings.filterwarnings("ignore")




    
def run_svm_model(X_train, X_test, y_train, y_test, category_names, args):

    print("Training default Support Vector Machine model...")
    
    default_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('lr', LinearSVC(max_iter=1000))
    ])

    default_pipeline.fit(X_train, y_train)
    y_pred_default = default_pipeline.predict(X_test)

    print("\nDefault Support Vector Mechine Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
    print(classification_report(y_true=y_test, y_pred=y_pred_default, target_names=category_names, digits=4))

    if (args.optimc):

        # Optimize Support Vector Machine with GridSearchCV
        print("Optimizing Support Vector Machine model with GridSearchCV...")

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
            estimator=pipeline,
            refit='f1_score',
            param_grid=param_grid,
            cv=cross_validation,
            #scoring=scoring
            scoring=scorers,
            return_train_score=True         # ensure train scores are calculated
            )

        # Fit the model
        grid_search.fit(X_train, y_train)

        print('Best parameters: {}'.format(grid_search.best_params_))
        print("best_estimator:", grid_search.best_estimator_)
        print('Best score: {}'.format(grid_search.best_score_))
        print("cv_results_:", grid_search.cv_results_)

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
        print(classification_report(y_true=y_test, y_pred=y_pred_best, target_names=category_names, digits=4))
        
        return y_test, y_pred_best
    else:
        return y_test, y_pred_default


def run_lr_model(X_train, X_test, y_train, y_test, category_names, args):

    # Default Logistic Regression Model
    print("Training default Logistic Regression model...")
    default_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('lr', LogisticRegression(max_iter=1000))
    ])

    default_pipeline.fit(X_train, y_train)
    y_pred_default = default_pipeline.predict(X_test)

    print("\nDefault Logistic Regression Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
    print(classification_report(y_true=y_test, y_pred=y_pred_default, target_names=category_names, digits=4))

    if (args.optimc):
        # Optimize Logistic Regression with GridSearchCV
        print("Optimizing Logistic Regression model with GridSearchCV...")

        # Define the pipeline
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
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scorers,
            refit='f1_score',  # Optimize on F1 Score
            cv=StratifiedKFold(n_splits=5),
            n_jobs=-1,
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
        print(classification_report(y_true=y_test, y_pred=y_pred_optimized, target_names=category_names, digits=4))
        
        return y_test, y_pred_optimized
    else:
        return y_test, y_pred_default



def run_nb_model(X_train, X_test, y_train, y_test, category_names, args):

    print("Building default Naive Bayes Classifier...")

    nb = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
        ])
    
    nb.fit(X_train,y_train)

    test_predict = nb.predict(X_test)

    train_accuracy = round(nb.score(X_train,y_train)*100)
    test_accuracy =round(accuracy_score(test_predict, y_test)*100)

    print("Naive Bayes Train Accuracy Score : {}% ".format(train_accuracy ))
    print("Naive Bayes Test Accuracy Score  : {}% ".format(test_accuracy ))
    print(classification_report(y_true=y_test, y_pred=test_predict, target_names=category_names, digits=4))

    if (args.optimc):

        print("Optimizing the model using GridSearchCV...")

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
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scorers,
            refit='f1_score',                           # Optimize on F1 Score
            cv=StratifiedKFold(n_splits=5),
            n_jobs=-1,
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
        print(classification_report(y_true=y_test, y_pred=y_pred, target_names=category_names, digits=4))

        return y_test, y_pred
    else:
        return y_test, test_predict
            
            

def run_model(X_train, X_test, y_train, y_test, category_names, confusion_matrix=False):

    print("\n\tRunning model...")

    print("category_names:", category_names)
    
    # Support Vector Machine Classifier
    if (args.learner == 'svm'):
        y_test, y_pred = run_svm_model(X_train, X_test, y_train, y_test, category_names, args)

    # Logistic Regression Classifier
    elif (args.learner == 'lr'):
        y_test, y_pred = run_lr_model(X_train, X_test, y_train, y_test, category_names, args)

    # Naive Bayes (MultinomialNB) Classifier
    elif (args.learner == 'nb'):
        y_test, y_pred = run_nb_model(X_train, X_test, y_train, y_test, category_names, args)

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
        print(classification_report(y_true=y_test, y_pred=test_predict, target_names=category_namees, digits=4))

        y_pred = test_predict
        
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
        print(classification_report(y_true=y_test, y_pred=test_predict, target_names=category_names, digits=4))

        y_pred = test_predict
    else:
        print(f"Invalid learner '{args.learner}'")
        return
    
    
    if (confusion_matrix):
        # Optionally, plot confusion matrix
        create_confusion_matrix(
            y_test, 
            y_pred, 
            category_names,
            title='Confusion Matrix',
            file_name=f'{OUT_DIR}{args.dataset}_{args.learner}_confusion_matrix.png',
            debug=False
        )



# --------------------------------------------------------------------------------------------------------------
# Core processing function
# --------------------------------------------------------------------------------------------------------------
def classify(dataset='20newsgrouops', args=None, device='cpu'):
    
    if (args is None):
        print("No arguments passed.")
        return
    
    if (args.dataset not in dataset_available):
        print(f"Dataset '{args.dataset}' not available.")
        return
    
    print("device:", device)
    
    """
    pickle_file_name=f'{dataset}_{args.mode}_tokenized.pickle'
    pickle_file = PICKLE_DIR + pickle_file_name                     # Define the path to the pickle file
    """

    print(f"Classifying {dataset}...")

    print(f"Loading data set {dataset}...")
    

    #df, categories = load_data(dataset)
    df, category_names = load_data(dataset)

    print("df:", df.shape)
    print(df.head())

    print("Tokenized data loaded, df:", df.shape)
    print(df.head())
    
    print("categories:", category_names)

    print("Processing data...")

    if (dataset == '20newsgroups'):

        print("classifying 20 newsgroups...")

        # Feature Extraction and Model Training
        print("Splitting the dataset...")

        X = df['CleanedText']
        y = df['category']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

        print("X_train:", type(X_train), X_train.shape)
        print("X_test:", type(X_test), X_test.shape)
        
        print("Y_train:", type(y_train), y_train.shape)
        print("Y_test:", type(y_test), y_test.shape)

        print("Running model...")

        run_model(X_train, X_test, y_train, y_test, category_names, args)
    
    elif (dataset == 'bbc-news'):

        print(f'classifying bbc-news...')
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['Text'],
            df['Category'], 
            test_size = 0.3, 
            random_state = 60,
            shuffle=True, 
            stratify=df['Category']
            )

        print("X_train:", type(X_train), X_train.shape)
        print("X_test:", type(X_test), X_test.shape)
        
        print("Y_train:", type(y_train), y_train.shape)
        print("Y_test:", type(y_test), y_test.shape)

        print("Running model...")

        run_model(X_train, X_test, y_train, y_test, category_names, args)
        


def get_model_data(dataset='reuters21578'):
    print("Getting model data...")

    # Load data    
    print(f"Loading data set {dataset}...")
    lcd = LCDataset(dataset)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = lcd.split()

    print("X_train:", type(X_train), len(X_train))
    print("X_test:", type(X_test), len(X_test))

    print("y_train:", type(y_train), y_train.shape)
    print("y_test:", type(y_test), y_test.shape)

    print("class_type:", lcd.class_type)

    # For multilabel datasets, y_train and y_test are already in the correct format (no need for binarization)
    if lcd.class_type in ['multilabel', 'multi-label']:
        print("Multilabel classification detected. Data is already in binary format, skipping binarization.")
    else:
        print("Single-label classification detected. Data is already encoded, skipping LabelEncoder.")
        # No need for additional processing as the data is already encoded

    print("y_train:", type(y_train), y_train.shape)
    print("y_test:", type(y_test), y_test.shape)

    # Return all necessary information
    return X_train, X_test, y_train, y_test, lcd.target_names, lcd.nC, lcd.class_type




def get_model_data_deprecated(dataset='20newsgroups'):

    print("getting model data...")
    

    # Load data    
    print(f"Loading data set {dataset}...")
    lcd = LCDataset(dataset)

    X_train, X_test, y_train, y_test = lcd.split()                  # split data

    print("X_train:", type(X_train), len(X_train))
    print("X_test:", type(X_test), len(X_test))

    print("y_train:", type(y_train), y_train.shape)
    print("y_test:", type(y_test), y_test.shape)

    print("encoding y data...")

    # Convert categorical labels in y_train and y_test to numeric
    label_encoder = LabelEncoder()

    # Fit on the training data and transform both y_train and y_test
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Print encoded labels to confirm
    print("y_train:", type(y_train), y_train.shape)
    print("Encoded y_train:", y_train[:5])
    print("y_test:", type(y_test), y_test.shape)
    print("Encoded y_test:", y_test[:5])
    
    return X_train, X_test, y_train, y_test, lcd.target_names, lcd.nC








# Custom dataset class for loading and tokenizing text data
class BertDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        target = self.targets[index]
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,                    # Add truncation to ensure proper token length
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )
        return {
            'ids': torch.tensor(inputs["input_ids"], dtype=torch.long),
            'mask': torch.tensor(inputs["attention_mask"], dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

# Updated classifier class to support both BERT and RoBERTa
class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, model_name='bert-base-uncased'):
        super(TransformerClassifier, self).__init__()
        if 'roberta' in model_name:
            self.transformer_model = RobertaModel.from_pretrained(model_name, cache_dir=VECTOR_CACHE + '/RoBERTa')
        else:
            self.transformer_model = BertModel.from_pretrained(model_name, cache_dir=VECTOR_CACHE + '/BERT')

        self.out = nn.Linear(self.transformer_model.config.hidden_size, num_classes)

    def forward(self, ids, mask):
        outputs = self.transformer_model(ids, attention_mask=mask)
        pooled_output = outputs[1]  # CLS token output for both BERT and RoBERTa
        return self.out(pooled_output)




# BERT Model for multi-class classification
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT')
        self.out = nn.Linear(768, num_classes)              # Output for multi-class classification

    def forward(self, ids, mask):
        _, pooled_output = self.bert_model(ids, attention_mask=mask, return_dict=False)
        return self.out(pooled_output)



def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, patience=2, multilabel=False):
    """
    Train the model for both single-label and multi-label classification.
    
    Args:
    - model: The PyTorch model to be trained.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - optimizer: Optimizer for updating model weights.
    - loss_fn: Loss function for both single-label and multi-label classification.
    - device: Device for computation (CPU/GPU).
    - epochs: Number of epochs for training.
    - multilabel: Boolean flag to indicate whether it is multilabel classification.
    - patience: Patience for early stopping.
    
    Returns:
    None
    """

    print("Training model...")

    model.to(device)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        # Training loop
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(ids=ids, mask=mask)

            if multilabel:
                # Compute loss for entire batch and all labels at once
                total_loss = loss_fn(outputs, targets.float())  # Convert targets to float for BCEWithLogitsLoss
            else:
                # Use a single loss function for single-label classification
                total_loss = loss_fn(outputs, targets)

            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                targets = batch['target'].to(device)
                outputs = model(ids=ids, mask=mask)

                if multilabel:
                    val_loss = loss_fn(outputs, targets.float())  # Convert targets to float for BCEWithLogitsLoss
                else:
                    val_loss = loss_fn(outputs, targets)

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

        # Early stopping logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), VECTOR_CACHE + '/best_model_state.bin')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break



def evaluate_model(model, test_loader, device, target_names, threshold=0.5, multilabel=False):
    """
    Evaluate the model on the test set and generate a classification report.

    Args:
    - model: Trained model.
    - test_loader: DataLoader for the test set.
    - device: Device on which to perform inference (CPU, CUDA, MPS).
    - target_names: List of class names for the classification report.
    - threshold: Threshold for converting logits to binary in multilabel classification.
    - multilabel: Flag indicating if the task is multilabel classification.
    """

    print("Evaluating model...")

    model.to(device)
    model.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['target'].cpu().numpy()  # Move targets to CPU and convert to numpy for comparison

            outputs = model(ids=ids, mask=mask)

            if multilabel:
                # Apply sigmoid and threshold to get binary predictions for multilabel classification
                probs = torch.sigmoid(outputs).cpu().numpy()  # Get probabilities
                preds = (probs >= threshold).astype(int)  # Binarize based on the threshold (0.5 default)
            else:
                # Single-label classification: Use argmax to get the predicted class
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_targets.extend(targets)
            all_preds.extend(preds)

    # Convert lists to numpy arrays for consistency
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Debugging: print the shapes of the targets and predictions
    print(f"all_targets shape: {all_targets.shape}, all_preds shape: {all_preds.shape}")

    if multilabel:
        # Multilabel classification metrics
        print("Multilabel Classification Report:")
        report = classification_report(y_true=all_targets, y_pred=all_preds, target_names=target_names, digits=4, zero_division=0, output_dict=True)
        
        # Calculate subset accuracy for multilabel classification
        subset_acc = accuracy_score(all_targets, all_preds)
        print(f"Subset Accuracy: {subset_acc:.4f}")

    else:
        # Single-label classification report
        print("Single-label Classification Report:")
        report = classification_report(y_true=all_targets, y_pred=all_preds, target_names=target_names, digits=4, zero_division=0, output_dict=True)

    # Print the detailed classification report
    for label, metrics in report.items():
        if label in target_names:  # Only print actual class labels, skip "accuracy", "macro avg", etc.
            print(f"Class: {label}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1-score']:.4f}")
            print(f"  Support: {metrics['support']}")

    # Summary statistics
    print(f"\nSummary:")
    if 'macro avg' in report:
        print(f"Macro Avg: Precision: {report['macro avg']['precision']:.4f}, Recall: {report['macro avg']['recall']:.4f}, F1-score: {report['macro avg']['f1-score']:.4f}")
    if 'weighted avg' in report:
        print(f"Weighted Avg: Precision: {report['weighted avg']['precision']:.4f}, Recall: {report['weighted avg']['recall']:.4f}, F1-score: {report['weighted avg']['f1-score']:.4f}")






def compute_multilabel_class_weights(y_train):
    """
    Compute class weights for each label in a multilabel classification setting.
    
    Parameters:
    - y_train: Numpy array with shape (n_samples, n_classes)
    
    Returns:
    - class_weights: List of class weights for each label
    """
    class_weights = []

    for i in range(y_train.shape[1]):
        y_label = y_train[:, i]
        # Compute weights for each label independently
        weights = compute_class_weight('balanced', classes=np.unique(y_label), y=y_label)
        class_weights.append(torch.tensor(weights, dtype=torch.float))  # Ensure this is a tensor

    return class_weights




def fine_tune_model2(args, device, batch_size=MPS_BATCH_SIZE, epochs=EPOCHS, max_length=TOKEN_TOKENIZER_MAX_LENGTH, layers_to_unfreeze=0, model_name='bert-base-uncased'):
    """
    Fine-tunes the BERT or RoBERTa model and tests it.
    
    Args:
    - args: Argument object containing dataset information.
    - device: Device to run the model on (CPU/GPU).
    - batch_size: Batch size for training.
    - epochs: Number of training epochs.
    - max_length: Maximum token length for BERT tokenization.
    - layers_to_unfreeze: Number of BERT layers to unfreeze for fine-tuning.
    
    Returns:
    None
    """
    print("Fine-tuning and testing model...")

    # Load training and testing data
    X_train, X_test, y_train, y_test, category_names, _, class_type = get_model_data(args.dataset)

    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
    print("y_train:", type(y_train), y_train.shape)
    print("y_test:", type(y_test), y_test.shape)

    # Split validation data from test set
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    print("X_val:", type(X_val), X_val.shape)
    print("X_test:", type(X_test), X_test.shape)
    print("y_val:", type(y_val), y_val.shape)
    print("y_test:", type(y_test), y_test.shape)

    # Load the appropriate tokenizer
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name, cache_dir=VECTOR_CACHE + '/RoBERTa')
    elif 'bert' in model_name:
        tokenizer = BertTokenizerFast.from_pretrained(model_name, cache_dir=VECTOR_CACHE + '/BERT')
    else:
        print("Invalid model name. Please provide a valid BERT or RoBERTa model.")
        return

    # Create dataset and data loaders
    train_dataset = BertDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = BertDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = BertDataset(X_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Ensure that num_classes matches the number of labels in the dataset
    if len(y_train.shape) > 1:  # Multi-label case
        num_classes = y_train.shape[1]
    else:  # Single-label case
        num_classes = len(np.unique(y_train))  # Use the number of unique labels
    print("Number of classes in the dataset:", num_classes)

    # Initialize the model based on the selected transformer (BERT or RoBERTa)
    model = TransformerClassifier(num_classes=num_classes, model_name=model_name).to(device)

    # Compute class weights for single-label case, ignored in multilabel case
    if class_type == 'multilabel':
        loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits for multilabel
    else:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize optimizer after model initialization
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Freeze layers if needed
    if layers_to_unfreeze > 0:
        print(f"Unfreezing the last {layers_to_unfreeze} layers...")
        for param in model.transformer_model.parameters():
            param.requires_grad = False
        for layer in model.transformer_model.encoder.layer[-layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
    else:
        print("BERT model is static (no layers are unfrozen).")

    # Fine-tune model
    train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs, multilabel=(class_type == 'multilabel'))

    # Test the model on test set
    evaluate_model(model, test_loader, device, category_names, multilabel=(class_type == 'multilabel'))






def fine_tune_model(args, device, batch_size=MPS_BATCH_SIZE, epochs=EPOCHS, max_length=TOKEN_TOKENIZER_MAX_LENGTH, layers_to_unfreeze=0):
    """
    Fine-tunes the BERT model and tests it.
    
    Args:
    - args: Argument object containing dataset information.
    - device: Device to run the model on (CPU/GPU).
    - batch_size: Batch size for training.
    - epochs: Number of training epochs.
    - max_length: Maximum token length for BERT tokenization.
    - layers_to_unfreeze: Number of BERT layers to unfreeze for fine-tuning.
    
    Returns:
    None
    """
    
    print("fine-tuning and testing model...")

    print("device:", device)
    print("batch_size:", batch_size)
    print("epochs:", epochs)
    print("max_length:", max_length)
    print("layers_to_unfreeze:", layers_to_unfreeze)

    # Load training and testing data
    X_train, X_test, y_train, y_test, category_names, num_classes = get_model_data(args.dataset)

    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
    print("y_train:", type(y_train), y_train.shape)
    print("y_test:", type(y_test), y_test.shape)

    # Split validation data from test set
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    print("X_val:", type(X_val), X_val.shape)
    print("X_test:", type(X_test), X_test.shape)
    print("y_val:", type(y_val), y_val.shape)
    print("y_test:", type(y_test), y_test.shape)

    # Load BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT')

    """
    # Reset indexes
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    """

    # Create dataset and data loaders
    train_dataset = BertDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = BertDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = BertDataset(X_test, y_test, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = BERTClassifier(num_classes=num_classes).to(device)
    print("model initialized:\n", model)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Set loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # Freeze BERT layers if layers_to_unfreeze is provided
    if layers_to_unfreeze > 0:
        print(f"Unfreezing the last {layers_to_unfreeze} layers of BERT...")
        for param in model.bert_model.parameters():
            param.requires_grad = False  # Freeze all layers initially
        
        # Unfreeze the last N layers as specified
        for layer in model.bert_model.encoder.layer[-layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
    else:
        print("BERT model is static (no layers are unfrozen).")

    # Fine-tune model with validation and early stopping
    train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs)

    # Test the model on test set
    evaluate_model(model, test_loader, device, category_names)







    
    
if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Text Classification with BERT.")
    
    parser.add_argument('--dataset', type=str, default='20newsgroups', help='Dataset to use: 20newsgroups or bbc-news.')
    
    parser.add_argument('--learner', type=str, default='svm', help='Choose the learner: nn, ft, svm, lr, nb.')

    parser.add_argument('--pretrained', type=str, default=None, metavar='bert|roberta',
                        help='pretrained embeddings, use "bert", or "roberta" (default None)')
    
    parser.add_argument('--embedding-dir', type=str, default='../.vector_cache', metavar='str',
                        help=f'path where to load and save document embeddings')
    
    parser.add_argument('--bert-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to BERT pretrained vectors, used only with --pretrained bert')
    
    parser.add_argument('--roberta-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to RoBERTa pretrained vectors, used only with --pretrained roberta')

    parser.add_argument('--static', action='store_true', help='keep the underlying pretrained model static (ie no unfrozen layers)')

    parser.add_argument('--optimc', action='store_true', help='Optimize classifier with GridSearchCV.')

    parser.add_argument('--cm', action='store_true', help='Generate confusion matrix.')
    
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
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)    
        
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

    if (args.static):
        num_unfrozen_layers = 0
    else:
        num_unfrozen_layers = NUM_UNFROZEN_MODEL_LAYERS

    print("num_unfrozen_layers:", num_unfrozen_layers)


    if (args.learner == 'ft'):
    
        print("learner is ft...")

        fine_tune_model2(
            args, 
            device, 
            batch_size=MPS_BATCH_SIZE, 
            epochs=EPOCHS,
            max_length=TOKEN_TOKENIZER_MAX_LENGTH,
            layers_to_unfreeze=num_unfrozen_layers,
            model_name=model_name
        )

    elif args.learner in ['svm', 'lr', 'nb', 'dt', 'rf']:

        print("learner:", args.learner)

        classify(args.dataset, args, device)
    else:
        print(f"Invalid learner '{args.learner}'")
        