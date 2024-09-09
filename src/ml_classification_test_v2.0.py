import numpy as np

import argparse
from time import time

from scipy.sparse import csr_matrix, csc_matrix

import plotly.offline as pyo
import plotly.graph_objs as go

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import make_scorer, recall_score, hamming_loss, jaccard_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder

from data.lc_dataset import LCDataset
from util.common import OUT_DIR, initialize, SystemResources
from util.metrics import evaluation
from util.multilabel_classifier import MLClassifier


import warnings
warnings.filterwarnings('ignore')

#
# we assume everything runs from bin directory
#
PICKLE_DIR = '../pickles/'
VECTOR_CACHE = '../.vector_cache'
OUT_DIR = '../out/'

TEST_SIZE = 0.2

NUM_JOBS = -1          # important to manage CUDA memory allocation
#NUM_JOBS = 40          # for rcv1 dataset which has 101 classes, too many to support in parallel


# -------------------------------------------------------------------------------------------------------------------------------------------------
# run_model()
# -------------------------------------------------------------------------------------------------------------------------------------------------
def run_model(X_train, X_test, y_train, y_test, args, target_names, class_type='singlelabel'):
    
    print("\n\tRunning model...")

    print('X_train:', type(X_train), X_train.shape)
    print('X_test:', type(X_test), X_test.shape)

    print('y_train:', type(y_train), y_train.shape)
    print('y_test:', type(y_test), y_test.shape)
    
    #print("y_train:", y_train)
    #print("y_test:", y_test)
        
    print("target_names:", target_names)
    print("class_type:", class_type)

    tinit = time()

    # Support Vector Machine Classifier
    if (args.learner == 'svm'):                                     
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = run_svm_model(
            args.dataset,
            X_train,
            X_test,
            y_train,
            y_test,
            args,
            target_names,
            class_type=class_type
            )
    
    # Logistic Regression Classifier
    elif (args.learner == 'lr'):                                  
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = run_lr_model(
            args.dataset,
            X_train,
            X_test,
            y_train,
            y_test,
            args,
            target_names,
            class_type=class_type
            )

    # Naive Bayes (MultinomialNB) Classifier
    elif (args.learner == 'nb'):                                  
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = run_nb_model(
            args.dataset,
            X_train,
            X_test,
            y_train,
            y_test,
            args,
            target_names,
            class_type=class_type
            )
    
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
def run_svm_model(dataset, X_train, X_test, y_train, y_test, args, target_names, class_type='singlelabel'):
    
    print("\n\trunning SVM model...")

    print("Training Support Vector Machine model using OneVsRestClassifier...")

    # Check if it's a multilabel problem, and use OneVsRestClassifier if true
    if class_type in ['multilabel', 'multi-label']:
        print("Multilabel classification detected. Using OneVsRestClassifier...")
        classifier = OneVsRestClassifier(LinearSVC(class_weight='balanced', dual='auto', max_iter=1000))
    else:
        print("Single-label classification detected. Using regular SVM...")
        classifier = LinearSVC(class_weight='balanced', dual='auto', max_iter=1000)

    if not args.optimc:
        svc = LinearSVC(dual='auto', max_iter=1000)
        ovr_svc = OneVsRestClassifier(svc)
        ovr_svc.fit(X_train, y_train)
        y_pred_default = ovr_svc.predict(X_test)

        #print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_default, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_default

    # Case with optimization using GridSearchCV
    else:

        print("Optimizing Support Vector Machine model with RandomizedSearchCV...")
        
        param_distributions = {
            'estimator__penalty': ['l1', 'l2'],
            'estimator__loss': ['hinge', 'squared_hinge'],
            'estimator__C': np.logspace(-3, 3, 7)
        } if class_type == 'multilabel' else {
            'penalty': ['l1', 'l2'],
            'loss': ['hinge', 'squared_hinge'],
            'C': np.logspace(-3, 3, 7)
        }

        # Add zero_division=0 to precision and recall to suppress the warnings
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro', zero_division=0),
            'precision_score': make_scorer(precision_score, average='micro', zero_division=0),
            'hamming_loss': make_scorer(hamming_loss),
        }

        # Wrap RandomizedSearchCV around OneVsRestClassifier if multilabel
        randomized_search = RandomizedSearchCV(
            estimator=classifier,
            param_distributions=param_distributions,
            scoring=scorers,
            refit='f1_score',
            n_jobs=-1,
            cv=5,
            return_train_score=True,
            n_iter=10  # Number of parameter settings sampled
        )

        randomized_search.fit(X_train, y_train)
        
        print('Best parameters:', randomized_search.best_params_)
        best_model = randomized_search.best_estimator_
        
        # Predict on test set
        y_pred_best = best_model.predict(X_test)

        #print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_best, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_best

        # confusion matrix reporting only works in single-label classification
        if (class_type in ['singlelabel', 'single-label']) and (args.cm):
            create_confusion_matrix(
                y_test,
                y_preds,
                target_names,
                title=f'Confusion Matrix for {dataset} with SVM Model',
                file_name=f'{OUT_DIR}{dataset}_svm_confusion.png',
                debug=True
            )

        y_preds = y_pred_best


        """
        print("Optimizing Support Vector Machine model with GridSearchCV...")
        
        param_grid = {
            'estimator__penalty': ['l1', 'l2'],
            'estimator__loss': ['hinge', 'squared_hinge'],
            'estimator__C': np.logspace(-3, 3, 7)
        } if class_type == 'multilabel' else {
            'penalty': ['l1', 'l2'],
            'loss': ['hinge', 'squared_hinge'],
            'C': np.logspace(-3, 3, 7)
        }

        # Add zero_division=0 to precision and recall to suppress the warnings
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro', zero_division=0),
            'precision_score': make_scorer(precision_score, average='micro', zero_division=0),
            'hamming_loss': make_scorer(hamming_loss),
        }

        # Wrap GridSearchCV around OneVsRestClassifier if multilabel
        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            scoring=scorers,
            refit='f1_score',
            n_jobs=-1,
            #cv=StratifiedKFold(n_splits=5),
            cv=5,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)
        
        print('Best parameters:', grid_search.best_params_)
        best_model = grid_search.best_estimator_
        
        # Predict on test set
        y_pred_best = best_model.predict(X_test)

        #print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_best, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_best

        # confusion matrix reporting only works in single-label classification
        if (class_type in ['singlelabel', 'single-label']) and (args.cm):
        
            create_confusion_matrix(
                y_test,
                y_preds,
                target_names,
                title=f'Confusion Matrix for {dataset} with SVM Model',
                file_name=f'{OUT_DIR}{dataset}_svm_confusion.png',
                debug=True
            )

        y_preds = y_pred_best
        """

    # Evaluate the model
    Mf1, mf1, accuracy, h_loss, precision, recall, j_index =    \
        evaluation(y_test, y_preds, classification_type=class_type, debug=False)
    
    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index



# ---------------------------------------------------------------------------------------------------------------------
# run_nb_model()
# ---------------------------------------------------------------------------------------------------------------------
def run_nb_model(dataset, X_train, X_test, y_train, y_test, args, target_names, class_type='singlelabel'):
    
    print("\n\trunning Naive Bayes model...")

    # Check if it's a multilabel problem, and use OneVsRestClassifier if true
    if class_type in ['multilabel', 'multi-label']:
        print("Multilabel classification detected. Using OneVsRestClassifier...")
        classifier = OneVsRestClassifier(MultinomialNB())
    else:
        print("Single-label classification detected. Using regular Naive Bayes...")
        classifier = MultinomialNB()

    if not args.optimc:
        nb = MultinomialNB()
        ovr_nb = OneVsRestClassifier(nb) if class_type in ['multilabel', 'multi-label'] else nb
        ovr_nb.fit(X_train, y_train)
        y_pred_default = ovr_nb.predict(X_test)

        #print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_default, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_default

    # Case with optimization using GridSearchCV
    else:
        print("Optimizing Naive Bayes model with RandomizedSearchCV...")

        param_distributions = {
            'estimator__alpha': [0.1, 0.5, 1.0, 2.0]  # Smoothing parameter for MultinomialNB
        } if class_type == 'multilabel' else {
            'alpha': [0.1, 0.5, 1.0, 2.0]
        }

        # Add zero_division=0 to precision and recall to suppress the warnings
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro', zero_division=0),
            'precision_score': make_scorer(precision_score, average='micro', zero_division=0),
            'hamming_loss': make_scorer(hamming_loss),
        }

        # Wrap RandomizedSearchCV around OneVsRestClassifier if multilabel
        randomized_search = RandomizedSearchCV(
            estimator=classifier,
            param_distributions=param_distributions,
            scoring=scorers,
            refit='f1_score',
            n_jobs=-1,
            cv=5,
            return_train_score=True,
            n_iter=10  # Number of parameter settings sampled
        )

        randomized_search.fit(X_train, y_train)
        
        print('Best parameters:', randomized_search.best_params_)
        best_model = randomized_search.best_estimator_
        
        # Predict on test set
        y_pred_best = best_model.predict(X_test)

        print(classification_report(y_true=y_test, y_pred=y_pred_best, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_best

        # Confusion matrix reporting only works in single-label classification
        if (class_type in ['singlelabel', 'single-label']) and (args.cm):
            create_confusion_matrix(
                y_test,
                y_preds,
                target_names,
                title=f'Confusion Matrix for {dataset} with Naive Bayes Model',
                file_name=f'{OUT_DIR}{dataset}_nb_confusion.png',
                debug=True
            )

            
        """
        print("Optimizing Naive Bayes model with GridSearchCV...")

        param_grid = {
            'estimator__alpha': [0.1, 0.5, 1.0, 2.0]  # Smoothing parameter for MultinomialNB
        } if class_type == 'multilabel' else {
            'alpha': [0.1, 0.5, 1.0, 2.0]
        }

        # Add zero_division=0 to precision and recall to suppress the warnings
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro', zero_division=0),
            'precision_score': make_scorer(precision_score, average='micro', zero_division=0),
            'hamming_loss': make_scorer(hamming_loss),
        }

        # Wrap GridSearchCV around OneVsRestClassifier if multilabel
        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            scoring=scorers,
            refit='f1_score',
            n_jobs=-1,
            cv=5,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)
        
        print('Best parameters:', grid_search.best_params_)
        best_model = grid_search.best_estimator_
        
        # Predict on test set
        y_pred_best = best_model.predict(X_test)

        #print(f"Accuracy: {accuracy_score(y_test, y_pred_best)::.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_best, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_best

        # Confusion matrix reporting only works in single-label classification
        if (class_type in ['singlelabel', 'single-label']) and (args.cm):
        
            create_confusion_matrix(
                y_test,
                y_preds,
                target_names,
                title=f'Confusion Matrix for {dataset} with Naive Bayes Model',
                file_name=f'{OUT_DIR}{dataset}_nb_confusion.png',
                debug=True
            )
        """

    # Evaluate the model
    Mf1, mf1, accuracy, h_loss, precision, recall, j_index =    \
        evaluation(y_test, y_preds, classification_type=class_type, debug=False)
    
    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index




# ---------------------------------------------------------------------------------------------------------------------
# run_lr_model()
# ---------------------------------------------------------------------------------------------------------------------
def run_lr_model(dataset, X_train, X_test, y_train, y_test, args, target_names, class_type='singlelabel'):

    print("\n\tRunning Logistic Regression model...")

    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
    print("y_train:", type(y_train), y_train.shape)
    print("y_test:", type(y_test), y_test.shape)
    print("Target Names:", target_names)

    # Check if it's a multilabel problem, and use OneVsRestClassifier if true
    if class_type in ['multilabel', 'multi-label']:
        print("Multilabel classification detected. Using OneVsRestClassifier...")
        classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced'))
    else:
        print("Single-label classification detected. Using regular Logistic Regression...")
        classifier = LogisticRegression(max_iter=1000, class_weight='balanced')

    if not args.optimc:
        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        ovr_lr = OneVsRestClassifier(lr)
        ovr_lr.fit(X_train, y_train)
        y_pred_default = ovr_lr.predict(X_test)

        #print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_default, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_default

    # Case with optimization using GridSearchCV
    else:
        print("Optimizing Logistic Regression model with RandomizedSearchCV...")

        param_distributions = {
            'estimator__C': np.logspace(-3, 3, 7),                      # Regularization strength
            'estimator__penalty': ['l1', 'l2'],                         # Regularization method
            'estimator__solver': ['liblinear', 'saga']                  # Solvers compatible with L1 and L2 regularization
        } if class_type == 'multilabel' else {
            'C': np.logspace(-3, 3, 7),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        # Add zero_division=0 to precision and recall to suppress the warnings
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro', zero_division=0),
            'precision_score': make_scorer(precision_score, average='micro', zero_division=0),
            'hamming_loss': make_scorer(hamming_loss),
        }

        # Wrap RandomizedSearchCV around OneVsRestClassifier if multilabel
        randomized_search = RandomizedSearchCV(
            estimator=classifier,
            param_distributions=param_distributions,
            scoring=scorers,
            refit='f1_score',
            n_jobs=-1,
            cv=5,
            return_train_score=True,
            n_iter=10  # Number of parameter settings sampled
        )

        randomized_search.fit(X_train, y_train)
        
        print('Best parameters:', randomized_search.best_params_)
        best_model = randomized_search.best_estimator_
        
        # Predict on test set
        y_pred_best = best_model.predict(X_test)

        print(classification_report(y_true=y_test, y_pred=y_pred_best, target_names=target_names, digits=4, zero_division=0))
        y_preds = y_pred_best

        # Confusion matrix reporting only works in single-label classification
        if (class_type in ['singlelabel', 'single-label']) and (args.cm):
            create_confusion_matrix(
                y_test,
                y_preds,
                target_names,
                title=f'Confusion Matrix for {dataset} with Logistic Regression Model',
                file_name=f'{OUT_DIR}{dataset}_logistic_regression_confusion.png',
                debug=True
            )
        """
        print("Optimizing Logistic Regression model with GridSearchCV...")

        param_grid = {
            'estimator__C': np.logspace(-3, 3, 7),                      # Regularization strength
            'estimator__penalty': ['l1', 'l2'],                         # Regularization method
            'estimator__solver': ['liblinear', 'saga']                  # Solvers compatible with L1 and L2 regularization
        } if class_type == 'multilabel' else {
            'C': np.logspace(-3, 3, 7),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        # Add zero_division=0 to precision and recall to suppress the warnings
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro', zero_division=0),
            'precision_score': make_scorer(precision_score, average='micro', zero_division=0),
            'hamming_loss': make_scorer(hamming_loss),
        }

        # Wrap GridSearchCV around OneVsRestClassifier if multilabel
        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            scoring=scorers,
            refit='f1_score',
            n_jobs=-1,
            cv=5,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)
        
        print('Best parameters:', grid_search.best_params_)
        best_model = grid_search.best_estimator_
        
        # Predict on test set
        y_pred_best = best_model.predict(X_test)

        #print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
        print(classification_report(y_true=y_test, y_pred=y_pred_best, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_best

        # Confusion matrix reporting only works in single-label classification
        if (class_type in ['singlelabel', 'single-label']) and (args.cm):
        
            create_confusion_matrix(
                y_test,
                y_preds,
                target_names,
                title=f'Confusion Matrix for {dataset} with Logistic Regression Model',
                file_name=f'{OUT_DIR}{dataset}_logistic_regression_confusion.png',
                debug=True
            )
        """

    # Evaluate the model
    Mf1, mf1, accuracy, h_loss, precision, recall, j_index =    \
        evaluation(y_test, y_preds, classification_type=class_type, debug=False)
    
    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index









def create_confusion_matrix(y_test, y_pred, category_names, title, file_name=OUT_DIR+'confusion_matrix.png', debug=True):
    """
    Create and display a confusion matrix with actual category names.
    
    Args:
    y_test (array-like): Ground truth (actual labels).
    y_pred (array-like): Predicted labels by the model.
    category_names (list): List of actual category names.
    title (str): Title of the plot.
    file_name (str): File name to save the confusion matrix.
    debug (bool): If True, will print additional information for debugging purposes.
    """

    print("Creating confusion matrix...")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix with category names on the axes
    fig, ax = plt.subplots(figsize=(12, 8))  # Set figure size

    # Display confusion matrix as a heatmap
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=category_names, yticklabels=category_names, ax=ax)

    # Set axis labels and title
    ax.set_xlabel('Predicted Categories', fontsize=14)
    ax.set_ylabel('Actual Categories', fontsize=14)
    plt.title(title, fontsize=16, pad=20)

    # Adjust layout and save the plot to a file
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')  # Save to file
    plt.show()  # Display plot

    print(f"Confusion matrix saved as {file_name}")

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%")

    # Optionally print more detailed information
    if debug:
        print("\nConfusion Matrix Debug Information:")
        print("------------------------------------------------------")
        print("Confusion matrix shows actual classes as rows and predicted classes as columns.")
        print("\nConfusion Matrix Values:")
        for i in range(len(conf_matrix)):
            print(f"Actual category '{category_names[i]}':")
            for j in range(len(conf_matrix[i])):
                print(f"  Predicted as '{category_names[j]}': {conf_matrix[i][j]}")




# -------------------------------------------------------------------------------------------------------------------
# gen_embeddings()
# -------------------------------------------------------------------------------------------------------------------
def gen_embeddings(X_train, y_train, X_test, dataset='bbc-news', pretrained=None, pretrained_vectors_dictionary=None, weighted_embeddings_train=None, 
                   weighted_embeddings_test=None, supervised=False, mix='solo'):
    
    print("\n\tgenerating embeddings...")
        
    print('X_train:', type(X_train), X_train.shape)
    print("y_train:", type(y_train), y_train.shape)
    print('X_test:', type(X_test), X_test.shape)
    
    print("dataset:", dataset)
    
    print("pretrained:", pretrained)
    
    """
    # Check if embedding_vocab_matrix is a dictionary or an ndarray and print accordingly
    if isinstance(pretrained_vectors_dictionary, dict):
        print("pretrained_vectors_dictionary:", type(pretrained_vectors_dictionary), "Length:", len(pretrained_vectors_dictionary))
    elif isinstance(pretrained_vectors_dictionary, np.ndarray):
        print("pretrained_vectors_dictionary:", type(pretrained_vectors_dictionary), "Shape:", pretrained_vectors_dictionary.shape)
    else:
        print("pretrained_vectors_dictionary:", type(pretrained_vectors_dictionary), "Unsupported type")
    """
    
    # should be a numpy array
    print("pretrained_vectors_dictionary:", type(pretrained_vectors_dictionary), "Shape:", pretrained_vectors_dictionary.shape)
    
    
    print("weighted_embeddings_train:", type(weighted_embeddings_train), weighted_embeddings_train.shape)
    print("weighted_embeddings_test:", type(weighted_embeddings_test), weighted_embeddings_test.shape)
    
    print("supervised:", supervised)
    print("mix:", mix)

    # build vocabulary numpy array
    #vocabulary = np.asarray(list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0])
    #print("vocabulary:", type(vocabulary), vocabulary.shape)

    pretrained_embeddings = []                  # List to store pretrained embeddings

    if pretrained in ['word2vec', 'glove', 'fasttext', 'bert', 'llama']:
        
        print("setting up pretrained embeddings...")

        #P = pretrained_vectors_dictionary.extract(vocabulary).numpy()
        #print("pretrained_vectors_dictionary: ", type(pretrained_vectors_dictionary), {pretrained_vectors_dictionary.shape})

        pretrained_embeddings.append(pretrained_vectors_dictionary)
        print(f'pretrained embeddings count after loading pretrained embeddings: {len(pretrained_embeddings[0])}')

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

    if mix == 'solo':
        
        print("Using just the word embeddings alone (solo)...")
        
        #
        # Here we directly return the (tfidf weighted average) embedding 
        # matrix for X_train and X_test
        #
        X_train = weighted_embeddings_train
        X_test = weighted_embeddings_test                   
        
    elif mix == 'cat':        
        print("Concatenating word embeddings with TF-IDF vectors...")
                
        # 
        # Here we concatenate the tfidf vectors and the (tfidf weighted 
        # average) pretrained embedding matrix together
        # 
        X_train = np.hstack([X_train.toarray(), weighted_embeddings_train])
        X_test = np.hstack([X_test.toarray(), weighted_embeddings_test])
    
    elif mix == 'dot':
        
        print("Dot product (matrix multiplication) of embeddings matrix with TF-IDF vectors...")
        
        #
        # here we project the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product 
        #
        """
        X_train = X_train.toarray().dot(embedding_matrix)
        X_test = X_test.toarray().dot(embedding_matrix)
        """
        
        print("before dot product...")
        
        X_train = X_train.toarray()
        X_test = X_test.toarray()
        
        print("X_train:", type(X_train), X_train.shape)
        #print("X_train[0]:\n", X_train[0])
        
        print("X_test:", type(X_test), X_test.shape)
        #print("X_test[0]\n:", X_test[0])
        
        print("pretrained_vectors_dictionary:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
        print("pretrained_vectors_dictionary[0]:\n", pretrained_vectors_dictionary[0])
               
        X_train = np.dot(X_train, pretrained_vectors_dictionary)
        X_test = np.dot(X_test, pretrained_vectors_dictionary)
        
        print("after dot product...")
        
        print("X_train:", type(X_train), X_train.shape)
        #print("X_train[0]:\n", X_train[0])
        
        print("X_test:", type(X_test), X_test.shape)
        #print("X_test[0]:\n", X_test[0])
    
    elif mix == 'vmode':
        # we simply return X_train and X_test as is here
        pass
    
    else:
        print(f"Unsupported mix mode '{mix}'")
        return None    
        
    print("after embedding generation...")
    
    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)

    return X_train, X_test



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# classify_data(): Core processing function
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def classify_data(dataset='20newsgrouops', vtype='tfidf', pretrained_embeddings=None, embedding_path=None, \
    supervised=False, method=None, args=None, logfile=None, system=None):
    """
    Core function for classifying text data using various configurations like embeddings, methods, and models.

    Parameters:
    - dataset (str): The name of the dataset to use (e.g., '20newsgroups', 'ohsumed').
    - vtype (str): The vectorization type (e.g., 'tfidf', 'count').
    - pretrained_embeddings (str or None): Specifies the type of pretrained embeddings to use (e.g., 'bert', 'llama'). If None, no embeddings are used.
    - embedding_path (str): Path to the pretrained embeddings file or directory.
    - supervised (bool): Specifies whether supervised embeddings are used.
    - method (str or None): Specifies the classification method (optional).
    - args: Argument parser object, containing various flags for optimization and configuration (e.g., --optimc).
    - logfile: Logfile object to store results and performance metrics.
    - system: System object containing hardware information like CPU and GPU details.

    Returns:
    None: The function does not return anything, but it prints results and logs them into the specified logfile.

    Workflow:
    - Loads the dataset and the corresponding embeddings.
    - Prepares the input data and target labels for training and testing.
    - Splits the data into train and test sets.
    - Generates embeddings if pretrained embeddings are specified.
    - Calls the classification model (run_model) and logs the evaluation metrics.
    """

    print("\n\tclassifying...")
    
    if (pretrained_embeddings in ['bert', 'llama']):
        embedding_type = 'token'
    else:
        embedding_type = 'word'
    
    print("pretrained_embeddings:", pretrained_embeddings)    
    print("embedding_type:", embedding_type)
    
    lcd = LCDataset()                                    # Create an instance of the LCDataset class
    
    X, y, target_names, class_type, embedding_vocab_matrix, weighted_embeddings = lcd.loadpt(
        dataset=dataset,
        vtype=args.vtype, 
        pretrained=args.pretrained,
        embedding_path=embedding_path,
        emb_type=embedding_type
        )                                                 # Load the dataset
    
    print("Tokenized data loaded.")
 
    print("X:", type(X), X.shape)
    print("y:", type(y), y.shape)
    
    # embedding_vocab_matrix should be a numpy array
    print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)
    print("weighted_embeddings:", type(weighted_embeddings), weighted_embeddings.shape)

    print("transforming labels...")
    if isinstance(y, (csr_matrix, csc_matrix)):
        y = y.toarray()  # Convert sparse matrix to dense array for multi-label tasks
    # Ensure y is in the correct format for classification type
    if class_type in ['singlelabel', 'single-label']:
        y = y.ravel()                       # Flatten y for single-label classification
    print("y after transformation:", type(y), y.shape)

    print("train_test_split...")

    # Perform the train-test split, including the weighted_embeddings
    X_train, X_test, y_train, y_test, weighted_embeddings_train, weighted_embeddings_test = train_test_split(
        X,
        y,
        weighted_embeddings,
        test_size=TEST_SIZE,
        random_state=44,
        shuffle=True 
    )
    
    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
    
    print("y_train:", type(y_train), y_train.shape)
    print("y_test:", type(y_test), y_test.shape)

    #print("y_train:", y_train)
    #print("y_test:", y_test)

    print("weighted_embeddings_train:", type(weighted_embeddings_train), weighted_embeddings_train.shape)
    print("weighted_embeddings_test:", type(weighted_embeddings_test), weighted_embeddings_test.shape)

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
            pretrained=pretrained_embeddings,
            pretrained_vectors_dictionary=embedding_vocab_matrix,
            weighted_embeddings_train=weighted_embeddings_train,
            weighted_embeddings_test=weighted_embeddings_test,
            supervised=args.supervised,
            mix=args.mix
            )
        
        sup_tend = time() - tinit

    Mf1, mf1, acc, h_loss, precision, recall, j_index, tend = run_model(X_train, X_test, y_train, y_test, args, target_names, class_type=class_type)

    tend += sup_tend

    logfile.add_layered_row(tunable=False, measure='final-te-macro-F1', value=Mf1, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='final-te-micro-F1', value=mf1, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='te-accuracy', value=acc, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='te-hamming-loss', value=h_loss, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='te-precision', value=precision, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='te-recall', value=recall, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())
    logfile.add_layered_row(tunable=False, measure='te-jacard-index', value=j_index, timelapse=tend, system_type=system.get_os(), cpus=system.get_cpu_details(), mem=system.get_total_mem(), gpus=system.get_gpu_summary())

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    available_datasets = LCDataset.dataset_available

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification Testing')
    
    parser.add_argument('--dataset', required=True, type=str, default='20newsgroups', metavar='N', help=f'dataset, one in {available_datasets}')
    
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str', help=f'path where to load the pickled dataset from')
    
    parser.add_argument('--log-file', type=str, default='../log/ml_classify.test', metavar='N', help='path to the application log file')
    
    parser.add_argument('--learner', type=str, default='svm', metavar='N', help=f'learner (svm, lr, or nb)')
    
    parser.add_argument('--vtype', type=str, default='tfidf', metavar='N', help=f'dataset base vectorization strategy, in [tfidf, count]')
                        
    parser.add_argument('--mix', type=str, default='solo', metavar='N', help=f'way to prepare the embeddings, in [vmode, solo, cat, dot]. NB presumes --pretrained is set')

    parser.add_argument('--supervised', action='store_true', default=False, help='use supervised embeddings')

    parser.add_argument('--cm', action='store_true', default=False, help=f'create confusion matrix for underlying model')
                             
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


    # check valid dataset
    assert args.dataset in available_datasets, \
        f'unknown dataset {args.dataset}'

    # check valid NB arguments
    if (args.learner in ['nb', 'NB'] and args.mix != 'vmode'):
        print(f'Warning: abandoning run. Naive Bayes model (nb) can only be run with no --pretrained parameter and with --mix vmode (due to lack of support for negative embeddings). Exiting...')
        exit(0)
                
    # initialize log file and run params
    already_modelled, vtype, learner, pretrained, embeddings, emb_path, supervised, mix, method_name, logfile = initialize(args)

    # check to see if model params have been computed already
    if (already_modelled) and not (args.force):
        print(f'Assertion warning: model {method_name} with embeddings {embeddings}, pretrained == {pretrained} and wc_supervised == {args.supervised} for {args.dataset} already calculated.')
        print("Run with --force option to override, returning...")
        exit(0)

    sys = SystemResources()
    print("system resources:", sys)

    classify_data(
        dataset=args.dataset, 
        vtype=vtype,
        pretrained_embeddings=embeddings,
        embedding_path=emb_path,
        supervised=supervised,
        method=method_name,
        args=args, 
        logfile=logfile, 
        system=SystemResources()
        )
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------