
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




# -------------------------------------------------------------------------------------------------------------------
# gen_embeddings()
# -------------------------------------------------------------------------------------------------------------------
def gen_embeddings(X_train, y_train, X_test, dataset='bbc-news', vocab=None, pretrained=None, pretrained_vectors_dictionary=None, weighted_embeddings_train=None, 
                   weighted_embeddings_test=None, supervised=False, mode='solo'):
    
    print("\n\tgenerating embeddings...")
        
    print('X_train:', type(X_train), X_train.shape)
    print("y_train:", type(y_train), y_train.shape)
    print('X_test:', type(X_test), X_test.shape)
    
    print("dataset:", dataset)
    print("vocab:", type(vocab), len(vocab))
    
    print("pretrained:", pretrained)
    print("pretrained_vectors_dictionary:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
    
    print("weighted_embeddings_train:", type(weighted_embeddings_train), weighted_embeddings_train.shape)
    print("weighted_embeddings_test:", type(weighted_embeddings_test), weighted_embeddings_test.shape)
    
    print("supervised:", supervised)
    print("mode:", mode)

    # build vocabulary numpy array
    vocabulary = np.asarray(list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0])
    print("vocabulary:", type(vocabulary), vocabulary.shape)

    pretrained_embeddings = []                  # List to store pretrained embeddings

    if pretrained in ['word2vec', 'glove', 'fasttext', 'bert', 'llama']:
        
        print("setting up pretrained embeddings...")

        #P = pretrained_vectors_dictionary.extract(vocabulary).numpy()
        print("pretrained_vectors_dictionary: ", type(pretrained_vectors_dictionary), {pretrained_vectors_dictionary.shape})

        pretrained_embeddings.append(pretrained_vectors_dictionary)
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
        
        #
        # Here we directly return the (tfidf weighted average) embedding 
        # matrix for X_train and X_test
        #
        X_train = weighted_embeddings_train
        X_test = weighted_embeddings_test                   
        
    elif mode == 'cat':
        
        print("Concatenating word embeddings with TF-IDF vectors...")
        
        # 
        # Here we concatenate the tfidf vectors and the (tfidf weighted 
        # average) pretrained embedding matrix together
        # 
        X_train = np.hstack([X_train.toarray(), weighted_embeddings_train])
        X_test = np.hstack([X_test.toarray(), weighted_embeddings_test])
    
    elif mode == 'dot':
        
        print("Dot product (matrix multiplication) of word embeddings with TF-IDF vectors...")
        
        #
        # here we project the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product 
        #
        X_train = X_train.dot(embedding_matrix)
        X_test = X_test.dot(embedding_matrix)

    print("after embedding generation...")
    
    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)

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

    print(f"Loading data set {dataset}...")

    pickle_file = PICKLE_DIR + pickle_file_name                                     # Define the path to the pickle file

    #
    # we pick up the vectorized dataset along with the associated pretrained 
    # embedding matrices when e load the data - either from data files directly
    # if the first time parsing the dataset or from the pickled file if it exists
    # and the data has been cached for faster loading
    #
    if os.path.exists(pickle_file):                                                 # if the pickle file exists
        
        print(f"Loading tokenized data from '{pickle_file}'...")
        
        X, y, embedding_vocab_matrix, weighted_embeddings, vocab = load_from_pickle(pickle_file)
    else:
        print(f"'{pickle_file}' not found, loading {dataset}...")
        
        X, y, embedding_vocab_matrix, weighted_embeddings, vocab = load_data(
            dataset=dataset,                            # dataset
            pretrained=args.pretrained,                 # pretrained embeddings
            embedding_path=embedding_path               # path to embeddings
            )

        # Save the tokenized matrices to a pickle file
        save_to_pickle(
            X,                          # vectorized data
            y,                          # labels
            embedding_vocab_matrix,     # vector representation of the dataset vocabulary
            weighted_embeddings,        # weighted avg embedding representation of dataset
            vocab,                      # tokenized dataset vocabulary
            pickle_file)         
    
    print("Tokenized data loaded.")
 
    print("X:", type(X), X.shape)
    print("y:", type(y), y.shape)
    
    print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)
    print("weighted_embeddings:", type(weighted_embeddings), weighted_embeddings.shape)
    print("vocab:", type(vocab), len(vocab))

    print("train_test_split...")

    # Perform the train-test split, including the weighted_embeddings
    X_train, X_test, y_train, y_test, weighted_embeddings_train, weighted_embeddings_test = train_test_split(
        X,
        y,
        weighted_embeddings,
        test_size=0.25,
        random_state=44,
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

    print("weighted_embeddings_train:", type(weighted_embeddings_train), weighted_embeddings_train.shape)
    print("weighted_embeddings_test:", type(weighted_embeddings_test), weighted_embeddings_test.shape)
    
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
            pretrained_vectors_dictionary=embedding_vocab_matrix,
            weighted_embeddings_train=weighted_embeddings_train,
            weighted_embeddings_test=weighted_embeddings_test,
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