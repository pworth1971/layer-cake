import numpy as np
import argparse
from time import time
import os

from scipy.sparse import csr_matrix, csc_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from embedding.supervised import get_supervised_embeddings

from util.common import SystemResources, NEURAL_MODELS, ML_MODELS
from util.common import SUPPORTED_LMS, SUPPORTED_TRANSFORMER_LMS
from util.common import VECTOR_CACHE, PICKLE_DIR, DATASET_DIR
from util.common import WORD_BASED_MODELS, TOKEN_BASED_MODELS


from data.lc_dataset import LCDataset, save_to_pickle, load_from_pickle

from model.classification import ml_classification

from util.csv_log import CSVLog

from model.LCRepresentationModel import FASTTEXT_MODEL, GLOVE_MODEL, WORD2VEC_MODEL
from model.LCRepresentationModel import BERT_MODEL, ROBERTA_MODEL, XLNET_MODEL, GPT2_MODEL


import warnings
warnings.filterwarnings('ignore')



# -------------------------------------------------------------------------------------------------------------------
# gen_model()
# -------------------------------------------------------------------------------------------------------------------
def gen_xdata(Xtr_raw, Xtr_vectorized, y_train, Xte_raw, Xte_vectorized, y_test, dataset='bbc-news', 
                   pretrained=None, pretrained_vectors_dictionary=None, 
                   weighted_embeddings_train=None, weighted_embeddings_test=None, 
                   avg_embeddings_train=None, avg_embeddings_test=None, 
                   summary_embeddings_train=None, summary_embeddings_test=None, 
                   dataset_embedding_type='weighted', mix='solo', supervised=False):
    """
    Generates the model input data, training and test data sets, based on the provided word embeddings and 
    TF-IDF (or Count) vectors and model options.

    Parameters:
    -----------
    Xtr_raw : numpy.ndarray
        The raw text data for the training set.
    Xtr_vectorized : scipy.sparse matrix
        The vectorized TF-IDF representation of the training set.
    y_train : numpy.ndarray
        The target labels for the training set.
    Xte_raw : numpy.ndarray
        The raw text data for the test set.
    Xte_vectorized : scipy.sparse matrix
        The vectorized TF-IDF representation of the test set.
    y_test : numpy.ndarray
        The target labels for the test set.
    dataset : str, optional
        The name of the dataset being used (default is 'bbc-news').
    pretrained : bool, optional
        Indicates whether to use pre-trained embeddings (default is None).
    pretrained_vectors_dictionary : numpy.ndarray, optional
        A dictionary or array of pre-trained word embeddings (default is None).
    weighted_embeddings_train : numpy.ndarray, optional
        Weighted word embeddings for the training set (default is None).
    weighted_embeddings_test : numpy.ndarray, optional
        Weighted word embeddings for the test set (default is None).
    avg_embeddings_train : numpy.ndarray, optional
        Average word embeddings for the training set (default is None).
    avg_embeddings_test : numpy.ndarray, optional
        Average word embeddings for the test set (default is None).
    summary_embeddings_train : numpy.ndarray, optional
        Summary (e.g., CLS token) word embeddings for the training set (default is None).
    summary_embeddings_test : numpy.ndarray, optional
        Summary word embeddings for the test set (default is None).
    dataset_embedding_type : str, optional
        The type of embedding to use for the dataset ('weighted', 'avg', 'summary') (default is 'weighted').
    mix : str, optional
        How to combine embeddings with TF-IDF vectors. Options are 'solo', 'cat', 'dot', 'lsa', or 'vmode' (default is 'solo').
    supervised : bool, optional
        Whether the process is supervised or not (default is False).

    Returns:
    --------
    X_train : numpy.ndarray or scipy.sparse matrix
        The generated embedding representation for the training set.
    X_test : numpy.ndarray or scipy.sparse matrix
        The generated embedding representation for the test set.
    """

    print("\n\tgenerating model X data (features)...")

    print("dataset:", dataset)    
    print("pretrained:", pretrained)
    print("dataset_embedding_type:", dataset_embedding_type)
    print("supervised:", supervised)    
    print("mix:", mix)
         
    #print('Xtr_raw:', type(Xtr_raw), Xtr_raw.shape)
    print("Xtr_vectorized:", type(Xtr_vectorized), Xtr_vectorized.shape)
    print("y_train:", type(y_train), y_train.shape)
    #print("Xte_raw:", type(Xte_raw), Xte_raw.shape)
    print("Xte_vectorized:", type(Xte_vectorized), Xte_vectorized.shape)
    print('y_test:', type(y_test), y_test.shape)
 
    print("pretrained_vectors_dictionary:", type(pretrained_vectors_dictionary), "Shape:", pretrained_vectors_dictionary.shape)    
    print("weighted_embeddings_train:", type(weighted_embeddings_train), weighted_embeddings_train.shape)
    print("weighted_embeddings_test:", type(weighted_embeddings_test), weighted_embeddings_test.shape)
    print("avg_embeddings_train:", type(avg_embeddings_train), avg_embeddings_train.shape)
    print("avg_embeddings_test:", type(avg_embeddings_test), avg_embeddings_test.shape)
    print("summary_embeddings_train:", type(summary_embeddings_train), summary_embeddings_train.shape)
    print("summary_embeddings_test:", type(summary_embeddings_test), summary_embeddings_test.shape)
        
        
    # here we are just using the embedding data representation of
    # the doc dataset data by itself, i.e. without the tfidf vectors    
    if mix == 'solo':
        
        print("Using just the embedding dataset representation alone (solo)...")
        
        #
        # Here we directly return the dataset embedding representation 
        # form as specified with X_train and X_test
        #
        if (dataset_embedding_type == 'weighted'):
            X_train = weighted_embeddings_train
            X_test = weighted_embeddings_test                
        elif (dataset_embedding_type == 'avg'):
            X_train = avg_embeddings_train
            X_test = avg_embeddings_test
        elif (dataset_embedding_type == 'summary'):

            # NB that summary type embeddings (i.e. CLS token) only supported with BERT embeddings
            # but we expect them in the pickle file so we swapped them for avg_embeddings when 
            # we compute them in LCDataset class 
            X_train = summary_embeddings_train
            X_test = summary_embeddings_test
        else:
            print(f"Unsupported dataset_embedding_type '{dataset_embedding_type}'")
            return None   

    # here we are just using the embedding data representation of
    # the doc dataset data by itself, i.e. without the tfidf vectors    
    elif mix == 'solo-wce':
        
        print("using the embedding dataset represntation with WCEs (solo-wce)...")

        #
        # Here we initialize the training and test data sets with the
        # appropriate dataset embedding representation 
        #
        if (dataset_embedding_type == 'weighted'):
            X_train = weighted_embeddings_train
            X_test = weighted_embeddings_test                
        elif (dataset_embedding_type == 'avg'):
            X_train = avg_embeddings_train
            X_test = avg_embeddings_test
        elif (dataset_embedding_type == 'summary'):

            # NB that summary type embeddings (i.e. CLS token) only supported with BERT embeddings
            # but we expect them in the pickle file so we swapped them for avg_embeddings when 
            # we compute them in LCDataset class 
            X_train = summary_embeddings_train
            X_test = summary_embeddings_test
        else:
            print(f"Unsupported dataset_embedding_type '{dataset_embedding_type}'")
            return None   

        #
        # Next webuild the WCE representation by projecting 
        # the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product. 
        #
        Xtr_wces = get_supervised_embeddings(Xtr_vectorized, y_train)
        print("Xtr_wces:", type(Xtr_wces), Xtr_wces.shape)
        Xte_wces = get_supervised_embeddings(Xte_vectorized, y_test)
        print("Xte_wces:", type(Xte_wces), Xte_wces.shape)
        
        print("stacking embedding_vocab_matrix with WCEs...")
        pretrained_vectors_dictionary = np.hstack([pretrained_vectors_dictionary, Xtr_wces])

        print("embedding dataset vocab representation after WCES:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
    
        X_train_wce = Xtr_vectorized @ pretrained_vectors_dictionary
        X_test_wce = Xte_vectorized @ pretrained_vectors_dictionary
        
        print("after dot product...")
        print("X_train_wce:", type(X_train_wce), X_train_wce.shape)
        print("X_test_wce:", type(X_test_wce), X_test_wce.shape)

        # Now we stack the WCEs (from the vectorized representation) with the 
        # dataset embedding representation from the LCRepresentationModel.encode_docs() 
        # method, ie the embedidng representation of the document dataset
        X_train = np.hstack([X_train, X_train_wce])
        X_test = np.hstack([X_test, X_test_wce])

        print("after stacking WCEs with embedding dataset representation...")
        print("X_train:", type(X_train), X_train.shape)
        print("X_test:", type(X_test), X_test.shape)

    # here we are concatenating the vectorized representation of the dataset 
    # along with the embedding representation of the dataset
    elif mix == 'cat-doc':

        print("Concatenating the vectorized text representation with embedding doc representation (cat-doc)...")
                
        # 
        # Here we concatenate the tfidf vectors and the specified form 
        # of dataset embedding representation to form the final input.
        # NB: we maintain the sparse representations here for faster compute performance
        #
        from scipy.sparse import hstack

        if (dataset_embedding_type == 'weighted'):
            X_train = hstack([Xtr_vectorized, csr_matrix(weighted_embeddings_train)])
            X_test = hstack([Xte_vectorized, csr_matrix(weighted_embeddings_test)])
        elif (dataset_embedding_type == 'avg'):
            X_train = hstack([Xtr_vectorized, csr_matrix(avg_embeddings_train)])
            X_test = hstack([Xte_vectorized, csr_matrix(avg_embeddings_test)])
        elif (dataset_embedding_type == 'summary'):
            X_train = hstack([Xtr_vectorized, csr_matrix(summary_embeddings_train)])
            X_test = hstack([Xte_vectorized, csr_matrix(summary_embeddings_test)])
        else:
            print(f"Unsupported dataset_embedding_type '{dataset_embedding_type}'")
            return None
    
    # here we are concatenating the vectorized representation of the dataset
    # along with the WCEs only
    elif mix == 'cat-wce':
        
        print("Concatenting WCEs with vectorized dataset representation (cat-wce)...")
        
        #
        # here we project the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product. NB we maintain the sparse 
        # representations here for faster compute performance
        #
        Xtr_wces = get_supervised_embeddings(Xtr_vectorized, y_train)
        print("Xtr_wces:", type(Xtr_wces), Xtr_wces.shape)
        Xte_wces = get_supervised_embeddings(Xte_vectorized, y_test)
        print("Xte_wces:", type(Xte_wces), Xte_wces.shape)
        
        print("stacking embedding_vocab_matrix with WCEs...")
        pretrained_vectors_dictionary = np.hstack([pretrained_vectors_dictionary, Xtr_wces])

        print("embedding dataset vocab representation after WCES:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
    
        X_train_wce = Xtr_vectorized @ pretrained_vectors_dictionary
        X_test_wce = Xte_vectorized @ pretrained_vectors_dictionary
        
        print("after dot product...")
        print("X_train_wce:", type(X_train_wce), X_train_wce.shape)
        print("X_test_wce:", type(X_test_wce), X_test_wce.shape)
        
        # After applying the WCEs, we concatenate the resulting matrix with the 
        # original feature matrix to augment the vectorized feature set with the 
        # embedding space for said vectorized feature space
        X_train = np.hstack([Xtr_vectorized.toarray(), X_train_wce])              
        X_test = np.hstack([Xte_vectorized.toarray(), X_test_wce])

    # here we are concatenating the vectorized representation of the dataset 
    # with the embedding representation of the dataset AND the WCEs
    elif mix == 'cat-doc-wce':

        print("Concatenating the vectorized text representation with embedding doc representation and the WCEs (cat-doc-wce)...")
                
        # 
        # Here we concatenate the tfidf vectors and the specified form 
        # of dataset embedding representation to form the final input.
        # NB: we maintain the sparse representations here for faster compute performance
        #
        from scipy.sparse import hstack

        if (dataset_embedding_type == 'weighted'):
            X_train = hstack([Xtr_vectorized, csr_matrix(weighted_embeddings_train)])
            X_test = hstack([Xte_vectorized, csr_matrix(weighted_embeddings_test)])
        elif (dataset_embedding_type == 'avg'):
            X_train = hstack([Xtr_vectorized, csr_matrix(avg_embeddings_train)])
            X_test = hstack([Xte_vectorized, csr_matrix(avg_embeddings_test)])
        elif (dataset_embedding_type == 'summary'):
            X_train = hstack([Xtr_vectorized, csr_matrix(summary_embeddings_train)])
            X_test = hstack([Xte_vectorized, csr_matrix(summary_embeddings_test)])
        else:
            print(f"Unsupported dataset_embedding_type '{dataset_embedding_type}'")
            return None
        
        print("after stacking vectorized represnetation and embedding representations of dataset...")
        print("X_train:", type(X_train), X_train.shape)
        print("X_test:", type(X_test), X_test.shape)

        #
        # here we project the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product. NB we maintain the sparse 
        # representations here for faster compute performance
        #
        Xtr_wces = get_supervised_embeddings(Xtr_vectorized, y_train)
        Xte_wces = get_supervised_embeddings(Xte_vectorized, y_test)
        print("Xtr_wces:", type(Xtr_wces), Xtr_wces.shape)
        print("Xte_wces:", type(Xte_wces), Xte_wces.shape)

        print("stacking embedding_vocab_matrix with WCEs...")
        pretrained_vectors_dictionary = np.hstack([pretrained_vectors_dictionary, Xtr_wces])

        print("embedding dataset vocab representation after WCES:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
    
        X_train_wce = Xtr_vectorized @ pretrained_vectors_dictionary
        X_test_wce = Xte_vectorized @ pretrained_vectors_dictionary

        print("after dot product...")
        print("X_train_wce:", type(X_train_wce), X_train_wce.shape)
        print("X_test_wce:", type(X_test_wce), X_test_wce.shape)
        
        # After computing the WCEs and the related vectorizxed + embedding+WCE representation, 
        # we concatenate the resulting matrix with the stacked vectorized + embedding dataset 
        # representation to further augment the already augmented vectorized feature set with 
        # the additional WCE information
        X_train = np.hstack([X_train.toarray(), X_train_wce])              
        X_test = np.hstack([X_test.toarray(), X_test_wce])

        print("after stacking WCEs with vectorized represnetation and embedding representations of dataset...")
        print("X_train:", type(X_train), X_train.shape)
        print("X_test:", type(X_test), X_test.shape)
        

    # here we are projecting the vectorized representation of the dataset (either tfidf or count)
    # into the pretrained embedding (vocabulary) space using the dot product (matrix multiplication)
    elif mix == 'dot':
        
        print("projecting vectorized representation into embedding space (dot)...")
                
        X_train = Xtr_vectorized @ pretrained_vectors_dictionary
        X_test = Xte_vectorized @ pretrained_vectors_dictionary
        
        print("after dot product...")
        print("X_train:", type(X_train), X_train.shape)
        #print("X_train[0]:\n", X_train[0])
        print("X_test:", type(X_test), X_test.shape)
        #print("X_test[0]:\n", X_test[0])
    
    # here we are projecting the vectorized representation of the dataset (either tfidf or count)
    # into the pretrained embedding (vocabulary) space using the dot product (matrix multiplication)
    elif mix == 'dot-wce':
        
        print("projecting vectorized representation into embedding space with WCEs (dot-wce)...")
        
        #
        # here we project the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product.
        # 
        # NB we maintain the sparse representations here for faster compute performance
        #
        wces = get_supervised_embeddings(Xtr_vectorized, y_train)
        print("WCEs:", type(wces), wces.shape)
        
        print("stacking embedding_vocab_matrix with WCEs...")
        pretrained_vectors_dictionary = np.hstack([pretrained_vectors_dictionary, wces])

        print("embedding dataset vocab representation after WCES:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
    
        X_train = Xtr_vectorized @ pretrained_vectors_dictionary
        X_test = Xte_vectorized @ pretrained_vectors_dictionary
        
        print("after dot product...")
        print("X_train:", type(X_train), X_train.shape)
        #print("X_train[0]:\n", X_train[0])
        print("X_test:", type(X_test), X_test.shape)
        #print("X_test[0]:\n", X_test[0])

    # here we are using LSA (Latent Semantic Analysis) to reduce the number of features, using the 
    # embedding dimension size as the input to the TruncatedSVD sklearn call
    elif mix == 'lsa':
        
        print("using SVD (LSA) to reduce feature set (lsa)...")

        n_dimensions = pretrained_vectors_dictionary.shape[1]
        print("n_dimensions:", n_dimensions)
        
        svd = TruncatedSVD(n_components=n_dimensions)           # Adjust dimensions to match pretrained embeddings
        
        # reset X_train and X_test to the original tfidf vectors
        # using the svd model
        X_train = svd.fit_transform(Xtr_vectorized)
        X_test = svd.transform(Xte_vectorized)

    # here we use LSA (Latent Semantic Analysis) to reduce the number of features, using the 
    # embedding dimension size as the input to the TruncatedSVD sklearn call, and then we augment
    # the reduced feature set with the WCEs
    elif mix == 'lsa-wce':
        
        print("using SVD (LSA) with WCEs (lsa-wce)...")

        n_dimensions = pretrained_vectors_dictionary.shape[1]
        print("n_dimensions:", n_dimensions)
        
        svd = TruncatedSVD(n_components=n_dimensions)           # Adjust dimensions to match pretrained embeddings
        
        # reset X_train and X_test to the original tfidf vectors
        # using the svd model
        X_train = svd.fit_transform(Xtr_vectorized)
        X_test = svd.transform(Xte_vectorized)

         #
        # here we project the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product. NB we maintain the sparse 
        # representations here for faster compute performance
        #
        Xtr_wces = get_supervised_embeddings(Xtr_vectorized, y_train)
        print("Xtr_wces:", type(Xtr_wces), Xtr_wces.shape)
        Xte_wces = get_supervised_embeddings(Xte_vectorized, y_test)
        print("Xte_wces:", type(Xte_wces), Xte_wces.shape)
        
        print("stacking embedding_vocab_matrix with WCEs...")
        pretrained_vectors_dictionary = np.hstack([pretrained_vectors_dictionary, Xtr_wces])

        print("embedding dataset vocab representation after WCES:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
    
        X_train_wce = Xtr_vectorized @ pretrained_vectors_dictionary
        X_test_wce = Xte_vectorized @ pretrained_vectors_dictionary
        
        print("after dot product...")
        print("X_train_wce:", type(X_train_wce), X_train_wce.shape)
        print("X_test_wce:", type(X_test_wce), X_test_wce.shape)
        
        # After applying the WCEs, we concatenate the resulting matrix with the 
        # LSA feature reduced vectiorized representation of the dataset to augment 
        # the LSA features with the WCEs
        X_train = np.hstack([X_train, X_train_wce])              
        X_test = np.hstack([X_test, X_test_wce])

    # here we are using the vectorized representation of the dataset (either tfidf or count) as the
    # input to the model directly
    elif mix == 'vmode':
        X_train = Xtr_vectorized
        X_test = Xte_vectorized

    else:
        print(f"Unsupported mix mode '{mix}'")
        return None    
        
    print("after input data matrix generation...")
    
    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)

    return X_train, X_test


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# run_model(): Core processing function
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_model(dataset='20newsgroups', vtype='tfidf', embeddings=None, embedding_path=None, representation=None, lang_model_type=None, optimized=False, logfile=None, args=None):
    """
    Core function for classifying text data using various configurations like embeddings, methods, and models.

    Parameters:
    - dataset (str): The name of the dataset to use (e.g., '20newsgroups', 'ohsumed').
    - vtype (str): The vectorization type (e.g., 'tfidf', 'count').
    - embeddings (str or None): Specifies the type of pretrained embeddings to use (e.g., 'bert', 'llama'). If None, no embeddings are used.
    - embedding_path (str): Path to the pretrained embeddings file or directory.
    - representation (str or None): Specifies the classification method (optional).
    - lang_model_type (str or None): Specifies the type of language model being used (tied to embeddings).
    - optimized (bool): Whether the model is optimized for performance. 
    - logfile: Logfile object to store results and performance metrics.
    - args: Argument parser object, containing various flags for optimization and configuration (e.g., --optimc).

    Returns:
    - acc: Accuracy of the model.
    - Mf1: Macro F1 score of the model.
    - mf1: Micro F1 score of the model.
    - h_loss: Hamming loss of the model.
    - precision: Precision of the model.
    - recall: Recall of the model.
    - j_index: Jaccard index of the model.
    - tend: Time taken to run the model.

    Workflow:
    - Loads the dataset and the corresponding embeddings.
    - Prepares the input data and target labels for training and testing.
    - Splits the data into train and test sets.
    - Generates embeddings if pretrained embeddings are specified.
    - Calls the classification model (run_model) and logs the evaluation metrics.
    """

    print("\n\trunning model...")

    print("representation:", representation)
    print("optimize:", optimized)

    if (args.pretrained is not None and args.pretrained in ['bert', 'roberta', 'llama', 'xlnet', 'gpt2']):
        embedding_type = 'token'
    else:
        embedding_type = 'word'
    
    print("embeddings:", embeddings)    
    print("embedding_path:", embedding_path)
    print("embedding_type:", embedding_type)
    
    print("lang_model_type:", lang_model_type)

    #
    # Load the dataset and the associated (pretrained) embedding structures
    # to be fed into the model
    #                                                          
    Xtr_raw, Xte_raw, Xtr_vectorized, Xte_vectorized, y_train_sparse, y_test_sparse, target_names, class_type, embedding_vocab_matrix, Xtr_weighted_embeddings, \
        Xte_weighted_embeddings, Xtr_avg_embeddings, Xte_avg_embeddings, Xtr_summary_embeddings, \
            Xte_summary_embeddings = loadpt_data(
                                            dataset=dataset,                                # Dataset name
                                            vtype=args.vtype,                               # Vectorization type
                                            pretrained=args.pretrained,                     # pretrained embeddings type
                                            embedding_path=embedding_path,                  # path to pretrained embeddings
                                            emb_type=embedding_type                         # embedding type (word or token)
                                            )                                                
    
    print("--- LCDataset loaded ---")
 
    #print("Xtr_raw:", type(Xtr_raw), Xtr_raw.shape)
    #print("Xte_raw:", type(Xte_raw), Xte_raw.shape)

    print("Xtr_vectorized:", type(Xtr_vectorized), Xtr_vectorized.shape)
    print("Xte_vectorized:", type(Xte_vectorized), Xte_vectorized.shape)

    print("y_train_sparse:", type(y_train_sparse), y_train_sparse.shape)
    print("y_test_sparse:", type(y_test_sparse), y_test_sparse.shape)
    
    print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

    print("Xtr_weighted_embeddings:", type(Xtr_weighted_embeddings), Xtr_weighted_embeddings.shape)
    print("Xte_weighted_embeddings:", type(Xte_weighted_embeddings), Xte_weighted_embeddings.shape)
    
    print("Xtr_avg_embeddings:", type(Xtr_avg_embeddings), Xtr_avg_embeddings.shape)
    print("Xte_avg_embeddings:", type(Xte_avg_embeddings), Xte_avg_embeddings.shape)
    
    print("Xtr_summary_embeddings:", type(Xtr_summary_embeddings), Xtr_summary_embeddings.shape)
    print("Xte_summary_embeddings:", type(Xte_summary_embeddings), Xte_summary_embeddings.shape)

    if class_type in ['multilabel', 'multi-label']:

        print("multi-label case, expanding (todense) y...")

        if isinstance(y_train_sparse, (csr_matrix, csc_matrix)):
            y_train = y_train_sparse.toarray()                      # Convert sparse matrix to dense array for multi-label tasks
        if isinstance(y_test_sparse, (csr_matrix, csc_matrix)):
            y_test = y_test_sparse.toarray()                        # Convert sparse matrix to dense array for multi-label tasks
        
        print("y_train after transformation:", type(y_train), y_train.shape)
        print("y_test after transformation:", type(y_test), y_test.shape)
    else:
        y_train = y_train_sparse
        y_test = y_test_sparse

    """
    # Ensure y is in the correct format for classification type
    if class_type in ['singlelabel', 'single-label']:
        y_train = y_train.ravel()                       # Flatten y for single-label classification
        y_test = y_test.ravel()                         # Flatten y for single-label classification
    
    print("y_train after transformation:", type(y_train), y_train.shape)
    print("y_test after transformation:", type(y_test), y_test.shape)
    """

    if args.pretrained is None:        # no embeddings in this case
        sup_tend = 0
    else:                              # embeddings are present
        tinit = time()

        # Generate embeddings
        X_train, X_test = gen_xdata(
            Xtr_raw=Xtr_raw, 
            Xtr_vectorized=Xtr_vectorized, 
            y_train=y_train, 
            Xte_raw=Xte_raw,
            Xte_vectorized=Xte_vectorized, 
            y_test=y_test, 
            dataset=dataset,
            pretrained=embeddings,
            pretrained_vectors_dictionary=embedding_vocab_matrix,
            weighted_embeddings_train=Xtr_weighted_embeddings,
            weighted_embeddings_test=Xte_weighted_embeddings,
            avg_embeddings_train=Xtr_avg_embeddings,
            avg_embeddings_test=Xte_avg_embeddings,
            summary_embeddings_train=Xtr_summary_embeddings,
            summary_embeddings_test=Xte_summary_embeddings,
            dataset_embedding_type=args.dataset_emb_comp,
            mix=args.mix
            )

        sup_tend = time() - tinit

    Mf1, mf1, acc, h_loss, precision, recall, j_index, tend = ml_classification(
        X_train,                                        # training data
        X_test,                                         # test data
        y_train,                                        # training labels
        y_test,                                         # test labels
        args,                                           # arguments
        target_names,                                   # target names
        class_type=class_type                           # class type
    )

    tend += sup_tend

    dims = X_train.shape[1]
    print("# dimensions:", dims)

    comp_method = get_model_computation_method(args, embedding_type)
    print("comp_method:", comp_method)

    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='final-te-macro-F1', value=Mf1, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='final-te-micro-F1', value=mf1, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-accuracy', value=acc, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-precision', value=precision, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-recall', value=recall, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-jacard-index', value=j_index, timelapse=tend)

    return acc, Mf1, mf1, h_loss, precision, recall, j_index, tend

    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_model_computation_method(args, embedding_type='word'):

    print("calculating model computation method...")
    print("embedding_type:", embedding_type)

    if (args.pretrained in ['bert', 'roberta', 'llama', 'xlnet', 'gpt2']):
        pt_type = 'attention:tokenized'
    elif (args.pretrained in ['glove', 'word2vec']):
        pt_type = 'co-occurrence:word'
    elif (args.pretrained in ['fasttext']):
        pt_type = 'co-occurrence:subword'
    else:
        pt_type = 'unkwnown'
        
    if args.pretrained == 'fasttext':
        embedding_type = 'subword'
        
    pt_type = f'{embedding_type}:{pt_type}'

    if (args.learner in ML_MODELS): 

        if args.mix in ['solo', 'solo-wce']:
            return pt_type
            
        elif args.mix == 'vmode':
            return f'frequency:{args.vtype}'

        elif args.mix in ['cat-doc', 'cat-wce', 'cat-doc-wce']:
            return f'frequency:{args.vtype}+{pt_type}'

        elif args.mix in ['dot', 'dot-wce']:
            return f'frequency:{args.vtype}.{pt_type}'

        elif args.mix in ['lsa', 'lsa-wce']:
            return f'frequency:{args.vtype}->SVD'
       
    elif (args.learner in NEURAL_MODELS):
        pass



def loadpt_data(dataset, vtype='tfidf', pretrained=None, embedding_path=VECTOR_CACHE, emb_type='word', embedding_comp_type='avg'):

    print("loadpt_data():", dataset, PICKLE_DIR)

    print("pretrained:", pretrained)
    
    #
    # load the dataset using appropriate tokenization method as dictated by pretrained embeddings
    #
    if (pretrained == 'glove'):
        model_name = GLOVE_MODEL
    elif(pretrained == 'word2vec'):
        model_name = WORD2VEC_MODEL
    elif(pretrained == 'fasttext'):
        model_name = FASTTEXT_MODEL
    elif(pretrained == 'bert'):
        model_name = BERT_MODEL
    elif (pretrained == 'roberta'):
        model_name = ROBERTA_MODEL
    elif (pretrained == 'gpt2'):
        model_name = GPT2_MODEL
    elif (pretrained == 'xlnet'):
        model_name = XLNET_MODEL
    else:
        model = None
        raise ValueError(f"Unsupported pretrained model '{pretrained}'")
        
    print("model_name:", model_name)
    
    pickle_file_name=f'{dataset}_{vtype}_{pretrained}_{model_name}.pickle'
    
    # remove '/' from the file name
    pickle_file_name = pickle_file_name.replace("/", "_")               # Replace forward slash with an underscore

    pickle_file = PICKLE_DIR + pickle_file_name                                     
    print("pickle_file:", pickle_file)

    #
    # we pick up the vectorized dataset along with the associated pretrained 
    # embedding matrices when e load the data - either from data files directly
    # if the first time parsing the dataset or from the pickled file if it exists
    # and the data has been cached for faster loading
    #
    if os.path.exists(pickle_file):                                                 # if the pickle file exists
        
        print(f"Loading tokenized data from '{pickle_file}'...")
        
        Xtr_raw, Xte_raw, Xtr_vectorized, Xte_vectorized, y_train_sparse, y_test_sparse, target_names, class_type, embedding_vocab_matrix, Xtr_weighted_embeddings, \
            Xte_weighted_embeddings, Xtr_avg_embeddings, Xte_avg_embeddings, Xtr_summary_embeddings, Xte_summary_embeddings = load_from_pickle(pickle_file)

        return Xtr_raw, Xte_raw, Xtr_vectorized, Xte_vectorized, y_train_sparse, y_test_sparse, target_names, class_type, embedding_vocab_matrix, Xtr_weighted_embeddings, \
            Xte_weighted_embeddings, Xtr_avg_embeddings, Xte_avg_embeddings, Xtr_summary_embeddings, Xte_summary_embeddings

    else:
        print(f"'{pickle_file}' not found, loading {dataset}...")
        
        lcd = LCDataset(
            name=dataset,                               # dataset name 
            vectorization_type=vtype,                   # vectorization type (one of 'tfidf', 'count')
            embedding_type=emb_type,                    # embedding type (one of 'word', 'token')
            pretrained=pretrained,                      # pretrained embeddings (model type or None)
            embedding_path=embedding_path,              # path to embeddings
            embedding_comp_type=embedding_comp_type     # embedding computation type (one of 'avg', 'weighted', 'summary')
        )    

        lcd.vectorize()                             # vectorize the dataset

        lcd.init_embedding_matrices()               # initialize the embedding matrices

        # Save the tokenized matrices to a pickle file
        save_to_pickle(
            lcd.Xtr,                                    # raw training data (not vectorized)
            lcd.Xte,                                    # raw test data (not vectorized)
            lcd.Xtr_vectorized,                         # vectorized training data
            lcd.Xte_vectorized,                         # vectorized test data
            lcd.y_train_sparse,                         # training data labels
            lcd.y_test_sparse,                          # test data labels
            lcd.target_names,                           # target names
            lcd.class_type,                             # class type (single-label or multi-label):
            lcd.embedding_vocab_matrix,                 # vector representation of the dataset vocabulary
            lcd.Xtr_weighted_embeddings,                # weighted avg embedding representation of dataset training data
            lcd.Xte_weighted_embeddings,                # weighted avg embedding representation of dataset test data
            lcd.Xtr_avg_embeddings,                     # avg embedding representation of dataset training data
            lcd.Xte_avg_embeddings,                     # avg embedding representation of dataset test data
            lcd.Xtr_summary_embeddings,                 # summary embedding representation of dataset training data
            lcd.Xte_summary_embeddings,                 # summary embedding representation of dataset test data
            pickle_file)         

        return lcd.Xtr, lcd.Xte, lcd.Xtr_vectorized, lcd.Xte_vectorized, lcd.y_train_sparse, lcd.y_test_sparse, lcd.target_names, lcd.class_type, lcd.embedding_vocab_matrix, \
            lcd.Xtr_weighted_embeddings, lcd.Xte_weighted_embeddings, lcd.Xtr_avg_embeddings, lcd.Xte_avg_embeddings, lcd.Xtr_summary_embeddings, lcd.Xte_summary_embeddings


# -------------------------------------------------------------------------------------------------------------------------------------------------
def initialize_ml_testing(args):

    """
    Initializes machine learning testing based on the provided arguments.

    Args:
    - args: A namespace or dictionary of arguments that specify the configuration for the ML experiment.
      Expected fields:
        - learner (str): Type of learner to use ('svm', 'lr', or 'nb').
        - vtype (str): Vectorization type, either 'count' or 'tfidf'.
        - pretrained (str): Pretrained model or embedding type (e.g., 'BERT', 'LLaMA').
        - dataset (str): Name of the dataset.
        - logfile (str): Path to the log file where results will be stored.
        - mix (str): Dataset and embedding comparison method.
        - dataset_emb_comp (str): Dataset embedding comparison method.
    
    Returns:
    - already_computed (bool): Whether the current configuration has already been computed.
    - vtype (str): Vectorization type ('count' or 'tfidf').
    - learner (class): The ML model class to be used (e.g., `LinearSVC` for SVM).
    - pretrained (bool): Whether to use a pretrained model or embeddings.
    - embeddings (str): Type of embeddings to use.
    - emb_path (str): Path to the embeddings or pretrained model files.
    - mix (str): The dataset and embedding comparison method.
    - representation (str): Type of data representation used for training.
    - ml_logger (CSVLog): Logger object to store model run details.
    - optimized (bool): Whether the model is optimized for performance.
    """

    print("\n\tinitializing ML testing...")
    
    print("args:", args)

    # set up model type
    if args.learner == 'svm':
        learner = LinearSVC
        learner_name = 'SVM' 
    elif args.learner == 'lr':
        learner = LogisticRegression
        learner_name = 'LR'
    elif args.learner == 'nb':
        learner = MultinomialNB
        #learner = GaussianNB
        learner_name = 'NB'
    else:
        print("** Unknown learner, possible values are svm, lr or nb **")
        return

    print("learner:", learner)
    print("learner_name: ", {learner_name})

    # default to tfidf vectorization type unless 'count' specified explicitly
    if args.vtype == 'count':
        vtype = 'count'
    else:
        vtype = 'tfidf'             
        
    print("vtype:", {vtype})

    pretrained = False
    embeddings ='none'
    emb_path = VECTOR_CACHE

    if (args.pretrained):
        pretrained = True

    # get the path to the embeddings
    emb_path = get_embeddings_path(args.pretrained, args)
    print("emb_path: ", {emb_path})

    model_type = f'{learner_name}:{args.vtype}-{args.mix}'
    print("model_type:", {model_type})
    
    print("initializing baseline layered log file...")

    ml_logger = CSVLog(
        file=args.logfile, 
        columns=[
            'os',
            'cpus',
            'mem',
            'gpus',
            'dataset', 
            'class_type',
            'model', 
            'embeddings',
            'lm_type',
            'mode',
            'comp_method',
            'representation',
            'optimized',
            'dimensions',
            'measure', 
            'value',
            'timelapse',
            'epoch',
            'run',
            ], 
        verbose=True, 
        overwrite=False)

    print("setting defaults...")
    print("embeddings:", embeddings)

    print("pretrained: ", {pretrained}, "; embeddings: ", {embeddings})
    
    #ml_logger.set_default('dataset', args.dataset)
    #ml_logger.set_default('model', learner_name)                 # method in the log file

    system = SystemResources()
    print("system:\n", system)

    run_mode = args.dataset + ':' + args.learner + ':' + args.pretrained + ':' + args.mix + ':' + args.dataset_emb_comp
    print("run_mode:", {run_mode})

    # set defauklt system params
    ml_logger.set_default('os', system.get_os())
    ml_logger.set_default('cpus', system.get_cpu_details())
    ml_logger.set_default('mem', system.get_total_mem())
    ml_logger.set_default('mode', run_mode)

    gpus = system.get_gpu_summary()
    if gpus is None:
        gpus = -1   
    ml_logger.set_default('gpus', gpus)

    # epoch and run fields are deprecated
    ml_logger.set_default('epoch', -1)
    ml_logger.set_default('run', -1)

    representation, optimized = get_representation(args)
    print("representation:", {representation})

    embeddings = get_embeddings(args)
    print("embeddings:", {embeddings})

    lm_type = get_language_model_type(args.pretrained)
    print("lm_type:", {lm_type})

    # check to see if the model has been run before
    already_computed = ml_logger.already_calculated(
        dataset=args.dataset,
        model=args.learner,
        representation=representation,
        #embeddings=embeddings
        )

    print("already_computed:", already_computed)

    return already_computed, vtype, learner, pretrained, embeddings, lm_type, emb_path, args.mix, representation, ml_logger, optimized

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_embeddings_path(pretrained, args):
    
    if (pretrained == 'bert'):
        return args.bert_path
    elif pretrained == 'roberta':
        return args.roberta_path
    elif pretrained == 'glove':
        return args.glove_path
    elif pretrained == 'word2vec':
        return args.word2vec_path
    elif pretrained == 'fasttext':
        return args.fasttext_path
    elif pretrained == 'llama':
        return args.llama_path
    elif pretrained == 'xlnet':
        return args.xlnet_path
    elif pretrained == 'gpt2':
        return args.gpt2_path
    else:
        raise ValueError(f"Unsupported pretrained model '{pretrained}'")
        


def get_embeddings(args):

    print("get_embeddings...")

    if (args.pretrained):
        return args.pretrained + ':' + args.mix
    else:
        return args.mix


def get_language_model_type(embeddings):

    if (embeddings in ['glove']):
        return 'static:word:co-occurrence:global'
    elif (embeddings in ['word2vec']):
        return 'static:word:co-occurrence:local'
    elif (embeddings in ['fasttext']):
        return 'static:subword:co-occurrence:local'
    elif (embeddings in ['llama', 'gpt2']):
        return 'transformer:token:autoregressive:unidirectional:causal'
    elif (embeddings in ['bert', 'roberta']):
        return 'transformer:token:autoregressive:bidirectional:masked'
    elif (embeddings in ['xlnet']):
        return 'transformer:token:autoregressive:bidirectional:permutated'
    else:
        return 'unknown'
    

def get_representation(args):

    print("calculating representation...")

    # set model and dataset
    method_name = f'[{args.learner}:{args.dataset}]:->'

    #set representation form

    if (args.mix == 'vmode'):
        method_name += f'{args.vtype}[{args.pretrained}]'

    # solo is when we project the doc, we represent it, in the 
    # underlying pretrained embedding space - with three options 
    # as to how that space is computed: 1) weighted, 2) avg, 3) summary
    elif (args.mix == 'solo'):
        method_name += f'{args.pretrained}({args.dataset_emb_comp})'

    elif (args.mix == 'solo-wce'):
        method_name += f'{args.pretrained}({args.dataset_emb_comp})+{args.vtype}.({args.pretrained}+wce({args.vtype}))'

    # cat is when we concatenate the doc representation in the
    # underlying pretrained embedding space with the tfidf vectors - 
    # we have the same three options for the dataset embedding representation
    elif (args.mix == 'cat-doc'):
        method_name += f'{args.vtype}+{args.pretrained}({args.dataset_emb_comp})'

    elif (args.mix == 'cat-wce'):
        method_name += f'{args.vtype}+{args.vtype}.({args.pretrained}+wce({args.vtype}))'
    
    elif (args.mix == 'cat-doc-wce'):
        method_name += f'{args.vtype}+{args.vtype}.({args.pretrained}+wce({args.vtype}))+{args.vtype}.({args.pretrained}+wce({args.vtype}))'
        
    # dot is when we project the tfidf vectors into the underlying
    # pretrained embedding space using matrix multiplication, i.e. dot product
    # we have the same three options for the dataset embedding representation computation
    elif (args.mix == 'dot'):
        method_name += f'{args.vtype}.{args.pretrained}'

    elif (args.mix == 'dot-wce'):
        method_name += f'{args.vtype}.({args.pretrained}+wce({args.vtype}))'
        
    # vmode is when we simply use the frequency vector representation (TF-IDF or Count)
    # as the dataset representation into the model
    elif (args.mix == 'vmode'):
        method_name += f'{args.vtype}[{args.pretrained}]'
    
    # lsa is when we use SVD (aka LSA) to reduce the number of featrues from 
    # the vectorized data set, LSA is a form of dimensionality reduction
    elif (args.mix == 'lsa'):
        method_name += f'{args.vtype}->LSA[{args.pretrained}].({args.pretrained})'

    elif (args.mix == 'lsa-wce'):
        method_name += f'{args.vtype}->LSA[{args.pretrained}]+({args.vtype}.{args.pretrained})+wce'

    #
    # set optimized field to true if its a neural model 
    # and we are tuning (fine-tuning) it or if its an ML
    # model and we are optimizing the prameters ofr best results
    #
    if (args.learner in NEURAL_MODELS and args.tunable) or (args.learner in ML_MODELS and args.optimc):
        method_name += ':[opt]'
        optimized = True
    else:
        method_name += ':[def]'
        optimized = False
    
    print("method_name:", method_name)

    return method_name, optimized



if __name__ == '__main__':

    available_datasets = LCDataset.dataset_available

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification Testing')
    
    parser.add_argument('--dataset', required=True, type=str, default='20newsgroups', metavar='N', help=f'dataset, one in {available_datasets}')
    
    parser.add_argument('--pickle-dir', type=str, default=PICKLE_DIR, metavar='str', help=f'path where to load the pickled dataset from')
    
    parser.add_argument('--logfile', type=str, default='../log/ml_classify.test', metavar='N', help='path to the application log file')
    
    parser.add_argument('--learner', type=str, default='svm', metavar='N', help=f'learner (svm, lr, or nb)')
    
    parser.add_argument('--vtype', type=str, default='tfidf', metavar='N', help=f'dataset base vectorization strategy, in [tfidf, count]')
                        
    parser.add_argument('--mix', type=str, default='solo', metavar='N', help=f'way to prepare the embeddings, in [vmode, cat, cat-wce, dot, dot-wce, solo, solo-wce, lsa, lsa-wce]. NB presumes --pretrained is set')

    parser.add_argument('--cm', action='store_true', default=False, help=f'create confusion matrix for underlying model')
                             
    parser.add_argument('--optimc', action='store_true', default=False, help='optimize the model using relevant models params')

    parser.add_argument('--force', action='store_true', default=False,
                    help='force the execution of the experiment even if a log already exists')
    
    parser.add_argument('--pretrained', type=str, default=None, metavar='glove|word2vec|fasttext|bert|roberta|gpt2|xlnet|llama',
                        help='pretrained embeddings, use "glove", "word2vec", "fasttext", "bert", "roberta", "xlnet", "gpt2" or "llama" (default None)')

    parser.add_argument('--dataset-emb-comp', type=str, default='avg', metavar='weighted|avg|summary',
                        help='how to compute dataset embedding representation form, one of "weighted", "avg", or "summary (cls)" (default avg)')
    
    parser.add_argument('--embedding-dir', type=str, default=VECTOR_CACHE, metavar='str',
                        help=f'path where to load and save embeddings')
    
    parser.add_argument('--word2vec-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'path + filename to Word2Vec pretrained vectors (e.g. ../.vector_cache/GoogleNews-vectors-negative300.bin), used only '
                             f'with --pretrained word2vec')
    
    parser.add_argument('--glove-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to pretrained glove embeddings (e.g. glove.840B.300d.txt.pt file), used only with --pretrained glove')
    
    parser.add_argument('--fasttext-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'path + filename to fastText pretrained vectors (e.g. --fasttext-path ../.vector_cache/crawl-300d-2M.vec), used only '
                            f'with --pretrained fasttext')
    
    parser.add_argument('--bert-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to BERT pretrained vectors (NB used only with --pretrained bert)')

    parser.add_argument('--roberta-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to RoBERTa pretrained vectors (NB used only with --pretrained roberta)')
    
    parser.add_argument('--gpt2-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to GPT2 pretrained vectors (NB used only with --pretrained gpt2)')
    
    parser.add_argument('--xlnet-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to XLNet pretrained vectors (NB used only with --pretrained xlnet)')
    
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
    already_modelled, vtype, learner, pretrained, embeddings, lm_type, emb_path, mix, representation, logfile, optimized = initialize_ml_testing(args)

    # check to see if model params have been computed already
    if (already_modelled) and not (args.force):
        print(f'Assertion warning: model {representation} with embeddings {embeddings} for {args.dataset} already calculated.')
        print("Run with --force option to override, exiting...")
        exit(0)
        
    #
    # run the model using input params
    #
    run_model(
        dataset=args.dataset, 
        vtype=vtype,
        embeddings=embeddings,
        embedding_path=emb_path,
        representation=representation,
        lang_model_type=lm_type,
        optimized=optimized,
        logfile=logfile, 
        args=args
    )
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------