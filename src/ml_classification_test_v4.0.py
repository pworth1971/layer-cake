import numpy as np
import argparse
from time import time

from scipy.sparse import csr_matrix, csc_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder

# custom imports
from embedding.supervised import get_supervised_embeddings, compute_tces

from util.common import *

from data.lc_dataset import LCDataset, loadpt_data
from data.lc_trans_dataset import PICKLE_DIR, RANDOM_SEED

from model.classification import ml_classification

from model.LCRepresentationModel import PICKLE_DIR, VECTOR_CACHE



import warnings
warnings.filterwarnings('ignore')



# -------------------------------------------------------------------------------------------------------------------
# gen_model()
# -------------------------------------------------------------------------------------------------------------------


def gen_xdata(Xtr_raw, 
              Xtr_vectorized, 
              y_train, 
              Xte_raw,
              Xte_vectorized, 
              y_test, 
              vec_vocab_size,
              dataset=None,
              pretrained=None, 
              pretrained_vectors_dictionary=None, 
              weighted_embeddings_train=None, 
              weighted_embeddings_test=None, 
              avg_embeddings_train=None, 
              avg_embeddings_test=None, 
              summary_embeddings_train=None, 
              summary_embeddings_test=None, 
              dataset_embedding_type='weighted', 
              mix='solo', 
              supervised=False):
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
    vec_vocab_size : int
        The size of the vocabulary used for the vectorized representation.
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
        
        
    # ------------------------------------------------------------------------------------------------------------------        
    # compute WCEs / TCEs
    #
    
    # Split the string by colon
    parts = pretrained.split(':')

    # Get the first part of the string
    pretrained = parts[0].strip()       # Using strip() to remove any leading/trailing whitespace
    print("pretrained:", pretrained)

    print("WORD_BASED_MODELS:", WORD_BASED_MODELS)
    print("TOKEN_BASED_MODELS:", TOKEN_BASED_MODELS)
    
    if (supervised):
        
        if (pretrained in WORD_BASED_MODELS):
            print("\n\tcomputing WCEs...")
            Xtr_ces = get_supervised_embeddings(Xtr_vectorized, y_train)
            Xte_ces = get_supervised_embeddings(Xte_vectorized, y_test)
            
        elif (pretrained in TOKEN_BASED_MODELS):
            print("\n\tcomputing TCEs...")
        
            """"
            if (dataset in LCDataset.single_label_datasets):
                print("single label, converting target labels to to one-hot for tce computation...")
                
                # Assuming labels are a numpy array of shape (num_samples,)
                def to_one_hot(labels):
                    encoder = OneHotEncoder(categories='auto', sparse_output=True, dtype=np.float32)
                    labels = labels.reshape(-1, 1)
                    one_hot = encoder.fit_transform(labels)  # Result is a sparse matrix
                    return one_hot

                one_hot_labels_train = to_one_hot(y_train.toarray())
                one_hot_labels_test = to_one_hot(y_test.toarray())

            else:
                one_hot_labels_train = y_train
                one_hot_labels_test = y_test

            # Ensure one_hot_labels_train is a csr_matrix
            if not isinstance(one_hot_labels_train, csr_matrix):
                print("Converting one_hot_labels_train to csr_matrix...")
                one_hot_labels_train = csr_matrix(one_hot_labels_train)
            """
            
            print("computing TCE training data...")
            Xtr_ces = compute_tces(
                vocabsize=vec_vocab_size,
                vectorized_training_data=Xtr_vectorized,
                #training_label_matrix=one_hot_labels_train,
                training_label_matrix=y_train,
                opt=args,
                debug=True,
                asarray=True
            )
            
            print("computing TCE test data...")
            Xte_ces = compute_tces(
                vocabsize=vec_vocab_size,
                vectorized_training_data=Xte_vectorized,
                #training_label_matrix=one_hot_labels_test,
                training_label_matrix=y_test,
                opt=args,
                debug=True,
                asarray=True
            )
        else:
            raise ValueError(f"Unsupported pretrained model '{pretrained}'")
    
        print("Xtr_ces:", type(Xtr_ces), Xtr_ces.shape)
        print("Xtr_ces[0]:\n", type(Xtr_ces[0]), Xtr_ces[0].shape, Xtr_ces[0])

        print("Xte_ces:", type(Xte_ces), Xte_ces.shape)
        print("Xte_ces[0]:", type(Xte_ces[0]), Xte_ces[0].shape, Xte_ces[0])

        # Check for null values in Xtr_ces
        if np.isnan(Xtr_ces).any() or np.isinf(Xtr_ces).any():
            print("[WARNING] Xtr_ces contains NaN or Inf values during initialization.")
            raise ValueError("[ERROR] Xtr_ces contains NaN or Inf values during initialization.")

        # Check for null values in Xte_ces
        if np.isnan(Xte_ces).any() or np.isinf(Xte_ces).any():
            print("[WARNING] Xte_ces contains NaN or Inf values during initialization.")
            raise ValueError("[ERROR] Xte_ces contains NaN or Inf values during initialization.")
        #
        # ------------------------------------------------------------------------------------------------------------------
        
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
        # Next we build the WCE representation by projecting 
        # the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product. 
        #
        print("stacking embedding_vocab_matrix with WCEs...")
        pretrained_vectors_dictionary = np.hstack([pretrained_vectors_dictionary, Xtr_ces])
        print("embedding dataset vocab representation after CE stacking:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
    
        X_train_ce = Xtr_vectorized @ pretrained_vectors_dictionary
        X_test_ce = Xte_vectorized @ pretrained_vectors_dictionary
        
        print("after dot product...")
        print("X_train_ce:", type(X_train_ce), X_train_ce.shape)
        print("X_test_ce:", type(X_test_ce), X_test_ce.shape)

        # Now we stack the CEs (from the vectorized representation) with the 
        # dataset embedding representation from the LCRepresentationModel.encode_docs() 
        # method, ie the embedidng representation of the document dataset
        X_train = np.hstack([X_train, X_train_ce])
        X_test = np.hstack([X_test, X_test_ce])

        print("after stacking CEs with embedding dataset representation...")
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
    
    # here we are concatenating the vectorized representation of the dataset along with the CEs only
    elif mix == 'cat-wce':
        
        print("Concatenting CEs with vectorized dataset representation (cat-wce)...")
        
        #
        # here we project the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product. NB we maintain the sparse 
        # representations here for faster compute performance
        #
        print("stacking embedding_vocab_matrix with CEs...")
        pretrained_vectors_dictionary = np.hstack([pretrained_vectors_dictionary, Xtr_ces])

        print("embedding dataset vocab representation after stacking CEs:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
    
        X_train_ce = Xtr_vectorized @ pretrained_vectors_dictionary
        X_test_ce = Xte_vectorized @ pretrained_vectors_dictionary
        
        print("after dot product...")
        print("X_train_ce:", type(X_train_ce), X_train_ce.shape)
        print("X_test_ce:", type(X_test_ce), X_test_ce.shape)
        
        # After applying the CEs, we concatenate the resulting matrix with the 
        # original feature matrix to augment the vectorized feature set with the 
        # embedding space for said vectorized feature space
        X_train = np.hstack([Xtr_vectorized.toarray(), X_train_ce])              
        X_test = np.hstack([Xte_vectorized.toarray(), X_test_ce])

    # here we are concatenating the vectorized representation of the dataset 
    # with the embedding representation of the dataset AND the CEs
    elif mix == 'cat-doc-wce':

        print("Concatenating the vectorized text representation with embedding doc representation and the CEs (cat-doc-wce)...")
                
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
        print("stacking embedding_vocab_matrix with CEs...")
        pretrained_vectors_dictionary = np.hstack([pretrained_vectors_dictionary, Xtr_ces])

        print("embedding dataset vocab representation after stacking CEs:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
    
        X_train_ce = Xtr_vectorized @ pretrained_vectors_dictionary
        X_test_ce = Xte_vectorized @ pretrained_vectors_dictionary

        print("after dot product...")
        print("X_train_ce:", type(X_train_ce), X_train_ce.shape)
        print("X_test_ce:", type(X_test_ce), X_test_ce.shape)
        
        # After computing the WCEs and the related vectorizxed + embedding+CE representation, 
        # we concatenate the resulting matrix with the stacked vectorized + embedding dataset 
        # representation to further augment the already augmented vectorized feature set with 
        # the additional CE information
        X_train = np.hstack([X_train.toarray(), X_train_ce])              
        X_test = np.hstack([X_test.toarray(), X_test_ce])

        print("after stacking CEs with vectorized represnetation and embedding representations of dataset...")
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
        
        print("projecting vectorized representation into embedding space with CEs (dot-wce)...")
        
        # here we project the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product.
        # 
        # NB we maintain the sparse representations here for faster compute performance
        print("stacking embedding_vocab_matrix with CEs...")
        pretrained_vectors_dictionary = np.hstack([pretrained_vectors_dictionary, Xtr_ces])

        print("embedding dataset vocab representation after stacking CESs", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
    
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
    # the reduced feature set with the CEs
    elif mix == 'lsa-wce':
        
        print("using SVD (LSA) with CEs (lsa-wce)...")

        n_dimensions = pretrained_vectors_dictionary.shape[1]
        print("n_dimensions:", n_dimensions)
        
        svd = TruncatedSVD(n_components=n_dimensions)           # Adjust dimensions to match pretrained embeddings
        
        # reset X_train and X_test to the original tfidf vectors
        # using the svd model
        X_train = svd.fit_transform(Xtr_vectorized)
        X_test = svd.transform(Xte_vectorized)

        # here we project the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product. NB we maintain the sparse 
        # representations here for faster compute performance
        print("stacking embedding_vocab_matrix with CEs...")
        pretrained_vectors_dictionary = np.hstack([pretrained_vectors_dictionary, Xtr_ces])

        print("embedding dataset vocab representation after stcking CES:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
    
        X_train_ce = Xtr_vectorized @ pretrained_vectors_dictionary
        X_test_ce = Xte_vectorized @ pretrained_vectors_dictionary
        
        print("after dot product...")
        print("X_train_wce:", type(X_train_ce), X_train_ce.shape)
        print("X_test_wce:", type(X_test_ce), X_test_ce.shape)
        
        # After applying the CEs, we concatenate the resulting matrix with the 
        # LSA feature reduced vectiorized representation of the dataset to augment 
        # the LSA features with the CEs
        X_train = np.hstack([X_train, X_train_ce])              
        X_test = np.hstack([X_test, X_test_ce])

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

    print(f"\n\trunning model...") 
    print(f"dataset={dataset}, vtype={vtype}, representation={representation}, lang_model_type={lang_model_type}, optimized={optimized}")
    
    embedding_type = get_embedding_type(args.pretrained)
    print(f"embeddings: {embeddings}, embedding_type: {embedding_type}, embedding_path: {embedding_path}")
    
    #
    # Load the dataset and the associated (pretrained) embedding structures
    # to be fed into the model
    #                                
    lc_dataset = loadpt_data(
                    dataset=dataset,                                            # Dataset name
                    vtype=args.vtype,                                           # Vectorization type
                    pretrained=args.pretrained,                                 # pretrained embeddings type
                    embedding_path=embedding_path,                              # path to pretrained embeddings
                    emb_type=embedding_type,                                    # embedding type (word or token)
                    embedding_comp_type=args.dataset_emb_comp,                  # dataset doc embedding method (avg, summary)
                    seed=args.seed                                              # random seed    
                    )                           
        
    Xtr_raw = lc_dataset.devel_raw
    Xte_raw = lc_dataset.test_raw
    Xtr_vectorized = lc_dataset.Xtr_vectorized
    Xte_vectorized = lc_dataset.Xte_vectorized
    y_train_sparse = lc_dataset.y_train_sparse
    y_test_sparse = lc_dataset.y_test_sparse
    target_names = lc_dataset.target_names
    class_type = lc_dataset.class_type
    embedding_vocab_matrix = lc_dataset.embedding_vocab_matrix
    Xtr_weighted_embeddings = lc_dataset.Xtr_weighted_embeddings
    Xte_weighted_embeddings = lc_dataset.Xte_weighted_embeddings
    Xtr_avg_embeddings = lc_dataset.Xtr_avg_embeddings
    Xte_avg_embeddings = lc_dataset.Xte_avg_embeddings
    Xtr_summary_embeddings = lc_dataset.Xtr_summary_embeddings
    Xte_summary_embeddings = lc_dataset.Xte_summary_embeddings
    
    print("--- LCDataset loaded ---")
 
    print("Xtr_raw:", type(Xtr_raw), Xtr_raw.shape)
    print("Xte_raw:", type(Xte_raw), Xte_raw.shape)

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

    """
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

    y_train = y_train_sparse
    y_test = y_test_sparse

    num_classes = lc_dataset.nC
    print("num_classes:", num_classes)
    
    vec_vocab_size = len(lc_dataset.vectorizer.vocabulary_)
    print("vec_vocab_size:", vec_vocab_size)
    
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
            vec_vocab_size=vec_vocab_size,
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
            mix=args.mix,
            supervised=args.supervised
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

    #dims = X_train.shape[1]
    dims = X_train.shape
    print("# dimensions:", dims)

    comp_method = get_model_computation_method(
        vtype=vtype,
        pretrained=args.pretrained, 
        embedding_type=embedding_type, 
        learner=args.net, 
        mix=args.mix
    )
    print("comp_method:", comp_method)

    logfile.insert(dataset=args.dataset, class_type=class_type, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='final-te-macro-f1', value=Mf1, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='final-te-micro-f1', value=mf1, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-accuracy', value=acc, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-precision', value=precision, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-recall', value=recall, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, embeddings=embeddings, representation=representation, lm_type=lang_model_type, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-jacard-index', value=j_index, timelapse=tend)

    return acc, Mf1, mf1, h_loss, precision, recall, j_index, tend

    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    available_datasets = LCDataset.dataset_available

    # Training settings
    parser = argparse.ArgumentParser(description='Layer Cake ML Text Classification Testing')
    
    parser.add_argument('--dataset', required=True, type=str, default='20newsgroups', 
                        help=f'dataset, one in {available_datasets}')
    parser.add_argument('--pickle-dir', type=str, default=PICKLE_DIR, metavar='DIR', 
                        help=f'path where to load the pickled dataset from')
    parser.add_argument('--logfile', type=str, default='../log/ml_classify.test', metavar='FILE', 
                        help='path to the application log file')
    parser.add_argument('--net', type=str, default='svm', metavar='MODEL', 
                        help=f'model / learner, one of [svm, lr, or nb]')
    parser.add_argument('--vtype', type=str, default='tfidf', 
                        help=f'dataset base vectorization strategy, in [tfidf, count]')
    parser.add_argument('--pretrained', type=str, default=None, metavar='MODEL',
                        help=f'Language model to use, either "None" or in {SUPPORTED_LMS}, default None.')
    parser.add_argument('--dataset-emb-comp', type=str, default='avg',
                        help='Pooling strategy transformer model dataset representation. One of "avg", "summary" (CLS) or "weighted", defaults to "avg".')
    parser.add_argument('--embedding-dir', type=str, default=VECTOR_CACHE, metavar='DIR',
                        help=f'path where to load and save embeddings')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, metavar='SEED',
                        help=f'random seed (default: {RANDOM_SEED})')
    parser.add_argument('--mix', type=str, default='solo', metavar='MIX', 
                        help=f'way to prepare the embeddings, in [vmode, cat, cat-wce, dot, dot-wce, solo, solo-wce, lsa, lsa-wce]. NB presumes --pretrained is set')
    parser.add_argument('--cm', action='store_true', default=False, 
                        help=f'create confusion matrix for underlying model')     
    parser.add_argument('--optimc', action='store_true', default=False, 
                        help='optimize the model using relevant models params')
    parser.add_argument('--force', action='store_true', default=False,
                    help='force the execution of the experiment even if a log already exists')
    
    # supervised options
    parser.add_argument('--supervised', action='store_true', 
                        help='Use supervised embeddings (TCEs')
    parser.add_argument('--nozscore', action='store_true', default=False,
                        help='disables z-scoring form the computation of TCE')
    parser.add_argument('--supervised-method', type=str, default='dotn',
                        help='WCE/TCE matrix computation method. One of dotn (default), ppmi (positive pointwise mutual information), ig (information gain) or chi (Chi-squared)')
    parser.add_argument('--max-label-space', type=int, default=300, metavar='int',
                        help='larger dimension allowed for the feature-label embedding (if larger, then PCA with this '
                             'number of components is applied (default 300)')
    
    args = parser.parse_args()
    print("args:", type(args), args)    
    
    #print("available_datasets:", available_datasets)

    # check valid dataset
    assert args.dataset in available_datasets, \
        f'unknown dataset {args.dataset}'

    # check valid NB arguments
    if (args.net in ['nb', 'NB'] and args.mix != 'vmode'):
        print(f'Warning: abandoning run. Naive Bayes model (nb) can only be run with no --pretrained parameter and with --mix vmode (due to lack of support for negative embeddings). Exiting...')
        exit(0)
    
    
    print("\n\t-------------------------------------------------------- Layer Cake: ML baseline classification code: main() --------------------------------------------------------")

    # Get the full model identifier and cache directory path for tokenizer/model
    model_name, model_path = get_model_identifier(args.pretrained)
    print("model_name:", model_name)
    print("model_path:", model_path)

    if (args.pretrained == 'glove'):
        args.glove_path = model_path
    elif (args.pretrained == 'word2vec'):
        args.word2vec_path = model_path
    elif (args.pretrained == 'fasttext'):
        args.fasttext_path = model_path
    elif (args.pretrained == 'bert'):
        args.bert_path = model_path
    elif (args.pretrained == 'roberta'):
        args.roberta_path = model_path
    elif (args.pretrained == 'distilbert'):
        args.distilbert_path = model_path
    elif (args.pretrained == 'xlnet'):
        args.xlnet_path = model_path
    elif (args.pretrained == 'gpt2'):
        args.gpt2_path = model_path
    elif (args.pretrained == 'llama'):
        args.llama_path = model_path
    elif (args.pretrained == 'deepseek'):
        args.deepseek_path = model_path
    else:
        raise ValueError("Unsupported pretrained model:", args.pretrained)
    
    if (args.mix in CLASS_EMBEDDING_MODES):
        args.supervised = True
    else:
        args.supervised = False

    print("args:", args)    
                
    program_name = "ml_classification_test"
    version = '4.0' 
    print(f"program_name: {program_name}, version: {version}")

    # initialize log file and run params
    already_modelled, vtype, learner, pretrained, embeddings, lm_type, emb_path, mix, representation, logfile, optimized = initialize_ml_testing(args, model_name, program_name, version)

    # check to see if model params have been computed already
    if (already_modelled) and not (args.force):
        print(f"\n\t ----- [WARNING] model {representation} with embeddings {embeddings} for {args.dataset} already computed, run with --force option to override. -----")
        exit(0)

    print("\n\tInitialization params:")    
    print("\tvtype:", vtype)
    print("\tlearner:", learner)
    print("\tpretrained:", pretrained)
    print("\tmodel_name:", model_name)
    print("\tembeddings:", embeddings)
    print("\tlm_type:", lm_type)
    print("\temb_path:", emb_path)
    print("\tmix:", mix)
    print("\trepresentation:", representation)
    print("\toptimized:", optimized)

    print()
    
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