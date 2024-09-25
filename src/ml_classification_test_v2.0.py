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

from data.lc_dataset import LCDataset, save_to_pickle, load_from_pickle

from model.classification import run_model

from util.csv_log import CSVLog

#from model.CustomRepresentationLearning import BERTRepresentationModel, RoBERTaRepresentationModel, LlaMaRepresentationModel
from model.LCRepresentationModel import FASTTEXT_MODEL, GLOVE_MODEL, WORD2VEC_MODEL
from model.LCRepresentationModel import BERT_MODEL, ROBERTA_MODEL, LLAMA_MODEL

#import pickle

#from util import file


import warnings
warnings.filterwarnings('ignore')





TEST_SIZE = 0.25




def embedding_matrix(dataset, pretrained=False, pretrained_vectors=None, supervised=False):
    
    """
    Constructs and returns embedding matrices using either pre-trained or supervised embeddings. Support for GloVe, Word2Vec
    FastText and BERT pre-trained embeddings supported (tested).

    Parameters:
    - dataset: The dataset object that contains vocabulary and other dataset-specific parameters.
    - pretrained: Boolean indicating whether to use pretrained embeddings.
    - pretrained_type: Type of pre-trained embeddings to use (e.g., 'glove', 'bert').
    - supervised: Boolean indicating whether to use supervised embeddings.
    - emb_path: Path to the directory containing embedding files.
    
    Returns:
    - vocabulary: Sorted list of vocabulary terms.
    - pretrained_embeddings: Numpy array of stacked embeddings from all specified sources.
    """

    print('----- embedding_matrix()-----')
    
    assert pretrained or supervised, 'useless call without requiring pretrained and/or supervised embeddings'
    
    vocabulary = dataset.vocabulary
    vocabulary = np.asarray(list(zip(*sorted(vocabulary.items(), key=lambda x: x[1])))[0])

    print("vocabulary:", type(vocabulary), vocabulary.shape)

    pretrained_embeddings = []
    
    print("pretrained:", {pretrained})
    print("supervised:", {supervised})

    # NB: embeddings should be already loaded and sent in as a param in this case

    if pretrained and pretrained_vectors:

        print("pretrained and pretrained_vectors, extracting vocab and building pretrained embeddings...")

        P = pretrained_vectors.extract(vocabulary).numpy()
        print("pretrained_vectors: ", type(P), {P.shape})

        pretrained_embeddings.append(P)
        print(f'pretrained embeddings count after loading pretrained embeddings: {len(pretrained_embeddings[0])}')

    if supervised:
        print('\t[supervised-matrix]')

        Xtr, _ = dataset.vectorize()
        Ytr = dataset.devel_labelmatrix

        print("Xtr:", type(Xtr), Xtr.shape)
        print("Ytr:", type(Ytr), Ytr.shape)

        S = get_supervised_embeddings(Xtr, Ytr)
        print("supervised_embeddings:", type(S), {S.shape})

        pretrained_embeddings.append(S)
        print(f'pretrained embeddings count after appending supervised: {len(pretrained_embeddings[1])}')

    pretrained_embeddings = np.hstack(pretrained_embeddings)
    print("after np.hstack(): pretrained_embeddings:", type(pretrained_embeddings), {pretrained_embeddings.shape})

    return vocabulary, pretrained_embeddings

# -----------------------------------------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------------------------
# gen_embeddings()
# -------------------------------------------------------------------------------------------------------------------
def gen_embeddings(Xtr_raw, Xtr_vectorized, y_train, Xte_raw, Xte_vectorized, y_test, dataset='bbc-news', pretrained=None, pretrained_vectors_dictionary=None, \
                   weighted_embeddings_train=None, weighted_embeddings_test=None, avg_embeddings_train=None, avg_embeddings_test=None, summary_embeddings_train=None, \
                    summary_embeddings_test=None, dataset_embedding_type='weighted', mix='solo', supervised=False):
    
    print("\n\tgenerating embeddings...")
 
    print("dataset:", dataset)    
    print("pretrained:", pretrained)
    
    # should be a numpy array
    print("pretrained_vectors_dictionary:", type(pretrained_vectors_dictionary), "Shape:", pretrained_vectors_dictionary.shape)    
    print("weighted_embeddings_train:", type(weighted_embeddings_train), weighted_embeddings_train.shape)
    print("weighted_embeddings_test:", type(weighted_embeddings_test), weighted_embeddings_test.shape)
    print("avg_embeddings_train:", type(avg_embeddings_train), avg_embeddings_train.shape)
    print("avg_embeddings_test:", type(avg_embeddings_test), avg_embeddings_test.shape)
    print("summary_embeddings_train:", type(summary_embeddings_train), summary_embeddings_train.shape)
    print("summary_embeddings_test:", type(summary_embeddings_test), summary_embeddings_test.shape)
    print("mix:", mix)
    print("dataset_embedding_type:", dataset_embedding_type)
      
    print('Xtr_raw:', type(Xtr_raw), Xtr_raw.shape)
    print("Xtr_vectorized:", type(Xtr_vectorized), Xtr_vectorized.shape)
    print("y_train:", type(y_train), y_train.shape)
    print("Xte_raw:", type(Xte_raw), Xte_raw.shape)
    print("Xte_vectorized:", type(Xte_vectorized), Xte_vectorized.shape)
    print('y_test:', type(y_test), y_test.shape)

    """
    # build vocabulary numpy array
    vocabulary = np.asarray(list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0])
    print("vocabulary:", type(vocabulary), vocabulary.shape)
    
    pretrained_embeddings = []                                  # List to store embedding matrix

    if (pretrained in SUPPORTED_LMS):
        
        print("setting up pretrained embeddings...")

        #P = pretrained_vectors_dictionary.extract(vocabulary).numpy()
        #print("pretrained_vectors_dictionary: ", type(pretrained_vectors_dictionary), {pretrained_vectors_dictionary.shape})

        pretrained_embeddings.append(pretrained_vectors_dictionary)
        print(f'pretrained embeddings count after loading pretrained embeddings: {len(pretrained_embeddings[0])}')

    
    if supervised:
        print('supervised...')

        #Xtr, _ = dataset.vectorize()
        #Ytr = dataset.devel_labelmatrix

        print("X_train:", type(X_train), X_train.shape)
        print("y_train:", type(y_train), y_train.shape)

        S = get_supervised_embeddings(X_train, y_train)
        print("supervised_embeddings:", type(S), {S.shape})

        pretrained_embeddings.append(S)
        print(f'pretrained embeddings count after appending supervised: {len(pretrained_embeddings[1])}')
    """
    
    
    #
    # TESTING
    #
    # load the appropriate custom (LC) pretrained representation model
    #
    """
    lcr_model = None          # custom representation model class object

    if pretrained == 'bert':
        lcr_model = BERTLCRepresentationModel(
            model_name=BERT_MODEL, 
            model_dir=args.bert_path, 
            comp_method=dataset_embedding_type, 
            device=None
        )
    
    elif pretrained == 'roberta':
        lcr_model = RoBERTaLCRepresentationModel(
            model_name=ROBERTA_MODEL, 
            model_dir=args.roberta_path, 
            comp_method=dataset_embedding_type, 
            device=None
        )

    elif pretrained == 'llama':
        lcr_model = LlaMaLCRepresentationModel(
            model_name=LLAMA_MODEL, 
            model_dir=args.llama_path, 
            comp_method=dataset_embedding_type, 
            device=None
        )
    else:    
        print(f"ValueError: Unsupported pretrained embeddings type '{pretrained}'")
        return None
    
    avg_embeddings_train = lcr_model.encode_sentences(Xtr_raw.tolist())                                     # encode the dataset in the specified form
    print("avg_embeddings_train:", type(avg_embeddings_train), avg_embeddings_train.shape)

    avg_embeddings_test = lcr_model.encode_sentences(Xte_raw.tolist())
    print("avg_embeddings_test:", type(avg_embeddings_test), avg_embeddings_test.shape)
    """
    #
    # 
    # END TESTING

    if mix == 'solo':
        
        print("Using just the word embeddings alone (solo)...")
        
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
            # we compuet them in LCDataset class 
            X_train = summary_embeddings_train
            X_test = summary_embeddings_test
        else:
            print(f"Unsupported dataset_embedding_type '{dataset_embedding_type}'")
            return None   
        
    elif mix == 'cat':        
        print("Concatenating word embeddings with TF-IDF vectors...")
                
        # 
        # Here we concatenate the tfidf vectors and the specified form 
        # of dataset embedding representation 
        #  
        if (dataset_embedding_type == 'weighted'):
            X_train = np.hstack([Xtr_vectorized.toarray(), weighted_embeddings_train])
            X_test = np.hstack([Xte_vectorized.toarray(), weighted_embeddings_test])
        elif (dataset_embedding_type == 'avg'):
            X_train = np.hstack([Xtr_vectorized.toarray(), avg_embeddings_train])
            X_test = np.hstack([Xte_vectorized.toarray(), avg_embeddings_test])
        elif (dataset_embedding_type == 'summary'):
            X_train = np.hstack([Xtr_vectorized.toarray(), summary_embeddings_train])
            X_test = np.hstack([Xte_vectorized.toarray(), summary_embeddings_test])
        else:
            print(f"Unsupported dataset_embedding_type '{dataset_embedding_type}'")
            return None

    elif mix == 'dot':
        
        print("Dot product (matrix multiplication) of embeddings matrix with TF-IDF vectors...")
        
        #
        # here we project the tfidf vectors into the pretrained embedding (vocabulary) space
        # using matrix multiplication, i.e. dot product 
        #
        print("before dot product...")
        
        X_train = Xtr_vectorized.toarray()
        X_test = Xte_vectorized.toarray()
        print("X_train:", type(X_train), X_train.shape)
        #print("X_train[0]:\n", X_train[0])
        print("X_test:", type(X_test), X_test.shape)
        #print("X_test[0]\n:", X_test[0])
        
        print("pretrained_vectors_dictionary:", type(pretrained_vectors_dictionary), pretrained_vectors_dictionary.shape)
        #print("pretrained_vectors_dictionary[0]:\n", pretrained_vectors_dictionary[0])
               
        X_train = np.dot(X_train, pretrained_vectors_dictionary)
        X_test = np.dot(X_test, pretrained_vectors_dictionary)
        
        print("after dot product...")
        print("X_train:", type(X_train), X_train.shape)
        #print("X_train[0]:\n", X_train[0])
        print("X_test:", type(X_test), X_test.shape)
        #print("X_test[0]:\n", X_test[0])
    
    elif mix == 'lsa':
        
        n_dimensions = pretrained_vectors_dictionary.shape[1]
        print("n_dimensions:", n_dimensions)
        
        svd = TruncatedSVD(n_components=n_dimensions)           # Adjust dimensions to match pretrained embeddings
        
        # reset X_train and X_test to the original tfidf vectors
        # using the svd model
        X_train = svd.fit_transform(Xtr_vectorized)
        X_test = svd.transform(Xte_vectorized)

    elif mix == 'vmode':
        X_train = Xtr_vectorized
        X_test = Xte_vectorized

    else:
        print(f"Unsupported mix mode '{mix}'")
        return None    
        
    print("after embedding generation...")
    
    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)

    # return pretrained_embeddings, sup_range
    return X_train, X_test




# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# classify_data(): Core processing function
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def classify_data(dataset='20newsgrouops', vtype='tfidf', embeddings=None, embedding_path=None, representation=None, optimized=False, logfile=None, args=None):
    """
    Core function for classifying text data using various configurations like embeddings, methods, and models.

    Parameters:
    - dataset (str): The name of the dataset to use (e.g., '20newsgroups', 'ohsumed').
    - vtype (str): The vectorization type (e.g., 'tfidf', 'count').
    - embeddings (str or None): Specifies the type of pretrained embeddings to use (e.g., 'bert', 'llama'). If None, no embeddings are used.
    - embedding_path (str): Path to the pretrained embeddings file or directory.
    - representation (str or None): Specifies the classification method (optional).
    - optimized (bool): Whether the model is optimized for performance. 
    - logfile: Logfile object to store results and performance metrics.
    - args: Argument parser object, containing various flags for optimization and configuration (e.g., --optimc).

    Returns:
    acc: Accuracy of the model.
    Mf1: Macro F1 score of the model.
    mf1: Micro F1 score of the model.
    h_loss: Hamming loss of the model.
    precision: Precision of the model.
    recall: Recall of the model.
    j_index: Jaccard index of the model.
    tend: Time taken to run the model.

    Workflow:
    - Loads the dataset and the corresponding embeddings.
    - Prepares the input data and target labels for training and testing.
    - Splits the data into train and test sets.
    - Generates embeddings if pretrained embeddings are specified.
    - Calls the classification model (run_model) and logs the evaluation metrics.
    """

    print("\n\tclassifying...")

    print("representation:", representation)
    print("optimize:", optimized)


    if (args.pretrained is not None and args.pretrained in ['bert', 'roberta', 'llama']):
        embedding_type = 'token'
    else:
        embedding_type = 'word'
    
    print("embeddings:", embeddings)    
    print("embedding_type:", embedding_type)
    
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
    
    print("Tokenized data loaded.")
 
    print("Xtr_raw:", type(Xtr_raw), Xtr_raw.shape)
    print("Xte_raw:", type(Xte_raw), Xte_raw.shape)

    print("Xtr_vectorized:", type(Xtr_vectorized), Xtr_vectorized.shape)
    print("Xte_vectorized:", type(Xte_vectorized), Xte_vectorized.shape)

    print("y_train_sparse:", type(y_train_sparse), y_train_sparse.shape)
    print("y_test_sparse:", type(y_test_sparse), y_test_sparse.shape)
    
    # embedding_vocab_matrix should be a numpy array
    print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

    print("Xtr_weighted_embeddings:", type(Xtr_weighted_embeddings), Xtr_weighted_embeddings.shape)
    print("Xte_weighted_embeddings:", type(Xte_weighted_embeddings), Xte_weighted_embeddings.shape)
    
    print("Xtr_avg_embeddings:", type(Xtr_avg_embeddings), Xtr_avg_embeddings.shape)
    print("Xte_avg_embeddings:", type(Xte_avg_embeddings), Xte_avg_embeddings.shape)

    print("Xtr_summary_embeddings:", type(Xtr_summary_embeddings), Xtr_summary_embeddings.shape)
    print("Xte_summary_embeddings:", type(Xte_summary_embeddings), Xte_summary_embeddings.shape)

    print("transforming labels...")
    if isinstance(y_train_sparse, (csr_matrix, csc_matrix)):
        y_train = y_train_sparse.toarray()  # Convert sparse matrix to dense array for multi-label tasks
    if isinstance(y_test_sparse, (csr_matrix, csc_matrix)):
        y_test = y_test_sparse.toarray()  # Convert sparse matrix to dense array for multi-label tasks
        
    # Ensure y is in the correct format for classification type
    if class_type in ['singlelabel', 'single-label']:
        y_train = y_train.ravel()                       # Flatten y for single-label classification
        y_test = y_test.ravel()                       # Flatten y for single-label classification
    
    print("y_train after transformation:", type(y_train), y_train.shape)
    print("y_test after transformation:", type(y_test), y_test.shape)

    if args.pretrained is None:        # no embeddings in this case
        sup_tend = 0
    else:                              # embeddings are present
        tinit = time()

        print("building the embeddings...")

        # Generate embeddings
        X_train, X_test = gen_embeddings(
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

    Mf1, mf1, acc, h_loss, precision, recall, j_index, tend = run_model(X_train, X_test, y_train, y_test, args, target_names, class_type=class_type)

    tend += sup_tend

    dims = X_train.shape[1]
    print("# dimensions:", dims)

    comp_method = get_model_computation_method(args, embedding_type)
    print("comp_method:", comp_method)

    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='final-te-macro-F1', value=Mf1, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='final-te-micro-F1', value=mf1, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-accuracy', value=acc, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-precision', value=precision, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-recall', value=recall, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=representation, comp_method=comp_method, optimized=optimized, dimensions=dims, measure='te-jacard-index', value=j_index, timelapse=tend)

    return acc, Mf1, mf1, h_loss, precision, recall, j_index, tend

    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_model_computation_method(args, embedding_type='word'):

    print("calculating model computation method...")
    print("embedding_type:", embedding_type)

    if (args.pretrained in ['bert', 'roberta', 'llama']):
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

        if args.mix == 'solo':
            return pt_type
            
        elif args.mix == 'vmode':
            return f'frequency:{args.vtype}'

        elif args.mix == 'cat':
            return f'frequency:{args.vtype}+{pt_type}'

        elif args.mix == 'dot':
            return f'frequency:{args.vtype}.{pt_type}'

        elif args.mix == 'lsa':
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
    elif (pretrained == 'llama'):
        model_name = LLAMA_MODEL
    else:
        model_name = None

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
    
    """
    # disable warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    """

    # default to tfidf vectorization type unless 'count' specified explicitly
    if args.vtype == 'count':
        vtype = 'count'
    else:
        vtype = 'tfidf'             
        
    print("vtype:", {vtype})

    pretrained = False
    embeddings ='none'
    emb_path = VECTOR_CACHE

    if (args.pretrained is None) and (args.pretrained in NEURAL_MODELS or args.pretrained in ML_MODELS):
        pretrained = True

    # get the path to the embeddings
    emb_path = get_embeddings_path(args.pretrained, args)
    print("emb_path: ", {emb_path})

    model_type = f'{learner_name}-{args.vtype}-{args.mix}-{args.dataset_emb_comp}'
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

    run_mode = args.dataset + ':' + args.learner + ':' + args.mix + ':' + args.dataset_emb_comp

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

    # check to see if the model has been run before
    already_computed = ml_logger.already_calculated(
        dataset=args.dataset,
        model=args.learner,
        representation=representation,
        embeddings=embeddings
        )

    print("already_computed:", already_computed)

    return already_computed, vtype, learner, pretrained, embeddings, emb_path, args.mix, representation, ml_logger, optimized

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
    else:
        return None



def get_embeddings(args):

    emb = ''                # initialize to empty string

    if (args.pretrained and args.wce is False):
        emb += args.pretrained
    elif (args.wce and args.pretrained is None):
        emb += 'wce'
    elif (args.wce and args.pretrained):
        emb += args.pretrained+'+wce'

    print("emb:", emb)

    return emb



def get_representation(args):

    print("calculating representation...")


    optimized = False

    # set model and dataset
    method_name = f'[{args.learner}:{args.dataset}]:->'

    #set representation form

    # solo is when we project the doc, we represent it, in the 
    # underlying pretrained embedding space - with three options 
    # as to how that space is computed: 1) weighted, 2) avg, 3) summary
    if (args.mix == 'solo'):
        method_name += f'{args.pretrained}:{args.dataset_emb_comp}'
    # cat is when we concatenate the doc representation in the
    # underlying pretrained embedding space with the tfidf vectors - 
    # we have the same three options for the dataset embedding representation
    elif (args.mix == 'cat'):
        method_name += f'{args.vtype}+{args.pretrained}:{args.dataset_emb_comp}'
    # dot is when we project the tfidf vectors into the underlying
    # pretrained embedding space using matrix multiplication, i.e. dot product
    # we have the same three options for the dataset embedding representation computation
    elif (args.mix == 'dot'):
        method_name += f'{args.vtype}->{args.pretrained}:{args.dataset_emb_comp}'
    # vmode is when we simply use the frequency vector representation (TF-IDF or Count)
    # as the dataset representation into the model
    elif (args.mix == 'vmode'):
        #method_name += f'{args.vtype}:{MAX_VOCAB_SIZE}.({args.pretrained}'
        method_name += f'{args.vtype}[{args.pretrained}]'
    # lsa is when we use SVD (aka LSA) to reduce the number of featrues from 
    # the vectorized data set, LSA is a form of dimensionality reduction
    elif (args.mix == 'lsa'):
        method_name += f'{args.vtype}->LSA/SVD.({args.pretrained})'

    #
    # set optimized field to true if its a neural model 
    # and we are tuning (fine-tuning) it or if its an ML
    # model and we are optimizing the prameters ofr best results
    #

    if (args.learner in NEURAL_MODELS and args.tunable) or (args.learner in ML_MODELS and args.optimc):
        method_name += ':(opt)'
        optimized = True
    else:
        method_name += ':(def)'
    
    print("method_name:", method_name)

    return method_name, optimized



if __name__ == '__main__':

    available_datasets = LCDataset.dataset_available

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification Testing')
    
    parser.add_argument('--dataset', required=True, type=str, default='20newsgroups', metavar='N', help=f'dataset, one in {available_datasets}')
    
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str', help=f'path where to load the pickled dataset from')
    
    parser.add_argument('--logfile', type=str, default='../log/ml_classify.test', metavar='N', help='path to the application log file')
    
    parser.add_argument('--learner', type=str, default='svm', metavar='N', help=f'learner (svm, lr, or nb)')
    
    parser.add_argument('--vtype', type=str, default='tfidf', metavar='N', help=f'dataset base vectorization strategy, in [tfidf, count]')
                        
    parser.add_argument('--mix', type=str, default='solo', metavar='N', help=f'way to prepare the embeddings, in [vmode, solo, cat, dot, lsa]. NB presumes --pretrained is set')

    parser.add_argument('--cm', action='store_true', default=False, help=f'create confusion matrix for underlying model')
                             
    parser.add_argument('--optimc', action='store_true', default=False, help='optimize the model using relevant models params')

    parser.add_argument('--wce', action='store_true', default=False, help='Use Word Calss Embeddings (supervised)')

    parser.add_argument('--force', action='store_true', default=False,
                    help='force the execution of the experiment even if a log already exists')
    
    parser.add_argument('--ngram-size', type=int, default=2, metavar='int',
                        help='ngram parameter into vectorization routines (TFIDFVectorizer)')
    
    parser.add_argument('--pretrained', type=str, default=None, metavar='glove|word2vec|fasttext|bert|roberta|llama',
                        help='pretrained embeddings, use "glove", "word2vec", "fasttext", "bert", "roberta", or "llama" (default None)')

    parser.add_argument('--dataset-emb-comp', type=str, default='avg', metavar='weighted|avg|summary',
                        help='how to compute dataset embedding representation form, one of "weighted", "avg", or "summary (cls)" (default weighted)')
    
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
                        help=f'directory to BERT pretrained vectors (NB used only with --pretrained bert)')

    parser.add_argument('--roberta-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to RoBERTa pretrained vectors (NB used only with --pretrained roberta)')
    
    parser.add_argument('--llama-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to LLaMA pretrained vectors (NB used only with --pretrained llama)')
    
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
    already_modelled, vtype, learner, pretrained, embeddings, emb_path, mix, representation, logfile, optimized = initialize_ml_testing(args)

    # check to see if model params have been computed already
    if (already_modelled) and not (args.force):
        print(f'Assertion warning: model {representation} with embeddings {embeddings} for {args.dataset} already calculated.')
        print("Run with --force option to override, exiting...")
        exit(0)

    print("executing model...")

    #
    # run the model using classifiy_data() method and the specified parameters
    #
    classify_data(
        dataset=args.dataset, 
        vtype=vtype,
        embeddings=embeddings,
        embedding_path=emb_path,
        representation=representation,
        optimized=optimized,
        logfile=logfile, 
        args=args
        )
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------