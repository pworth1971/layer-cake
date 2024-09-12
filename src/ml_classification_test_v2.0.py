import numpy as np
import argparse
from time import time
import os

from scipy.sparse import csr_matrix, csc_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from data.lc_dataset import LCDataset, MAX_VOCAB_SIZE, save_to_pickle, load_from_pickle
from util.common import initialize_logging, SystemResources

from model.classification import run_model


import warnings
warnings.filterwarnings('ignore')



#
# we assume everything runs from bin directory
#
PICKLE_DIR = '../pickles/'
VECTOR_CACHE = '../.vector_cache'
OUT_DIR = '../out/'

TEST_SIZE = 0.2


# -------------------------------------------------------------------------------------------------------------------
# gen_embeddings()
# -------------------------------------------------------------------------------------------------------------------
def gen_embeddings(X_train, y_train, X_test, dataset='bbc-news', pretrained=None, pretrained_vectors_dictionary=None, weighted_embeddings_train=None, weighted_embeddings_test=None, \
    avg_embeddings_train=None, avg_embeddings_test=None, summary_embeddings_train=None, summary_embeddings_test=None, dataset_embedding_type='weighted', mix='solo'):
    
    print("\n\tgenerating embeddings...")
        
    print('X_train:', type(X_train), X_train.shape)
    print("y_train:", type(y_train), y_train.shape)
    print('X_test:', type(X_test), X_test.shape)
    
    print("dataset:", dataset)
    
    print("pretrained:", pretrained)
    
    # should be a numpy array
    print("pretrained_vectors_dictionary:", type(pretrained_vectors_dictionary), "Shape:", pretrained_vectors_dictionary.shape)
    
    print("weighted_embeddings_train:", type(weighted_embeddings_train), weighted_embeddings_train.shape)
    print("weighted_embeddings_test:", type(weighted_embeddings_test), weighted_embeddings_test.shape)
    
    print("avg_embeddings_train:", type(avg_embeddings_train), avg_embeddings_train.shape)
    print("avg_embeddings_test:", type(avg_embeddings_test), avg_embeddings_test.shape)

    if (summary_embeddings_train is not None) and (summary_embeddings_test is not None):
        print("summary_embeddings_train:", type(summary_embeddings_train), summary_embeddings_train.shape)
        print("summary_embeddings_test:", type(summary_embeddings_test), summary_embeddings_test.shape)
    else:
        print("summary_embeddings_train:", summary_embeddings_train)
        print("summary_embeddings_test:", summary_embeddings_test)

    print("mix:", mix)
    print("dataset_embedding_type:", dataset_embedding_type)

    # build vocabulary numpy array
    #vocabulary = np.asarray(list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0])
    #print("vocabulary:", type(vocabulary), vocabulary.shape)

    pretrained_embeddings = []                  # List to store pretrained embeddings

    if pretrained in ['word2vec', 'glove', 'fasttext', 'bert', 'roberta', 'llama']:
        
        print("setting up pretrained embeddings...")

        #P = pretrained_vectors_dictionary.extract(vocabulary).numpy()
        #print("pretrained_vectors_dictionary: ", type(pretrained_vectors_dictionary), {pretrained_vectors_dictionary.shape})

        pretrained_embeddings.append(pretrained_vectors_dictionary)
        print(f'pretrained embeddings count after loading pretrained embeddings: {len(pretrained_embeddings[0])}')


    embedding_matrix = np.hstack(pretrained_embeddings)
    print("after np.hstack():", type(embedding_matrix), {embedding_matrix.shape})

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
            X_train = np.hstack([X_train.toarray(), weighted_embeddings_train])
            X_test = np.hstack([X_test.toarray(), weighted_embeddings_test])
        elif (dataset_embedding_type == 'avg'):
            X_train = np.hstack([X_train.toarray(), avg_embeddings_train])
            X_test = np.hstack([X_test.toarray(), avg_embeddings_test])
        elif (dataset_embedding_type == 'summary'):
            X_train = np.hstack([X_train.toarray(), summary_embeddings_train])
            X_test = np.hstack([X_test.toarray(), summary_embeddings_test])
        else:
            print(f"Unsupported dataset_embedding_type '{dataset_embedding_type}'")
            return None

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
        X_train = svd.fit_transform(X_train)
        X_test = svd.transform(X_test)

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



    
def loadpt_data(dataset, vtype='tfidf', pretrained=None, embedding_path=VECTOR_CACHE, emb_type='word'):

    print("loadpt_data():", dataset, PICKLE_DIR)

    #
    # load the dataset using appropriate tokenization method as dictated by pretrained embeddings
    #
    pickle_file_name=f'{dataset}_{vtype}_{pretrained}_{MAX_VOCAB_SIZE}_tokenized.pickle'

    print(f"Loading data set {dataset}...")

    pickle_file = PICKLE_DIR + pickle_file_name                                     
        
    #
    # we pick up the vectorized dataset along with the associated pretrained 
    # embedding matrices when e load the data - either from data files directly
    # if the first time parsing the dataset or from the pickled file if it exists
    # and the data has been cached for faster loading
    #
    if os.path.exists(pickle_file):                                                 # if the pickle file exists
        
        print(f"Loading tokenized data from '{pickle_file}'...")
        
        X_vectorized, y_sparse, target_names, class_type, embedding_vocab_matrix, weighted_embeddings, \
            avg_embeddings, summary_embeddings = load_from_pickle(pickle_file)

        return X_vectorized, y_sparse, target_names, class_type, embedding_vocab_matrix, weighted_embeddings, avg_embeddings, summary_embeddings

    else:
        print(f"'{pickle_file}' not found, loading {dataset}...")
        
        lcd = LCDataset(name=dataset)    

        lcd.initialize(
            vectorization_type=vtype,                   # vectorization type
            embedding_type=emb_type,                    # embedding type
            pretrained=pretrained,                      # pretrained embeddings
            pretrained_path=embedding_path              # path to embeddings
            )

        # Save the tokenized matrices to a pickle file
        save_to_pickle(
            lcd.X_vectorized,               # vectorized data
            lcd.y_sparse,                   # labels
            lcd.target_names,               # target names
            lcd.class_type,                 # class type (single-label or multi-label):
            lcd.embedding_vocab_matrix,     # vector representation of the dataset vocabulary
            lcd.weighted_embeddings,        # weighted avg embedding representation of dataset
            lcd.avg_embeddings,             # avg embedding representation of dataset
            lcd.summary_embeddings,         # summary embedding representation of dataset
            pickle_file)         

        return lcd.X_vectorized, lcd.y_sparse, lcd.target_names, lcd.class_type, lcd.embedding_vocab_matrix, \
            lcd.weighted_embeddings, lcd.avg_embeddings, lcd.summary_embeddings



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# classify_data(): Core processing function
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def classify_data(dataset='20newsgrouops', vtype='tfidf', pretrained_embeddings=None, embedding_path=None, method=None, args=None, logfile=None, system=None):
    """
    Core function for classifying text data using various configurations like embeddings, methods, and models.

    Parameters:
    - dataset (str): The name of the dataset to use (e.g., '20newsgroups', 'ohsumed').
    - vtype (str): The vectorization type (e.g., 'tfidf', 'count').
    - pretrained_embeddings (str or None): Specifies the type of pretrained embeddings to use (e.g., 'bert', 'llama'). If None, no embeddings are used.
    - embedding_path (str): Path to the pretrained embeddings file or directory.
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
    
    if (pretrained_embeddings in ['bert', 'roberta', 'llama']):
        embedding_type = 'token'
    else:
        embedding_type = 'word'
    
    print("pretrained_embeddings:", pretrained_embeddings)    
    print("embedding_type:", embedding_type)
    
    X, y, target_names, class_type, embedding_vocab_matrix, weighted_embeddings, avg_embeddings, summary_embeddings = loadpt_data(
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
    print("avg_embeddings:", type(avg_embeddings), avg_embeddings.shape)
    
    # summary embeddings are None for word embeddings (uses CLS token)
    if (summary_embeddings is not None):
        print("summary_embeddings:", type(summary_embeddings), summary_embeddings.shape)
    else:
        print("summary_embeddings: None")

    print("transforming labels...")
    if isinstance(y, (csr_matrix, csc_matrix)):
        y = y.toarray()  # Convert sparse matrix to dense array for multi-label tasks
    # Ensure y is in the correct format for classification type
    if class_type in ['singlelabel', 'single-label']:
        y = y.ravel()                       # Flatten y for single-label classification
    print("y after transformation:", type(y), y.shape)


    print("splitting data...")

    # Ensure all structures have the same number of samples
    assert X.shape[0] == y.shape[0] == weighted_embeddings.shape[0] == avg_embeddings.shape[0]

    if (summary_embeddings is not None):
        assert summary_embeddings.shape[0] == X.shape[0]  # Check only if summary_embeddings exist

        # Handle splitting differently if summary_embeddings is None
        split_data = train_test_split(
            X, y, weighted_embeddings, avg_embeddings, summary_embeddings,
            test_size=TEST_SIZE, random_state=44, shuffle=True
        )

        (X_train, X_test, y_train, y_test,
        weighted_embeddings_train, weighted_embeddings_test,
        avg_embeddings_train, avg_embeddings_test,
        summary_embeddings_train, summary_embeddings_test) = split_data

    else:
        split_data = train_test_split(
            X, y, weighted_embeddings, avg_embeddings,
            test_size=TEST_SIZE, random_state=44, shuffle=True
        )

        (X_train, X_test, y_train, y_test,
        weighted_embeddings_train, weighted_embeddings_test,
        avg_embeddings_train, avg_embeddings_test) = split_data
        summary_embeddings_train, summary_embeddings_test = None, None

    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
    
    print("y_train:", type(y_train), y_train.shape)
    print("y_test:", type(y_test), y_test.shape)

    #print("y_train:", y_train)
    #print("y_test:", y_test)

    print("weighted_embeddings_train:", type(weighted_embeddings_train), weighted_embeddings_train.shape)
    print("weighted_embeddings_test:", type(weighted_embeddings_test), weighted_embeddings_test.shape)

    print("avg_embeddings_train:", type(avg_embeddings_train), avg_embeddings_train.shape)
    print("avg_embeddings_test:", type(avg_embeddings_test), avg_embeddings_test.shape)

    if (summary_embeddings is not None):
        print("summary_embeddings_train:", type(summary_embeddings_train), summary_embeddings_train.shape)
        print("summary_embeddings_test:", type(summary_embeddings_test), summary_embeddings_test.shape)
    else:
        print("summary_embeddings_train: None")
        print("summary_embeddings_test: None")

    if args.pretrained is None:        # no embeddings in this case
        sup_tend = 0
    else:                              # embeddings are present
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
            avg_embeddings_train=avg_embeddings_train,
            avg_embeddings_test=avg_embeddings_test,
            summary_embeddings_train=summary_embeddings_train,
            summary_embeddings_test=summary_embeddings_test,
            dataset_embedding_type=args.dataset_emb_comp,
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
                        
    parser.add_argument('--mix', type=str, default='solo', metavar='N', help=f'way to prepare the embeddings, in [vmode, solo, cat, dot, lsa]. NB presumes --pretrained is set')

    parser.add_argument('--cm', action='store_true', default=False, help=f'create confusion matrix for underlying model')
                             
    parser.add_argument('--optimc', action='store_true', default=False, help='optimize the model using relevant models params')

    parser.add_argument('--force', action='store_true', default=False,
                    help='force the execution of the experiment even if a log already exists')
    
    parser.add_argument('--ngram-size', type=int, default=2, metavar='int',
                        help='ngram parameter into vectorization routines (TFIDFVectorizer)')
    
    parser.add_argument('--pretrained', type=str, default=None, metavar='glove|word2vec|fasttext|bert|roberta|llama',
                        help='pretrained embeddings, use "glove", "word2vec", "fasttext", "bert", "roberta", or "llama" (default None)')

    parser.add_argument('--dataset-emb-comp', type=str, default='weighted', metavar='weighted|avg|summary',
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
    already_modelled, vtype, learner, pretrained, embeddings, emb_path, mix, method_name, logfile = initialize_logging(args)

    # check to see if model params have been computed already
    if (already_modelled) and not (args.force):
        print(f'Assertion warning: model {method_name} with embeddings {embeddings}, pretrained == {pretrained} for {args.dataset} already calculated.')
        print("Run with --force option to override, returning...")
        exit(0)

    sys = SystemResources()
    print("system resources:", sys)

    classify_data(
        dataset=args.dataset, 
        vtype=vtype,
        pretrained_embeddings=embeddings,
        embedding_path=emb_path,
        method=method_name,
        args=args, 
        logfile=logfile, 
        system=SystemResources()
        )
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------