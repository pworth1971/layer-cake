import numpy as np
import argparse
from time import time

from scipy.sparse import csr_matrix, csc_matrix

from sklearn.decomposition import TruncatedSVD

from data.lc_dataset import LCDataset

#from embedding import supervised

from util.common import SystemResources, NEURAL_MODELS, ML_MODELS

from data.lc_dataset import LCDataset, loadpt_data, MAX_VOCAB_SIZE
from model.classification import run_model

from util.csv_log import CSVLog

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import torch

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
    avg_embeddings_train=None, avg_embeddings_test=None, summary_embeddings_train=None, summary_embeddings_test=None, dataset_embedding_type='weighted', mix='solo', supervised=False):
    
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

    
    print("summary_embeddings_train:", type(summary_embeddings_train), summary_embeddings_train.shape)
    print("summary_embeddings_test:", type(summary_embeddings_test), summary_embeddings_test.shape)

    print("mix:", mix)
    print("dataset_embedding_type:", dataset_embedding_type)

    # build vocabulary numpy array
    #vocabulary = np.asarray(list(zip(*sorted(vocab.items(), key=lambda x: x[1])))[0])
    #print("vocabulary:", type(vocabulary), vocabulary.shape)

    pretrained_embeddings = []                  # List to store pretrained embeddings

    if (pretrained in ['word2vec', 'glove', 'fasttext', 'bert', 'roberta', 'llama']) or supervised:
        
        print("setting up pretrained embeddings...")

        #P = pretrained_vectors_dictionary.extract(vocabulary).numpy()
        #print("pretrained_vectors_dictionary: ", type(pretrained_vectors_dictionary), {pretrained_vectors_dictionary.shape})

        pretrained_embeddings.append(pretrained_vectors_dictionary)
        print(f'pretrained embeddings count after loading pretrained embeddings: {len(pretrained_embeddings[0])}')

        """         
        word_list = get_word_list(word2index, out_of_vocabulary)
        weights = pretrained.extract(word_list)
        pretrained_embeddings.append(weights)
        print('\t[pretrained-matrix]', weights.shape)
        del pretrained
        """

        
        if supervised:
            # add supervised logic here
            pass    
            


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

    # return pretrained_embeddings, sup_range
    return X_train, X_test




# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# classify_data(): Core processing function
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def classify_data(dataset='20newsgrouops', vtype='tfidf', embeddings=None, embedding_path=None, method=None, logfile=None, args=None):
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
    
    #
    # Load the dataset and the associated (pretrained) embedding structures
    # to be fed into the model
    #                                                          
    Xtr, Xte, y_train_sparse, y_test_sparse, target_names, class_type, embedding_vocab_matrix, Xtr_weighted_embeddings, Xte_weighted_embeddings, \
        Xtr_avg_embeddings, Xte_avg_embeddings, Xtr_summary_embeddings, Xte_summary_embeddings = loadpt_data(
                                                                                                    dataset=dataset,                                # Dataset name
                                                                                                    vtype=args.vtype,                               # Vectorization type
                                                                                                    pretrained=args.pretrained,                     # pretrained embeddings type
                                                                                                    embedding_path=embedding_path,                  # path to pretrained embeddings
                                                                                                    emb_type=embedding_type                         # embedding type (word or token)
                                                                                                    )                                                
    print("Tokenized data loaded.")
 
    print("Xtr:", type(Xtr), Xtr.shape)
    print("Xte:", type(Xte), Xte.shape)

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

    data = None

    if args.pretrained is None:        # no embeddings in this case
        sup_tend = 0
    else:                              # embeddings are present
        tinit = time()

        print("building the embeddings...")

        # Generate embeddings
        X_train, X_test = gen_embeddings(
            X_train=Xtr,
            y_train=y_train, 
            X_test=Xte, 
            dataset=dataset,
            pretrained=pretrained_embeddings,
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

    dims = X_train.shape[1] if data is not None else 0

    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=method_name, optimized=optimized, dimensions=dims, measure='final-te-macro-F1', value=Mf1, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=method_name, optimized=optimized, dimensions=dims, measure='final-te-micro-F1', value=mf1, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=method_name, optimized=optimized, dimensions=dims, measure='te-accuracy', value=acc, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=method_name, optimized=optimized, dimensions=dims, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=method_name, optimized=optimized, dimensions=dims, measure='te-precision', value=precision, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=method_name, optimized=optimized, dimensions=dims, measure='te-recall', value=recall, timelapse=tend)
    logfile.insert(dataset=args.dataset, class_type=class_type, model=args.learner, embeddings=embeddings, representation=method_name, optimized=optimized, dimensions=dims, measure='te-jacard-index', value=j_index, timelapse=tend)

    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------------------------------------------
def initialize_ml_testing(args):

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

    if (args.pretrained == 'bert'):
        emb_path = args.bert_path
    elif args.pretrained == 'roberta':
        emb_path = args.roberta_path
    elif args.pretrained == 'glove':
        emb_path = args.glove_path
    elif args.pretrained == 'word2vec':
        emb_path = args.word2vec_path
    elif args.pretrained == 'fasttext':
        emb_path = args.fasttext_path
    elif args.pretrained == 'llama':
        emb_path = args.llama_path

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
            'representation',
            'optimized',
            'dimensions'
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

    # set defauklt system params
    ml_logger.set_default('os', system.get_os())
    ml_logger.set_default('cpus', system.get_cpu_details())
    ml_logger.set_default('mem', system.get_total_mem())
    ml_logger.set_default('gpus', system.get_gpu_summary())

    #
    # normalize data fields - these are NA for ML models
    #
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

def get_embeddings(args):

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
        method_name += f'{args.vtype}:{MAX_VOCAB_SIZE}'
    # lsa is when we use SVD (aka LSA) to reduce the number of featrues from 
    # the vectorized data set, LSA is a form of dimensionality reduction
    elif (args.mix == 'lsa'):
        method_name += f'{args.vtype}->LSA/SVD'

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
    already_modelled, vtype, learner, pretrained, embeddings, emb_path, mix, method_name, logfile, optimized = initialize_ml_testing(args)

    # check to see if model params have been computed already
    if (already_modelled) and not (args.force):
        print(f'Assertion warning: model {method_name} with embeddings {embeddings} for {args.dataset} already calculated.')
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
        method=method_name,
        optimized=optimized,
        logfile=logfile, 
        args=args
        )
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------