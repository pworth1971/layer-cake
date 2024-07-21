import warnings
import argparse
from time import time

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

from scipy.sparse import issparse, csr_matrix
from model.CustomRepresentationLearning import CustomRepresentationModel
from util import file
from util.multilabel_classifier import MLClassifier
from util.metrics import evaluation
from util.csv_log import CSVLog
from util.common import *
from data.dataset import *
from embedding.supervised import get_supervised_embeddings



def run_model(Xtr, ytr, Xte, yte, classification_type, optimizeC=True, estimator=LinearSVC, mode='tfidf', scoring='accuracy'):
    """
    Trains a classification model using SVM or Logistic Regression, performing hyperparameter tuning if specified
    using GridSearchCV. LinearSVC and LogisticRegression sklearn estimators are tested but technically should
    support any standard sklearn classifier. 
    
    Parameters:
    - Xtr, ytr: Training data features and labels.
    - Xte, yte: Test data features and labels.
    - classification_type: 'singlelabel' or 'multilabel' classification.
    - optimizeC: Boolean flag to determine if hyperparameter tuning is needed.
    - estimator: The machine learning estimator (model).
    - mode: The feature extraction mode (e.g., 'tfidf').
    - pretrained, supervised: Flags for using pretrained or supervised models.
    - dataset_name: Name of the dataset for identification.
    - scoring: Metric used for optimizing the model during GridSearch.
    """

    print('\n--- run_model() ---')
    tinit = time()

    print("classification_type: ", classification_type)
    print("estimator: ", estimator)
    print("mode: ", mode)
    
    print("Xtr", type(Xtr), Xtr.shape)
    print("ytr", type(ytr), ytr.shape)
    print("Xte", type(Xte), Xte.shape)
    print("yte", type(yte), yte.shape)

    # Setup the parameter grid
    if not optimizeC:
        param_grid = None
    else:
        if estimator==LinearSVC or estimator==LogisticRegression:
            #param_grid = {'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}
            param_grid = {'C': np.logspace(-3, 3, 7)}
        elif estimator==MultinomialNB:
            param_grid = {
                'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],           # Range of alpha values
                'fit_prior': [True, False]                                          # Whether to learn class prior probabilities
            }       
        else:
            print("Unsupported estimator, exiting...")
            return
    
    # Normalize data to be non-negative if using Naive Bayes model
    if estimator==MultinomialNB:
        scaler = MinMaxScaler()

        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

    print("param_grid:", param_grid)
    cv = 5

    if classification_type == 'multilabel':
        print("------- multi-label case -------")
    
        # set up the esimator params based upon teh model type
        if estimator==LinearSVC:
            cls = MLClassifier(n_jobs=-1, estimator=estimator, dual='auto', class_weight='balanced', verbose=False, max_iter=1000)
        elif estimator==LogisticRegression:
            cls = MLClassifier(n_jobs=-1, estimator=estimator, dual=False, class_weight='balanced', verbose=False, solver='saga', max_iter=1000)
        elif estimator==MultinomialNB:
            cls = MLClassifier(n_jobs=-1, estimator=estimator, fit_prior=False, class_prior=None)
        else:
            print("ERR: unsupported estimator.")
            return

        cls.fit(Xtr, _todense(ytr), param_grid=param_grid, cv=cv)
        yte_ = cls.predict(Xte)
        
        print("predictions (yte_):", type(yte_), yte_)
        print("actuals (yte):", type(yte), yte)
        
        Mf1, mf1, acc, h_loss, precision, recall, j_index = evaluation(
            _tosparse(yte), 
            _tosparse(yte_), 
            classification_type, 
            debug=True
            )

    else:
        print("------- single label case -------")      

        # set up the esimator params based upon teh model type
        if estimator==LinearSVC:
            cls = estimator(dual='auto', class_weight='balanced', verbose=False, max_iter=1000)
        elif estimator==LogisticRegression:
            cls = estimator(dual=False, class_weight='balanced', verbose=False, solver='saga', max_iter=1000)
        elif estimator==MultinomialNB:
            cls = estimator(fit_prior=True, class_prior=None)
        else:
            print("ERR: unsupported estimator.")
            return

        # Print the estimator type and parameters
        print(f"Estimator type: {cls.__class__.__name__}")
        print(f"Estimator params: {cls.get_params()}")

        #cls = estimator(dual=False)
        cls = GridSearchCV(cls, param_grid, cv=5, n_jobs=-1, scoring=scoring) if optimizeC else cls
        cls.fit(Xtr, ytr)

        yte_ = cls.predict(Xte)
        #print("predictions (yte_):", type(yte_), yte_)
        #print("actuals (yte):", type(yte), yte)
        Mf1, mf1, acc, h_loss, precision, recall, j_index = evaluation(yte, yte_, classification_type)

    tend = time() - tinit
    
    return Mf1, mf1, acc, h_loss, precision, recall, j_index, tend

# --------------------------------------------------------------------------------------------------------


def embedding_matrix(dataset, pretrained=False, pretrained_type=None, pretrained_vectors=None, supervised=False, emb_path='../.vector_cache'):
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

    pretrained_embeddings = []
    
    print("pretrained:", {pretrained})
    print("supervised:", {supervised})

    if pretrained and pretrained_vectors:

        #
        # NB: embeddings should be already loaded and sent in as a param in this case
        #

        print("pretrained and pretrained_vectors, extracting vocab and building pretrained embeddings...")
        
        """
        if (pretrained_type in ['glove', 'glove-sup']):
            print('\t[pretrained-matrix: GloVe]')
            pretrained = GloVe(path=emb_path)

        elif (pretrained_type in ['bert', 'bert-sup']):
            print('\t[pretrained-matrix: BERT]')
            pretrained = BERT(model_name=DEFAULT_BERT_PRETRAINED_MODEL, emb_path=emb_path)
        
        elif (pretrained_type in ['word2vec', 'word2vec-sup']):
            print('\t[pretrained-matrix: Word2Vec]')
            pretrained = Word2Vec(path=emb_path)
        
        elif (pretrained_type in ['fasttext', 'fasttext-sup']):
            print('\t[pretrained-matrix: FastText]')
            pretrained = FastText(path=emb_path)
        
        P = pretrained.extract(vocabulary).numpy()
        """

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



def main(args):
    """
    Main function to handle model training and evaluation. Processes command line arguments to 
    configure and initiate text classification experiments, including data loading, model configuration, 
    training, and evaluation. Initializes logging and sets up experiment configurations based on 
    command line arguments, loads datasets and applies vector transformations or embeddings as specified 
    by the mode, handles different classification scenarios (binary or multilabel) and different embedding 
    methods (TF-IDF, BERT, GloVe, etc.), and evaluates the performance using custom metrics and logs the results.
    
    Uses:
    - args: Command line arguments parsed using argparse.
    """

    print()
    print("------------------------------------------------------------------------------------ MAIN(ARGS) ------------------------------------------------------------------------------------")

    # Print the full command line
    print("Command line:", ' '.join(sys.argv))

    # set up model type
    if args.learner == 'svm':
        learner = LinearSVC
        learner_name = 'SVM' 
    elif args.learner == 'lr':
        learner = LogisticRegression
        learner_name = 'LR'
    elif args.learner == 'nb':
        learner = MultinomialNB
        learner_name = 'NB'
    else:
        print("** Unknown learner, possible values are svm, lr or nb **")
        return

    print("learner:", learner)
    print("learner_name: ", {learner_name})
    
    # disable warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    mode = args.mode
    
    if args.count:
        vtype = 'count'
    else:
        vtype = 'tfidf'             # default to tfidf

    method_name = f'{learner_name}-{mode}-{vtype}-{"opC" if args.optimc else "default"}'
    print("method_name: ", {method_name})

    pretrained = False
    embeddings ='none'
    emb_path = VECTOR_CACHE

    if args.mode in ['bert', 'bert-sup']:
        pretrained = True
        embeddings = 'bert'
        emb_path = args.bert_path
    elif args.mode in ['glove', 'glove-sup']:
        pretrained = True
        embeddings = 'glove'
        emb_path = args.glove_path
    elif args.mode in ['word2vec', 'word2vec-sup']:
        pretrained = True
        embeddings = 'word2vec'
        emb_path = args.word2vec_path
    elif args.mode in ['fasttext', 'fasttext-sup']:
        pretrained = True
        embeddings = 'fasttext'
        emb_path = args.fasttext_path
   
    print("emb_path: ", {emb_path})

    supervised = False

    if args.mode in ['sup', 'bert-sup', 'glove-sup', 'word2vec-sup', 'fasttext-sup']:
        supervised = True

    print("pretrained: ", {pretrained}, "; supervised: ", {supervised}, "; embeddings: ", {embeddings})

     #print("initializing logfile embeddings value to:", {embeddings})
    logfile = init_layered_baseline_logfile(                             
        logfile=args.log_file,
        method_name=method_name, 
        dataset=args.dataset, 
        model=learner_name,
        pretrained=pretrained, 
        embeddings=embeddings,
        supervised=supervised
        )

    # check to see if the model has been run before
    already_modelled = logfile.already_calculated(
        dataset=args.dataset,
        embeddings=embeddings,
        model=learner_name, 
        params=method_name,
        pretrained=pretrained, 
        wc_supervised=supervised
        )

    print("already_modelled: ", already_modelled)

    if (already_modelled) and not (args.force):
        print('Assertion warning: baseline {method_name} for {args.dataset} already calculated')
        print("run with --force option to override, exiting...")
        return
        
    print("new model, loading embeddings...")
    pretrained, pretrained_vector = load_pretrained_embeddings(embeddings, args)           

    

    print("loading dataset ", {args.dataset})
    
    #dataset = Dataset.load(dataset_name=args.dataset, vectorization_type=vtype, pickle_path=args.pickle_dir).show()
    
    # here we force the load and vectorization of the Dataset every time
    # to ensure we use the specified vectorization method (Count or TFIDF)
    dataset = Dataset.load(
        dataset_name=args.dataset, 
        vectorization_type=vtype, 
        pickle_path=None
        ).show()        
    
    word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, pretrained_vector)

    vocabsize = len(word2index) + len(out_of_vocabulary)
    print("vocabsize:", {vocabsize})

    ytr, yte = dataset.devel_target, dataset.test_target
    #print("dev_target (ytr):", type(ytr), ytr)
    #print("test_target (yte):", type(yte), yte)

    Xtr, Xte = dataset.vectorize()
    
    #print("Xtr:", type(Xtr), Xtr)
    #print("Xte:", type(Xte), Xte)     

    # convert to arrays if need be
    Xtr = _todense(Xtr)
    Xte = _todense(Xte)

    if args.mode in ['count', 'tfidf']:
        sup_tend = 0
    else:
        tinit = time()

        print("building the embeddings...")
 
        _, F = embedding_matrix(
            dataset, 
            pretrained=pretrained, 
            pretrained_type=embeddings,
            pretrained_vectors=pretrained_vector,
            supervised=supervised, 
            emb_path=emb_path
            )

        print("before matrix multiplication; Xtr, Xte:", type(Xtr), Xtr.shape, type(Xte), Xte.shape)
        
        Xtr = Xtr.dot(F)
        Xte = Xte.dot(F)

        print("after matrix multiplication; Xtr, Xte:", type(Xtr), Xtr.shape, type(Xte), Xte.shape)

        sup_tend = time() - tinit
    
    ytr = _todense(ytr)
    yte = _todense(yte)
    
    print('final matrix types (Xtr, ytr, Xte, yte):', type(Xtr), type(ytr), type(Xte), type(yte)) 
    print('final matrix shapes (Xtr, ytr, Xte, yte):', Xtr.shape, ytr.shape, Xte.shape, yte.shape)

    # run the model
    print("running model...")
    Mf1, mf1, acc, h_loss, precision, recall, j_index, tend = run_model(
        Xtr, 
        ytr, 
        Xte, 
        yte, 
        dataset.classification_type, 
        args.optimc, 
        learner,
        mode=args.mode, 
        scoring=args.scoring
        )
    
    tend += sup_tend

    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-macro-F1', value=Mf1, timelapse=tend)
    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-micro-F1', value=mf1, timelapse=tend)
    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-accuracy', value=acc, timelapse=tend)
    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-precision', value=precision, timelapse=tend)
    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-recall', value=recall, timelapse=tend)
    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-jacard-index', value=j_index, timelapse=tend)

# -------------------------------------------------------------------------------------------------------------------------------------------------


def _todense(y):
    """Convert sparse matrix to dense format as needed."""
    return y.toarray() if issparse(y) else y


def _tosparse(y):
    """Ensure matrix is in CSR format for efficient arithmetic operations."""
    return y if issparse(y) else csr_matrix(y)


# -------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification with Embeddings')
    
    parser.add_argument('--dataset', type=str, default='20newsgroups', metavar='N',
                        help=f'dataset, one in {Dataset.dataset_available}')
    
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str',
                        help=f'path where to load the pickled dataset from')
    
    parser.add_argument('--log-file', type=str, default='../log/svm_baseline.test', metavar='N', help='path to the application log file')
    
    parser.add_argument('--learner', type=str, default='svm', metavar='N', 
                        help=f'learner (svm, lr, or nb)')
    
    parser.add_argument('--mode', type=str, default='tfidf', metavar='N',
                        help=f'mode, in [tfidf, sup, glove, glove-sup, bert, bert-sup, word2vec, word2vec-sup, fasttext, fasttext-sup]')

    parser.add_argument('--count', action='store_true', default=False,
                        help='use CountVectorizer')
    
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='use supervised embeddings')
                        
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
                             
    parser.add_argument('--optimc', action='store_true', default=False, help='optimize the C parameter in the SVM')
    
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
                        help=f'directory to BERT pretrained vectors (e.g. bert-base-uncased-20newsgroups.pkl), used only with --pretrained bert')

    parser.add_argument('--force-embeddings', action='store_true', default=False,
                        help='force the computation of embeddings even if a precomputed version is available')
    
    parser.add_argument('--batch-size', type=int, default=512, metavar='int',
                        help='batch size for computation of BERT document embeddings')
    
    parser.add_argument('--force', action='store_true', default=False,
                        help='force the execution of the experiment even if a log already exists')

    parser.add_argument('--nozscore', action='store_true', default=False,
                        help='disables z-scoring form the computation of WCE')

    parser.add_argument('--max-label-space', type=int, default=300, metavar='int',
                        help='larger dimension allowed for the feature-label embedding (if larger, then PCA with this '
                             'number of components is applied (default 300)')

    parser.add_argument('--scoring', type=str, default='accuracy',
                        help=f'scoring parameter to GridSearchCV sklearn call. Must be one of sklearn scoring metricsd.')

    args = parser.parse_args()
    
    assert args.mode in ['tfidf', 'sup', 'glove', 'glove-sup', 'word2vec', 'word2vec-sup', 'fasttext', 'fasttext-sup', 'bert', 'bert-sup'], 'unknown mode'
    
    main(args)
