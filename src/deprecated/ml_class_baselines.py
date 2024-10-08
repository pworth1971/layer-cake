import argparse
from time import time

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC 
from sklearn.preprocessing import MinMaxScaler

from util.multilabel_classifier import MLClassifier
from util.metrics import evaluation
from util.common import *
from data.lc_dataset import LCDataset

from embedding.supervised import get_supervised_embeddings


VECTOR_CACHE = '../.vector_cache'
NUM_JOBS = -1          # important to manage CUDA memory allocation
#NUM_JOBS = 40          # for rcv1 dataset which has 101 classes, too many to support in parallel


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

    print('--- run_model() ---')
    tinit = time()

    print("classification_type:", classification_type)
    print("estimator:", estimator)
    print("mode:", mode)
    
    print("Xtr", type(Xtr), Xtr.shape)
    print("ytr", type(ytr), ytr.shape)
    print("Xte", type(Xte), Xte.shape)
    print("yte", type(yte), yte.shape)

    #
    # Setup the parameter grid for LinearSVC and LogisticRegression optimization
    #
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
        elif estimator==GaussianNB:
            param_grid = {
                #'var_smoothing': np.logspace(0,-9, num=100)                        # Range of var_smoothing values
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],                          # Range of var_smoothing values
                'priors': [None]                                                    # Prior probabilities of the classes
            }       
        else:
            print("Unsupported estimator, exiting...")
            return
    
    # --------------------------------------------------------------------------------------------
    # Normalize data to be non-negative if using Naive Bayes Multinomial model so as
    # to remove negative values from the data. This is a requirement for the MultinomialNB model.
    # NB: This model has problems with BERT and LLAMA embeddings for some reason
    #
    if estimator==MultinomialNB:
        scaler = MinMaxScaler()
        
        Xtr = scaler.fit_transform(todense(Xtr))
        Xte = scaler.transform(todense(Xte))

        Xtr = tosparse(Xtr)
        Xte = tosparse(Xte)
    # --------------------------------------------------------------------------------------------

    print("param_grid:", param_grid)
    cv = 5

    if classification_type == 'multilabel':                 # multi-label case

        print("------- multi-label case -------")
    
        # set up the esimator params based upon the model type
        if estimator==LinearSVC:
            cls = MLClassifier(n_jobs=NUM_JOBS, estimator=estimator, dual='auto', class_weight='balanced', verbose=False, max_iter=1000)
        elif estimator==LogisticRegression:
            cls = MLClassifier(n_jobs=NUM_JOBS, estimator=estimator, dual=False, class_weight='balanced', verbose=False, solver='lbfgs', max_iter=1000)
        elif estimator==MultinomialNB:
            cls = MLClassifier(n_jobs=NUM_JOBS, estimator=estimator, fit_prior=False, class_prior=None)
        elif estimator==GaussianNB:
            Xtr = todense(Xtr)
            Xte = todense(Xte)
            cls = MLClassifier(n_jobs=NUM_JOBS, estimator=estimator)
        else:
            print("ERR: unsupported estimator.")
            return

        cls.fit(Xtr, todense(ytr), param_grid=param_grid, cv=cv)
        yte_ = cls.predict(Xte)
        
        print("predictions (yte_):", type(yte_), yte_.shape)
        print("actuals (yte):", type(yte), yte.shape)
        
        Mf1, mf1, acc, h_loss, precision, recall, j_index = evaluation(
            tosparse(yte), 
            tosparse(yte_), 
            classification_type
            )

    else:                           # single label case     

        print("------- single label case -------")      

        # set up the esimator params based upon the model type
        if estimator==LinearSVC:
            cls = estimator(dual='auto', class_weight='balanced', verbose=False, max_iter=1000)
        elif estimator==LogisticRegression:
            cls = estimator(dual=False, class_weight='balanced', verbose=False, solver='lbfgs', max_iter=1000)
        elif estimator==MultinomialNB:
            cls = estimator(fit_prior=False, class_prior=None)
        elif estimator==GaussianNB:
            Xtr = Xtr.toarray()
            Xte = Xte.toarray()
            cls = estimator()
        else:
            print("ERR: unsupported estimator.")
            return

        # Print the estimator type and parameters
        print(f"Estimator type: {cls.__class__.__name__}")
        print(f"Estimator params: {cls.get_params()}")

        cls = GridSearchCV(cls, param_grid, cv=5, n_jobs=NUM_JOBS, scoring=scoring) if optimizeC else cls
        cls.fit(Xtr, ytr)

        yte_ = cls.predict(Xte)
        print("predictions (yte_):", type(yte_), yte_.shape)
        print("actuals (yte):", type(yte), yte.shape)
        
        Mf1, mf1, acc, h_loss, precision, recall, j_index = evaluation(yte, yte_, classification_type)

    tend = time() - tinit
    
    return Mf1, mf1, acc, h_loss, precision, recall, j_index, tend

# --------------------------------------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------------------------------------------------------------
# classify()
# 
# Main function to handle model training and evaluation. Takes command line arguments (args) and uses them to 
# configure and initiate text classification experiments, including data loading, model configuration, training, and 
# evaluation. Initializes logging and sets up experiment configurations based on input parameters (in args and other). 
# Supports different classification scenarios (binary or multilabel), different models (SVM, LR, NB) and different feature 
# preparation (TF-IDF, BERT, GloVe, etc.), and evaluates the performance using custom metrics and logs the results.
#
# -----------------------------------------------------------------------------------------------------------------------------------   

def classify(args, learner, pretrained, pretrained_vectors, supervised, logfile, cpus, mem, gpus):

    print("\t-- classify() -- ")

    print("args:", type(args), args)

    vocabsize = len(word2index) + len(out_of_vocabulary)
    print("vocabsize:", {vocabsize})

    ytr, yte = dataset.devel_target, dataset.test_target
    print("dev_target (ytr):", type(ytr), ytr.shape)
    print("test_target (yte):", type(yte), yte.shape)

    labels = dataset.get_labels()                           # retrieve labels
    label_names = dataset.get_label_names()                 # retrieve label names
    print("labels:", labels)
    print("label_names:", label_names)

    Xtr, Xte = dataset.vectorize()
    print("Xtr:", type(Xtr), Xtr.shape)
    print("Xte:", type(Xte), Xte.shape)     

    if (args.pretrained is None) and (args.supervised == False):        # no embeddings in this case
        sup_tend = 0
    else:                                                               # embeddings are present
        tinit = time()

        print("building the embeddings...")
 
        _, F = embedding_matrix(
            dataset, 
            pretrained=pretrained, 
            pretrained_vectors=pretrained_vectors,
            supervised=supervised, 
            )

        print("before matrix multiplication; Xtr, Xte:\n", type(Xtr), Xtr.shape, type(Xte), Xte.shape)
        
        Xtr = Xtr.dot(F)
        Xte = Xte.dot(F)

        print("after matrix multiplication; Xtr, Xte:\n", type(Xtr), Xtr.shape, type(Xte), Xte.shape)

        sup_tend = time() - tinit
    
    print('final matrix types (Xtr, ytr, Xte, yte):\n', type(Xtr), type(ytr), type(Xte), type(yte)) 
    print('final matrix shapes (Xtr, ytr, Xte, yte):\n', Xtr.shape, ytr.shape, Xte.shape, yte.shape)

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

    
    logfile.add_layered_row(tunable=False, measure='final-te-macro-F1', value=Mf1, timelapse=tend, cpus=cpus, mem=mem, gpus=gpus)
    logfile.add_layered_row(tunable=False, measure='final-te-micro-F1', value=mf1, timelapse=tend, cpus=cpus, mem=mem, gpus=gpus)
    logfile.add_layered_row(tunable=False, measure='te-accuracy', value=acc, timelapse=tend, cpus=cpus, mem=mem, gpus=gpus)
    logfile.add_layered_row(tunable=False, measure='te-hamming-loss', value=h_loss, timelapse=tend, cpus=cpus, mem=mem, gpus=gpus)
    logfile.add_layered_row(tunable=False, measure='te-precision', value=precision, timelapse=tend, cpus=cpus, mem=mem, gpus=gpus)
    logfile.add_layered_row(tunable=False, measure='te-recall', value=recall, timelapse=tend, cpus=cpus, mem=mem, gpus=gpus)
    logfile.add_layered_row(tunable=False, measure='te-jacard-index', value=j_index, timelapse=tend, cpus=cpus, mem=mem, gpus=gpus)

    """
    logfile.add_layered_row(tunable=False, measure='final-te-macro-F1', value=Mf1, timelapse=tend)
    logfile.add_layered_row(tunable=False, measure='final-te-micro-F1', value=mf1, timelapse=tend)
    logfile.add_layered_row(tunable=False, measure='te-accuracy', value=acc, timelapse=tend)
    logfile.add_layered_row(tunable=False, measure='te-hamming-loss', value=h_loss, timelapse=tend)
    logfile.add_layered_row(tunable=False, measure='te-precision', value=precision, timelapse=tend)
    logfile.add_layered_row(tunable=False, measure='te-recall', value=recall, timelapse=tend)
    logfile.add_layered_row(tunable=False, measure='te-jacard-index', value=j_index, timelapse=tend)
    """



# -------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    available_datasets = LCDataset.dataset_available

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification with Embeddings')
    
    parser.add_argument('--dataset', type=str, default='20newsgroups', metavar='N',
                        help=f'dataset, one in {LCDataset.dataset_available}')
    
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str',
                        help=f'path where to load the pickled dataset from')
    
    parser.add_argument('--log-file', type=str, default='../log/svm_baseline.test', metavar='N', help='path to the application log file')
    
    parser.add_argument('--learner', type=str, default='svm', metavar='N', 
                        help=f'learner (svm, lr, or nb)')
    
    parser.add_argument('--mode', type=str, default='tfidf', metavar='N',
                        help=f'mode, in [tfidf, count]')

    parser.add_argument('--pretrained', type=str, default=None, metavar='glove|word2vec|fasttext|bert|llama',
                        help='pretrained embeddings, use "glove", "word2vec", "fasttext", "bert", or "llama" (default None)')
    
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
                        help=f'directory to BERT pretrained vectors, used only with --pretrained bert')
    
    parser.add_argument('--llama-path', type=str, default=VECTOR_CACHE,
                        metavar='PATH',
                        help=f'directory to LLaMA pretrained vectors, used only with --pretrained llama')

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
    
    parser.add_argument('--batch-file', type=str, default=None, metavar='str',
                        help='path to the config file used for batch processing of multiple experiments')

    args = parser.parse_args()

    print("\n ---------------------------- Layer Cake: ML baseline classification code: main() ----------------------------")

    print("args:", type(args), args)

    # single run case
    if (args.batch_file is None):                        # running single command 

        print("single run processing...")

        # check dataset
        assert args.dataset in available_datasets, \
            f'unknown dataset {args.dataset}'

        # check mode
        assert args.mode in ['tfidf', 'count'], 'unknown mode'
    
        # initialize log file and run params
        already_modelled, vtype, learner, pretrained, embeddings, emb_path, supervised, method_name, logfile = initialize(args)

        # check to see if model params have been computed already
        if (already_modelled) and not (args.force):
            print(f'Assertion warning: model {method_name} with embeddings {embeddings}, pretrained == {pretrained} and wc_supervised == {args.supervised} for {args.dataset} already calculated.')
            print("Run with --force option to override, returning...")
            exit(0)
            
        cpus, mem, gpus = get_system_resources()

        print("new model, loading embeddings...")
        pretrained, pretrained_vectors, pretrained_vocab = load_pretrained_embeddings(embeddings, args)           

        print("loading LCDataset", {args.dataset}, "...")
        dataset = LCDataset.load(
            dataset_name=args.dataset, 
            vectorization_type=vtype,                   # TFIDF or Count
            base_pickle_path=args.pickle_dir
            ).show()        
    
        word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, pretrained_vectors)

        # classify the dataset
        classify(
            args, 
            learner, 
            pretrained, 
            pretrained_vectors, 
            supervised, 
            logfile,
            cpus,
            mem,
            gpus
            )
    else: 

        #
        # we are in batch mode so we build the array of args Namespace 
        # arguments for all of the batch runs from config file
        #
        print("batch processing, reading from file:", args.batch_file)

        #args = vars(parser.parse_args([]))  # Get default values as a dictionary

        #all_args = parse_arguments(opt.batch_file)

        #parse_arguments(opt.batch_file, args)

        """
        for opt in all_args:
            print("opt:", type(opt), opt)
        """

        # get batch config params
        configurations = parse_config_file(args.batch_file, parser)

        line_num = 1

        last_config = None
        pretrained = False
        pretrained_vector = None
        dataset = None

        cpus, mem, gpus = get_system_resources()
        
        for current_config in configurations:

            # roll these two argument params over to all current_configs
            #current_config.device = opt.device
            current_config.batch_file = args.batch_file

            print(f'\n\t--------------------- processing batch configuration file line #: {line_num} ---------------------')

            print(f'current_config: {type(current_config)}: {current_config}')
            print(f'last_config: {type(last_config)}: {last_config}')

            # -------------------------------------------------------------------------------------------------------------
            # check argument parameters
            # -------------------------------------------------------------------------------------------------------------
            if (current_config.dataset not in available_datasets):
                print(f'unknown dataset in config file line # {line_num}', current_config['dataset'])
                line_num += 1
                continue
        
            if (current_config.mode not in ['tfidf', 'count']):
                print(f'unknown mode in config file line # {line_num}', current_config['mode'])
                line_num += 1
                continue
        
            """
            if current_config.pickle_dir:
                current_config.pickle_path = join(current_config.pickle_dir, current_config.dataset + '.pickle')
            """
            # -------------------------------------------------------------------------------------------------------------

            already_modelled = False

            # initialize log file and run params
            already_modelled, vtype, learner, pretrained, embeddings, emb_path, supervised, method_name, logfile = initialize(args)

            # check to see if model params have been computed already
            if (already_modelled) and not (current_config.force):
                print(f'Assertion warning: model {method_name} with embeddings {embeddings}, pretrained == {pretrained} and wc_supervised == {current_config.supervised} for {current_config.dataset} already calculated.')
                print("Run with --force option to override, continuing...")
                line_num += 1
                continue

            # initialize embeddings if need be
            if 'pretrained' in current_config and (not last_config or current_config.pretrained != last_config.pretrained):
                print(f"loading pretrained embeddings: {current_config.pretrained}")
                pretrained, pretrained_vectors = load_pretrained_embeddings(current_config.pretrained, current_config)

            # initialize dataset if need be
            if 'dataset' in current_config and (not last_config or current_config.dataset != last_config.dataset):
                print(f"initializing dataset: {current_config.dataset}")

                dataset = LCDataset.load(dataset_name=current_config.dataset, vectorization_type=vtype, base_pickle_path=current_config.pickle_dir).show()
                word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, pretrained_vectors)

            # run layer_cake
            classify(
                current_config,                         # current_onfig is already a Namespace
                learner, 
                pretrained, 
                pretrained_vectors, 
                supervised, 
                logfile, 
                cpus,
                mem,
                gpus
                )
            
            last_config = current_config  # Update last_config to current for next iteration check

            line_num += 1

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------