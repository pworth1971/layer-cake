import warnings
from sklearn.exceptions import ConvergenceWarning

import argparse
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

from scipy.sparse import issparse, csr_matrix

from model.CustomRepresentationLearning import CustomRepresentationModel

from util import file
from util.multilabel_classifier import MLClassifier
from util.metrics import evaluation

from util.csv_log import CSVLog
from util.common import *

from data.dataset import *

from embedding.supervised import get_supervised_embeddings



# --------------------------------------------------------------------------------------------------------
# run_classification_model: 
# 
# Trains a classifier using SVM or logistic regression, optionally performing 
# hyperparameter tuning via grid search.
# --------------------------------------------------------------------------------------------------------

def run_classification_model(Xtr, ytr, Xte, yte, classification_type, optimizeC=True, estimator=LinearSVC, 
                    mode='tfidf', pretrained=False, supervised=False, dataset_name='unknown'):

    print('\n--- run_classification_model() ---')
    print('training learner...')
    
    tinit = time()

    print("classification_type: ", classification_type)
    print("estimator: ", estimator)
    print("mode: ", mode)
    
    print("Xtr", type(Xtr), Xtr.shape)
    print("ytr", type(ytr), ytr.shape)
    print("Xte", type(Xte), Xte.shape)
    print("yte", type(yte), yte.shape)

    # Setup the parameter grid
    param_grid = {'C': np.logspace(-3, 3, 7)} if optimizeC else None
    print("param_grid:", param_grid)
    cv = 5

    if classification_type == 'multilabel':
        print("-- multi-label --")
        cls = MLClassifier(n_jobs=-1, dataset_name=dataset_name, pretrained=pretrained, supervised=supervised, estimator=estimator, verbose=False)
        cls.fit(Xtr, _todense(ytr), param_grid=param_grid, cv=cv)
        yte_ = cls.predict(Xte)
        #print("predictions (yte_):", type(yte_), yte_)
        #print("actuals (yte):", type(yte), yte)
        Mf1, mf1, acc = evaluation(_tosparse(yte), _tosparse(yte_), classification_type)

    else:
        print("-- single label --")      
        cls = estimator(dual=False)
        cls = GridSearchCV(cls, param_grid, cv=5, n_jobs=-1) if optimizeC else cls
        cls.fit(Xtr, ytr)
        yte_ = cls.predict(Xte)
        #print("predictions (yte_):", type(yte_), yte_)
        #print("actuals (yte):", type(yte), yte)
        Mf1, mf1, acc = evaluation(yte, yte_, classification_type)

    tend = time() - tinit
    return Mf1, mf1, acc, tend

# --------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------
# embedding_matrix: 
# 
# Constructs embedding matrices either from pre-trained embeddings (like GloVe), supervised embeddings, 
# or a combination of both.
# -----------------------------------------------------------------------------------------------------------------------------------

def embedding_matrix(dataset, pretrained=False, pretrained_type=None, supervised=False, emb_path='../.vector_cache'):

    print('----------------------------------------- svm_baseline::embedding_matrix() -----------------------------------------')
    
    assert pretrained or supervised, 'useless call without requiring pretrained and/or supervised embeddings'
    
    vocabulary = dataset.vocabulary
    vocabulary = np.asarray(list(zip(*sorted(vocabulary.items(), key=lambda x: x[1])))[0])

    pretrained_embeddings = []
    
    print("pretrained:", {pretrained})
    print("supervised:", {supervised})

    if pretrained:

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
        print("pretrained.shape: ", {P.shape})

        pretrained_embeddings.append(P)
        print(f'pretrained embeddings count: {len(pretrained_embeddings[0])}')


    if supervised:
    
        print('\t[supervised-matrix]')

        Xtr, _ = dataset.vectorize()
        Ytr = dataset.devel_labelmatrix

        print(Xtr.shape, Ytr.shape)

        S = get_supervised_embeddings(Xtr, Ytr)

        print("supervised.shape: ", {S.shape})
                
        pretrained_embeddings.append(S)
        print(f'supervised word-class embeddings count: {len(pretrained_embeddings[1])}')

    pretrained_embeddings = np.hstack(pretrained_embeddings)

    print("after np.hstack(): pretrained_embeddings shape:", {pretrained_embeddings.shape})

    return vocabulary, pretrained_embeddings

# -----------------------------------------------------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------------------------
# Main Function (main):
# 
# Initializes logging and sets up experiment configurations based on command line arguments.
# Loads datasets and applies vector transformations or embeddings as specified by the mode.
# Handles different classification scenarios (binary or multilabel) and different embedding 
# methods (TF-IDF, BERT, GloVe, etc.). Evaluates the performance using custom metrics and logs 
# the results.
# --------------------------------------------------------------------------------------------------------

def main(args):

    # disable warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print()
    print("------------------------------------------- #####-- MAIN(ARGS) --##### -------------------------------------------")
    print("........   ml_class_baselines::main(args).  ........ ")

    # set up model type
    learner = LinearSVC if args.learner == 'svm' else LogisticRegression
    learner_name = 'SVM' if args.learner == 'svm' else 'LR'

    print("learner: ", {learner_name})
    
    mode = args.mode
    if args.mode == 'stw':                              # supervised term weighting (stw)
        mode += f'-{args.tsr}-{args.stwmode}'
    
    method_name = f'{learner_name}-{mode}-{"opC" if args.optimc else "default"}'

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
    print()
    
    print("loading pretrained embeddings...")
    pretrained, pretrained_vector = load_pretrained_embeddings(embeddings, args)                

    embeddings_log_val ='none'

    if pretrained:
        embeddings_log_val = args.embedding_dir

    #print("initializing logfile embeddings value to:", {embeddings})
    logfile = init_layered_logfile_svm(                             
        logfile=args.log_file,
        method_name=method_name, 
        dataset=args.dataset, 
        model=learner_name,
        pretrained=pretrained, 
        embeddings=embeddings,
        supervised=supervised
        )

    # TODO: fix this assertion with new log file format
    #assert not logfile.already_calculated() or args.force, f'baseline {method_name} for {args.dataset} already calculated'

    print("loading dataset ", {args.dataset})
    dataset = Dataset.load(dataset_name=args.dataset, pickle_path=args.pickle_dir).show()
    word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, pretrained_vector)

    vocabsize = len(word2index) + len(out_of_vocabulary)
    print("vocabsize:", {vocabsize})

    ytr, yte = dataset.devel_target, dataset.test_target
    #print("dev_target (ytr):", type(ytr), ytr)
    #print("test_target (yte):", type(yte), yte)

    Xtr, Xte = dataset.vectorize()
    #print("Xtr:", type(Xtr), Xtr)
    #print("Xte:", type(Xte), Xte)

    if args.mode in ['tfidf']:
        sup_tend = 0
    else:
        tinit = time()

        print("building the embeddings...")
 
        _, F = embedding_matrix(
            dataset, 
            pretrained=pretrained, 
            pretrained_type=embeddings,
            supervised=supervised, 
            emb_path=emb_path
            )

        Xtr = Xtr.dot(F)
        Xte = Xte.dot(F)
        
        # convert to arrays
        Xtr = np.asarray(Xtr)
        Xte = np.asarray(Xte)
        
        sup_tend = time() - tinit
     
    print('final matrix shapes (Xtr, ytr, Xte, yte):', Xtr.shape, ytr.shape, Xte.shape, yte.shape)

    Mf1, mf1, acc, tend = run_classification_model(
        Xtr, 
        ytr, 
        Xte, 
        yte, 
        dataset.classification_type, 
        args.optimc, 
        learner,
        mode=args.mode,
        pretrained=pretrained,
        supervised=supervised,
        dataset_name=args.dataset
        )
    
    tend += sup_tend

    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-macro-F1', value=Mf1, timelapse=tend)
    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-micro-F1', value=mf1, timelapse=tend)
    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-accuracy', value=acc, timelapse=tend)

# -------------------------------------------------------------------------------------------------------------------------------------------------



def _todense(y):
    return y.toarray() if issparse(y) else y


def _tosparse(y):
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
                        help=f'learner (svm or lr)')
    
    parser.add_argument('--mode', type=str, default='tfidf', metavar='N',
                        help=f'mode, in [tfidf, sup, glove, glove-sup, bert, bert-sup, word2vec, word2vec-sup, fasttext, fasttext-sup]')
    
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

    args = parser.parse_args()
    
    assert args.mode in ['tfidf', 'sup', 'glove', 'glove-sup', 'word2vec', 'word2vec-sup', 'fasttext', 'fasttext-sup', 'bert', 'bert-sup'], 'unknown mode'
    
    assert args.learner in ['svm', 'lr'], 'unknown learner'
    
    
    main(args)
