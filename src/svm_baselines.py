import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import argparse
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from scipy.sparse import issparse, csr_matrix

from supervised_term_weighting.supervised_vectorizer import TSRweighting
from supervised_term_weighting.tsr_functions import *

from model.CustomRepresentationLearning import CustomRepresentationModel

from util import file
from util.multilabelsvm import MLSVC
from util.metrics import evaluation

from util.csv_log import CSVLog
from util.common import *

from data.dataset import *

from embedding.supervised import get_supervised_embeddings


"""
Defintition of a Python script for text classification using various embedding techniques and machine learning models, 
structured to be highly modular and configurable via command line arguments. It supports multiple modes of operation, 
including simple term frequency (TF-IDF), supervised term weighting, and embedding-based models such as GloVe and 
BERT. 

The script uses argparse to configure various options like dataset selection, learner type (SVM or 
Logistic Regression), embedding modes, optimization flags, and more. Sample Command Line Arguments to Call 
the Code. Sample command line argument calls:


- Basic TF-IDF with SVM:
'python script_name.py --dataset rcv1 --learner svm --mode tfidf'

- Supervised Term Weighting with Logistic Regression:
'python script_name.py --dataset rcv1 --learner lr --mode stw --tsr ig --stwmode wave --optimc'

- Using GloVe Embeddings with SVM and Optimization:
'python script_name.py --dataset rcv1 --learner svm --mode glove --optimc'

- BERT Embeddings Combined with SVM for Document Classification:
'python script_name.py --dataset rcv1 --learner svm --mode bert --combine-strategy mean --batch-size 256'

- Forcing Re-computation of BERT Embeddings:
'python script_name.py --dataset rcv1 --learner svm --mode bert --force-embeddings'
--------------------------------------------------------------------------------------------------------
"""



# --------------------------------------------------------------------------------------------------------
# cls_performance: 
# 
# Trains a classifier using SVM or logistic regression, optionally performing 
# hyperparameter tuning via grid search.
# --------------------------------------------------------------------------------------------------------
def cls_performance(Xtr, ytr, Xte, yte, classification_type, optimizeC=True, estimator=LinearSVC, class_weight='balanced', mode='tfidf'):
    
    print()
    print('........ cls_performance() ........')
    
    tinit = time()
    
    print("-- Xtr, ytr, Xte, yte shapes --")
    print(Xtr.shape, ytr.shape, Xte.shape, yte.shape)
    print()

    param_grid = {'C': np.logspace(-3, 3, 7)} if optimizeC else None
    cv = 5
    
    if classification_type == 'multilabel':
        print("---------***** multi-label classification *****---------")
        
        cls = MLSVC(n_jobs=-1, estimator=estimator, class_weight=class_weight, verbose=True)
        cls.fit(Xtr, _todense(ytr), param_grid=param_grid, cv=cv)
        yte_ = cls.predict(Xte)

        Mf1, mf1, acc = evaluation(_tosparse(yte), _tosparse(yte_), classification_type)
    else:
        print("---------***** single label classification *****---------")
        
        #cls = estimator(class_weight=class_weight, dual=True, max_iter=10000)             
        
        cls = estimator(class_weight=class_weight, dual="auto")                             
        cls = GridSearchCV(cls, param_grid, cv=cv, n_jobs=-1) if optimizeC else cls

        print("fitting Xtr and ytr", {Xtr.shape}, {ytr.shape})
        
        if mode in ['tfidf']:        
            cls.fit(Xtr, ytr)
            yte_ = cls.predict(Xte)       
        else: 
            """
            # convert to arrays
            XtrArr = np.asarray(Xtr)
            ytrArr = np.asarray(ytr)
            XteArr = np.asarray(Xte)
            print("Xtr, ytr, Xte as arrays: ", {XtrArr.shape}, {ytrArr.shape}, {XteArr.shape})

            cls.fit(XtrArr, ytrArr)
            yte_ = cls.predict(XteArr)
            """
            cls.fit(Xtr, ytr)
            yte_ = cls.predict(Xte)
            

        Mf1, mf1, acc = evaluation(yte, yte_, classification_type)

    tend = time() - tinit

    return Mf1, mf1, acc, tend

# --------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------
# tsr: 
#
# Maps string identifiers to term specificity ranking (TSR) functions used for supervised term weighting.
# --------------------------------------------------------------------------------------------------------

def tsr(name):
    name = name.lower()
    if name == 'ig':
        return information_gain
    elif name == 'pmi':
        return pointwise_mutual_information
    elif name == 'gr':
        return gain_ratio
    elif name == 'chi':
        return chi_square
    elif name == 'rf':
        return relevance_frequency
    elif name == 'cw':
        return conf_weight
    else:
        raise ValueError(f'unknown function {name}')

# --------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------
# embedding_matrix: 
# 
# Constructs embedding matrices either from pre-trained embeddings (like GloVe), supervised embeddings, 
# or a combination of both.
# -----------------------------------------------------------------------------------------------------------------------------------

def embedding_matrix(dataset, pretrained=False, pretrained_type=None, supervised=False, vector_cache='../.vector_cache'):

    print()
    print('----------------------------------------- svm_baseline::embedding_matrix() -----------------------------------------')
    
    assert pretrained or supervised, 'useless call without requiring pretrained and/or supervised embeddings'
    
    vocabulary = dataset.vocabulary
    vocabulary = np.asarray(list(zip(*sorted(vocabulary.items(), key=lambda x: x[1])))[0])

    pretrained_embeddings = []
    
    print("pretrained:", {pretrained})
    print("supervised:", {supervised})

    if pretrained:
        
        print()

        if (pretrained_type in ['glove', 'glove-sup']):
            print('\t[pretrained-matrix: GloVe]')
            pretrained = GloVe(path=vector_cache)

        elif (pretrained_type in ['bert', 'bert-sup']):
            print('\t[pretrained-matrix: BERT]')
            pretrained = BERT(model_name=DEFAULT_BERT_PRETRAINED_MODEL, cache_dir=vector_cache, dataset_name=dataset)
        
        elif (pretrained_type in ['word2vec', 'word2vec-sup']):
            print('\t[pretrained-matrix: Word2Vec]')
            pretrained = Word2Vec(path=vector_cache)
        
        elif (pretrained_type in ['fasttext', 'fasttext-sup']):
            print('\t[pretrained-matrix: FastText]')
            pretrained = FastText(path=vector_cache)

        P = pretrained.extract(vocabulary).numpy()
        print("P.shape: ", {P.shape})

        pretrained_embeddings.append(P)
        print(f'pretrained embeddings count: {len(pretrained_embeddings[0])}')


    if supervised:
        
        print()
        print('\t[supervised-matrix]')

        Xtr, _ = dataset.vectorize()
        Ytr = dataset.devel_labelmatrix

        print(Xtr.shape, Ytr.shape)

        S = get_supervised_embeddings(Xtr, Ytr)

        print("S.shape: ", {S.shape})
                
        pretrained_embeddings.append(S)
        print(f'supervised word-class embeddings count: {len(pretrained_embeddings[1])}')

    pretrained_embeddings = np.hstack(pretrained_embeddings)

    print()
    print("-- after np.hstack(): pretrained_embeddings shape: ", {pretrained_embeddings.shape})
    print()

    return vocabulary, pretrained_embeddings

# -----------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------
# embedding_matrix2()
#
# adapted from layer_cake primary code
#
# -----------------------------------------------------------------------------------------------------------------------------------
def embedding_matrix2(dataset, pretrained, supervised, vocabsize, word2index, out_of_vocabulary, nozscore, supervised_method, max_label_space):

    print()
    print('------------------------- embedding_matrix2() -------------------------')

    pretrained_embeddings = None
    sup_range = None
    
    if pretrained or supervised:
        pretrained_embeddings = []

        if pretrained is not None:
            word_list = get_word_list(word2index, out_of_vocabulary)
            weights = pretrained.extract(word_list)
            pretrained_embeddings.append(weights)
            print('\t[pretrained-matrix]', weights.shape)
            del pretrained

        if supervised:
            Xtr, _ = dataset.vectorize()
            Ytr = dataset.devel_labelmatrix
            F = get_supervised_embeddings(Xtr, Ytr,
                                          method=supervised_method,
                                          max_label_space=max_label_space,
                                          dozscore=(not nozscore))
            num_missing_rows = vocabsize - F.shape[0]
            F = np.vstack((F, np.zeros(shape=(num_missing_rows, F.shape[1]))))
            F = torch.from_numpy(F).float()
            print('\t[supervised-matrix]', F.shape)

            offset = 0
            if pretrained_embeddings:
                offset = pretrained_embeddings[0].shape[1]
            sup_range = [offset, offset + F.shape[1]]
            pretrained_embeddings.append(F)

        pretrained_embeddings = torch.cat(pretrained_embeddings, dim=1)
        print('\t[final pretrained_embeddings]\n\t', pretrained_embeddings.shape)

    return pretrained_embeddings, sup_range

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

    print()
    print()
    print("------------------------------------------- #####-- MAIN(ARGS) --##### -------------------------------------------")
    print()
    print("........   svm_baseline::main(args).  ........ ")

    # set up model type
    learner = LinearSVC if args.learner == 'svm' else LogisticRegression
    learner_name = 'SVM' if args.learner == 'svm' else 'LR'
    
    mode = args.mode
    if args.mode == 'stw':                              # supervised term weighting (stw)
        mode += f'-{args.tsr}-{args.stwmode}'
    
    method_name = f'{learner_name}-{mode}-{"opC" if args.optimc else "default"}'

    pretrained = False
    embeddings ='none'

    if args.mode in ['bert', 'bert-sup']:
        pretrained = True
        embeddings = 'bert'
    elif args.mode in ['glove', 'glove-sup']:
        pretrained = True
        embeddings = 'glove'
    elif args.mode in ['word2vec', 'word2vec-sup']:
        pretrained = True
        embeddings = 'word2vec'
    elif args.mode in ['fasttext', 'fasttext-sup']:
        pretrained = True
        embeddings = 'fasttext'
   
    supervised = False

    if args.mode in ['sup', 'bert-sup', 'glove-sup', 'word2vec-sup', 'fasttext-sup']:
        supervised = True

    print("loading pretrained embeddings...")

    # load pre-trained embeddings (vectors) 
    pretrained, pretrained_vector = load_pretrained_embeddings(
        args.dataset,
        embeddings,
        args
        )                

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

    #assert not logfile.already_calculated() or args.force, f'baseline {method_name} for {args.dataset} already calculated'

    print("loading dataset ", {args.dataset})
    dataset = Dataset.load(dataset_name=args.dataset, pickle_path=args.pickle_dir).show()
    word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, pretrained_vector)

    """
    print("train / test data split...")
    val_size = min(int(len(devel_index) * .2), 20000)                   # dataset split tr/val/test

    train_index, val_index, ytr, yval = train_test_split(
        devel_index, dataset.devel_target, test_size=val_size, random_state=opt.seed, shuffle=True
    )
    
    yte = dataset.test_target
    """

    vocabsize = len(word2index) + len(out_of_vocabulary)
    print("vocabsize:", {vocabsize})

    class_weight = 'balanced' if args.balanced else None
    print(f'running with class_weight={class_weight}')

    # tfidf = TfidfVectorizer(min_df=5)
    # Xtr = tfidf.fit_transform(dataset.devel_raw)
    # Xte = tfidf.transform(dataset.test_raw)
    ytr, yte = dataset.devel_target, dataset.test_target

    if args.mode == 'stw':                                          # supervised term weighting config
        print('Supervised Term Weighting')
        coocurrence = CountVectorizer(vocabulary=dataset.vocabulary)
        Ctr = coocurrence.transform(dataset.devel_raw)
        Cte = coocurrence.transform(dataset.test_raw)
        stw = TSRweighting(tsr_function=tsr(args.tsr), global_policy=args.stwmode)
        
        Xtr = stw.fit_transform(Ctr, dataset.devel_labelmatrix)
        Xte = stw.transform(Cte)
    else:
        Xtr, Xte = dataset.vectorize()

    print("Xtr, Xte:", Xtr.shape, Xte.shape)

    if args.mode in ['tfidf', 'stw']:
        sup_tend = 0
    else:
        tinit = time()
        """
        pretrained, supervised = False, False
        
        if args.mode in ['sup', 'glove-sup', 'bert-sup', 'word2vec-sup', 'fasttext-sup']:
            supervised = True
        elif args.mode in ['glove', 'bert', 'word2vec', 'fasttext', 'glove-sup', 'bert-sup', 'word2vec-sup', 'fasttext-sup']:
            pretrained = True
        """

        #
        # build the embedding matrix
        #     
        """ 
        _, F = embedding_matrix(
            dataset, 
            pretrained=pretrained, 
            pretrained_type=embeddings,
            supervised=supervised, 
            vector_cache='../.vector_cache'
            )
        """

        print("building the embeddings...")

        F, sup_range = embedding_matrix2(
            dataset, 
            pretrained_vector, 
            supervised, 
            vocabsize, 
            word2index, 
            out_of_vocabulary,
            args.nozscore,
            args.supervised_method,
            args.max_label_space)

        #Xtr = Xtr.dot(F)    
        #Xte = Xte.dot(F)
        
        sup_tend = time() - tinit

    
    print(Xtr.shape, Xte.shape)

    #
    # conversion due to np.matrix deprecation in numpy library
    #
    """
    print("matrix array conversion (numpy suppoort)...")
    
    XtrArr = np.asarray(Xtr)
    ytrArr = np.asarray(ytr)
    XteArr = np.asarray(Xte)
    yteArr = np.asarray(yte)

    print(XtrArr.shape, XteArr.shape)
    """

    Mf1, mf1, acc, tend = cls_performance(
        Xtr, 
        ytr, 
        Xte, 
        yte, 
        dataset.classification_type, 
        args.optimc, 
        learner,
        class_weight=class_weight, 
        mode=args.mode)
    
    """
    Mf1, mf1, acc, tend = cls_performance(
        XtrArr, 
        ytrArr, 
        XteArr, 
        yteArr, 
        dataset.classification_type, 
        args.optimc, 
        learner,
        class_weight=class_weight, 
        mode=args.mode)
    """

    tend += sup_tend

    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-macro-F1', value=Mf1, timelapse=tend)
    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-micro-F1', value=mf1, timelapse=tend)
    logfile.add_layered_row(epoch=0, tunable=False, run=0, measure='te-accuracy', value=acc, timelapse=tend)


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
    
    parser.add_argument('--learner', type=str, default='svm', metavar='N', help=f'learner (svm or lr)')
    
    parser.add_argument('--mode', type=str, default='tfidf', metavar='N',
                        help=f'mode, in [tfidf, stw, sup, glove, glove-sup, bert, bert-sup, word2vec, word2vec-sup, fasttext, fasttext-sup]')
    
    parser.add_argument('--stwmode', type=str, default='wave', metavar='N',
                        help=f'mode in which the term relevance will be merged (wave, ave, max). Only for --mode stw. '
                             f'Default "wave"')
    
    parser.add_argument('--tsr', type=str, default='ig', metavar='TSR',
                        help=f'indicates the accronym of the TSR function to use in supervised term weighting '
                             f'(only if --mode stw). Valid functions are '
                             f'IG (information gain), '
                             f'PMI (pointwise mutual information) '
                             f'GR (gain ratio) '
                             f'CHI (chi-square) '
                             f'RF (relevance frequency) '
                             f'CW (ConfWeight)')
    
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='use supervised embeddings')
                        
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
                             
    parser.add_argument('--optimc', action='store_true', default=False, help='optimize the C parameter in the SVM')
    
    parser.add_argument('--balanced', action='store_true', default=False, help='class weight balanced')
    
    parser.add_argument('--combine-strategy', default=None, type=str,
                        help='Method to determine BERT document embeddings.'
                             'No value takes the [CLS] embedding.'
                             '"mean" makes the mean of token embeddings.')
    
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
    
    """                        
    parser.add_argument('--model-dir', type=str, default='../models', metavar='str',
                        help=f'path where the BERT model is stored. Dataset name is added')
    """
    
    args = parser.parse_args()
    
    assert args.mode in ['tfidf', 'stw', 'sup', 'glove', 'glove-sup', 'word2vec', 'word2vec-sup', 'fasttext', 'fasttext-sup', 'bert', 'bert-sup'], 'unknown mode'
    assert args.mode != 'stw' or args.tsr in ['ig', 'pmi', 'gr', 'chi', 'rf', 'cw'], 'unknown tsr'
    assert args.stwmode in ['wave', 'ave', 'max'], 'unknown stw-mode'
    assert args.learner in ['svm', 'lr'], 'unknown learner'
    assert args.combine_strategy in [None, 'mean'], 'unknown combine strategy'

    if args.combine_strategy is None:
        args.combine_strategy = 0

    main(args)
