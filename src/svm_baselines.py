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

from embedding.pretrained import GloVe
from embedding.supervised import get_supervised_embeddings


# --------------------------------------------------------------------------------------------------------
# cls_performance: 
# 
# Trains a classifier using SVM or logistic regression, optionally performing 
# hyperparameter tuning via grid search.
# --------------------------------------------------------------------------------------------------------
def cls_performance(Xtr, ytr, Xte, yte, classification_type, optimizeC=True, estimator=LinearSVC, class_weight='balanced', mode='tfidf'):
    
    print('training learner...')
    
    tinit = time()
    
    print("-- Xtr, ytr, Xte, yte shapes --")
    print(Xtr.shape, ytr.shape, Xte.shape, yte.shape)
    print()

    param_grid = {'C': np.logspace(-3, 3, 7)} if optimizeC else None
    cv = 5
    
    if classification_type == 'multilabel':
        print("multi-label classification...")
        
        cls = MLSVC(n_jobs=-1, estimator=estimator, class_weight=class_weight, verbose=True)
        cls.fit(Xtr, _todense(ytr), param_grid=param_grid, cv=cv)
        yte_ = cls.predict(Xte)

        Mf1, mf1, acc = evaluation(_tosparse(yte), _tosparse(yte_), classification_type)
    else:
        print("single label classification...")
        
        #cls = estimator(class_weight=class_weight, dual=True, max_iter=10000)             
        
        cls = estimator(class_weight=class_weight, dual="auto")                             
        cls = GridSearchCV(cls, param_grid, cv=cv, n_jobs=-1) if optimizeC else cls

        print("fitting Xtr and ytr", {Xtr.shape}, {ytr.shape})
        XtrArr = np.asarray(Xtr)
        ytrArr = np.asarray(ytr)
        XteArr = np.asarray(Xte)
        print("Xtr, ytr, Xte as arrays: ", {XtrArr.shape}, {ytrArr.shape}, {XteArr.shape})

        if mode in ['tfidf']:        
            cls.fit(Xtr, ytr)
            yte_ = cls.predict(Xte)       
        else: 
            cls.fit(XtrArr, ytrArr)
            yte_ = cls.predict(XteArr)

        Mf1, mf1, acc = evaluation(yte, yte_, classification_type)

    tend = time() - tinit

    return Mf1, mf1, acc, tend



# --------------------------------------------------------------------------------------------------------
# embedding_matrix: 
# 
# Constructs embedding matrices either from pre-trained embeddings (like GloVe), supervised embeddings, 
# or a combination of both.
# --------------------------------------------------------------------------------------------------------
def embedding_matrix(dataset, pretrained=False, supervised=False):

    assert pretrained or supervised, 'useless call without requiring pretrained and/or supervised embeddings'
    
    vocabulary = dataset.vocabulary
    vocabulary = np.asarray(list(zip(*sorted(vocabulary.items(), key=lambda x: x[1])))[0])

    print()
    print('[embedding matrix]')
    
    pretrained_embeddings = []

    if pretrained:

        print('\t[pretrained-matrix: GloVe]')
        pretrained = GloVe()
        P = pretrained.extract(vocabulary).numpy()

        print("P.shape: ", {P.shape})

        pretrained_embeddings.append(P)
        print(f'[GloVe pretrained embeddings count: {len(pretrained_embeddings[0])}')

        del pretrained

    if supervised:
        
        print('\t[supervised-matrix]')
        Xtr, _ = dataset.vectorize()
        Ytr = dataset.devel_labelmatrix

        print(Xtr.shape, Ytr.shape)

        S = get_supervised_embeddings(Xtr, Ytr)

        print("S.shape: ", {S.shape})
                
        pretrained_embeddings.append(S)
        print(f'supervised word-class embeddings count:{len(pretrained_embeddings[1])}')

    pretrained_embeddings = np.hstack(pretrained_embeddings)

    print("-- after np.hstack(): pretrained_embeddings shape: ", {pretrained_embeddings.shape})
    print()

    return vocabulary, pretrained_embeddings

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
    print("--- svm_baseline::main() ---")
        
    # set up model type
    learner = LinearSVC if args.learner == 'svm' else LogisticRegression
    learner_name = 'SVM' if args.learner == 'svm' else 'LR'
    
    #
    # initialize layered log file with core settings
    #
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

    supervised = False
    if args.mode in ['sup', 'bert-sup', 'glove-sup']:
        supervised = True

    print("initializing logfile embeddings value to:", {embeddings})
    
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

    print("loading dataset...:", {args.dataset})

    dataset = Dataset.load(
        dataset_name=args.dataset,
        pickle_path=os.path.join(args.pickle_dir, 
        f'{args.dataset}.pickle'))

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

    if mode in ['bert', 'bert-sup']:                                # load best model and get document embeddings for the dataset
        
        print("loading BERT embeddings...")
    
        bert_filename = os.path.join(args.embedding_dir, f'{args.dataset}_BERTembeddings_{args.combine_strategy}.pickle')
    
        if file.exists(bert_filename) and not args.force_embeddings:
            print('Loading pre-computed BERT document embeddings')
            with open(bert_filename, mode='rb') as inputfile:
                NLMtr, NLMte = pickle.load(inputfile)
        else:
            print('Computing BERT document embeddings')
            model = CustomRepresentationModel('bert', os.path.join(args.model_dir, args.dataset, 'best_model'))

            NLMtr = model.encode_sentences(dataset.devel_raw, combine_strategy=args.combine_strategy,
                                           batch_size=args.batch_size)
            NLMte = model.encode_sentences(dataset.test_raw, combine_strategy=args.combine_strategy,
                                           batch_size=args.batch_size)

            with open(bert_filename, mode='wb') as outputfile:
                pickle.dump((NLMtr, NLMte), outputfile)

    if args.mode in ['tfidf', 'stw', 'bert']:
        sup_tend = 0
    else:
        tinit = time()
        pretrained, supervised = False, False
        if args.mode in ['sup', 'bert-sup']:
            supervised = True
        elif args.mode == 'glove':
            pretrained = True
        elif args.mode == 'glove-sup':
            pretrained, supervised = True, True
        _, F = embedding_matrix(dataset, pretrained=pretrained, supervised=supervised)
        
        Xtr = Xtr.dot(F)    
        Xte = Xte.dot(F)
        
        sup_tend = time() - tinit

    # concatenating documents vectors from indexing with those from BERT model
    if mode == 'bert':
        Xtr = NLMtr
        Xte = NLMte
    elif mode == 'bert-sup':
        Xtr = np.hstack((Xtr, NLMtr))
        Xte = np.hstack((Xte, NLMte))

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

    logfile.add_layered_row(measure='te-macro-F1', value=Mf1, timelapse=tend)
    logfile.add_layered_row(measure='te-micro-F1', value=mf1, timelapse=tend)
    logfile.add_layered_row(measure='te-accuracy', value=acc, timelapse=tend)

    print('Done!')


def _todense(y):
    return y.toarray() if issparse(y) else y


def _tosparse(y):
    return y if issparse(y) else csr_matrix(y)


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification with Embeddings')
    
    parser.add_argument('--dataset', type=str, default='20newsgroups', metavar='N',
                        help=f'dataset, one in {Dataset.dataset_available}')
    
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str',
                        help=f'path where to load the pickled dataset from')
    
    parser.add_argument('--log-file', type=str, default='../log/log.csv', metavar='N', help='path to the log csv file')
    
    parser.add_argument('--learner', type=str, default='svm', metavar='N', help=f'learner (svm or lr)')
    
    parser.add_argument('--mode', type=str, default='tfidf', metavar='N',
                        help=f'mode, in tfidf, stw, sup, glove, glove-sup, bert, bert-sup')
    
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
    
    parser.add_argument('--optimc', action='store_true', default=False, help='optimize the C parameter in the SVM')
    
    parser.add_argument('--balanced', action='store_true', default=False, help='class weight balanced')
    
    parser.add_argument('--combine-strategy', default=None, type=str,
                        help='Method to determine BERT document embeddings.'
                             'No value takes the [CLS] embedding.'
                             '"mean" makes the mean of token embeddings.')
    
    parser.add_argument('--embedding-dir', type=str, default='../.vector_cache', metavar='str',
                        help=f'path where to load and save BERT document embeddings')
    
    parser.add_argument('--model-dir', type=str, default='../models', metavar='str',
                        help=f'path where the BERT model is stored. Dataset name is added')
    
    parser.add_argument('--force-embeddings', action='store_true', default=False,
                        help='force the computation of embeddings even if a precomputed version is available')
    
    parser.add_argument('--batch-size', type=int, default=512, metavar='int',
                        help='batch size for computation of BERT document embeddings')
    
    parser.add_argument('--force', action='store_true', default=False,
                        help='force the execution of the experiment even if a log already exists')
    
    args = parser.parse_args()
    
    assert args.mode in ['tfidf', 'sup', 'glove', 'glove-sup', 'stw', 'bert', 'bert-sup'], 'unknown mode'
    assert args.mode != 'stw' or args.tsr in ['ig', 'pmi', 'gr', 'chi', 'rf', 'cw'], 'unknown tsr'
    assert args.stwmode in ['wave', 'ave', 'max'], 'unknown stw-mode'
    assert args.learner in ['svm', 'lr'], 'unknown learner'
    assert args.combine_strategy in [None, 'mean'], 'unknown combine strategy'

    if args.combine_strategy is None:
        args.combine_strategy = 0

    main(args)
