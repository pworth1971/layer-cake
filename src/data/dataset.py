import os,sys

from sklearn.datasets import get_data_home, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from data.jrcacquis_reader import fetch_jrcacquis, JRCAcquis_Document
from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1
from data.wipo_reader import fetch_WIPOgamma, WipoGammaDocument

import pickle
import numpy as np
from tqdm import tqdm
from os.path import join
import re


TEST_SIZE = 0.2
VAL_SIZE = 0.2

MIN_DF_COUNT = 5                    # minimum document frequency count for a term to be included in the vocabulary




def init_vectorizer(vtype='tfidf', custom_tokenizer=None):
    """
    Initializes a text vectorizer for transforming raw text into numerical representations
    (TF-IDF or Count-based) for NLP tasks.

    Parameters:
    ----------
    vtype : str, optional, default='tfidf'
        Specifies the type of vectorizer to use:
        - 'tfidf': Use `TfidfVectorizer` for term frequency-inverse document frequency representation.
        - 'count': Use `CountVectorizer` for raw frequency counts.

    custom_tokenizer : callable, optional, default=None
        A custom tokenizer function for tokenizing input text. If provided, the vectorizer
        will use this tokenizer instead of its default analyzer. Typically used for token-based models
        like BERT, where a specialized tokenizer is required.

    Returns:
    -------
    vectorizer : TfidfVectorizer or CountVectorizer
        A fitted vectorizer instance based on the specified `vtype` and tokenizer.

        - For `vtype='tfidf'`, returns an instance of `TfidfVectorizer`.
        - For `vtype='count'`, returns an instance of `CountVectorizer`.

    Notes:
    -----
    - If `custom_tokenizer` is provided, the vectorizer will not lowercase text or apply its default
      tokenization logic. Instead, it will rely on the custom tokenizer for tokenization.
    - The vectorizer ignores terms with a document frequency strictly lower than `MIN_DF_COUNT`.
    - Sublinear TF scaling is enabled by default for `TfidfVectorizer`.

    Example:
    -------
    # Initialize a TF-IDF vectorizer with a custom tokenizer
    tokenizer = lambda text: text.split()  # Example tokenizer
    vectorizer = init_vectorizer(vtype='tfidf', custom_tokenizer=tokenizer)

    # Initialize a Count vectorizer without a custom tokenizer
    vectorizer = init_vectorizer(vtype='count')
    """
    print(f'initializing vectorizer... vtype: {vtype}, custom_tokenizer: {custom_tokenizer}')

    vectorizer = None

    if custom_tokenizer is not None:                # if a custom tokenizer is provided (e.g. for transformer models)
        if vtype == 'tfidf':
            vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                sublinear_tf=True,                              # use sublinear TF scaling
                analyzer='word',                                # analyze the words
                lowercase=False,                                # dont lowercase the tokens
                tokenizer=custom_tokenizer                      # use the custom tokenizer
            )
        elif vtype == 'count':
            vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                analyzer='word',                                # analyze the words     
                lowercase=False,                                # dont lowercase the tokens
                tokenizer=custom_tokenizer                      # use the custom tokenizer
            )
    else:                                           # if no custom tokenizer is provided (e.g. for word based models like GloVe, Word2vec...)
        if vtype == 'tfidf':
            vectorizer = TfidfVectorizer(
                min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                sublinear_tf=True,                              # use sublinear TF scaling
                analyzer='word',                                # analyze the words
                lowercase=True                                  # dont lowercase the tokens
            )
        elif vtype == 'count':
            vectorizer = CountVectorizer(
                min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                analyzer='word',                                # analyze the words     
                lowercase=True                                  # dont lowercase the tokens
            )
    
    return vectorizer




class Dataset:

    dataset_available = {'reuters21578', '20newsgroups', 'ohsumed', 'rcv1', 'ohsumed', 'jrcall',
                         'wipo-sl-mg','wipo-ml-mg','wipo-sl-sc','wipo-ml-sc'}

    def __init__(self, dataset, vtype='tfidf', custom_tokenizer=None):

        assert dataset in Dataset.dataset_available, f'dataset {dataset} is not available'

        if dataset=='reuters21578':
            self._load_reuters()
        elif dataset == '20newsgroups':
            self._load_20news()
        elif dataset == 'rcv1':
            self._load_rcv1()
        elif dataset == 'ohsumed':
            self._load_ohsumed()
        elif dataset == 'jrcall':
            self._load_jrc(version='all')
        elif dataset == 'wipo-sl-mg':
            self._load_wipo('singlelabel', 'maingroup')
        elif dataset == 'wipo-ml-mg':
            self._load_wipo('multilabel', 'maingroup')
        elif dataset == 'wipo-sl-sc':
            self._load_wipo('singlelabel', 'subclass')
        elif dataset == 'wipo-ml-sc':
            self._load_wipo('multilabel', 'subclass')

        self.nC = self.devel_labelmatrix.shape[1]

        self._vectorizer = init_vectorizer(vtype=vtype, custom_tokenizer=custom_tokenizer)
        print("vectorizer:\n", self._vectorizer)

        self._vectorizer.fit(self.devel_raw)
        self.vocabulary = self._vectorizer.vocabulary_
        print("vocabulary:", type(self.vocabulary), len(self.vocabulary))

    def show(self):
        nTr_docs = len(self.devel_raw)
        nTe_docs = len(self.test_raw)
        nfeats = len(self._vectorizer.vocabulary_)
        nC = self.devel_labelmatrix.shape[1]
        nD=nTr_docs+nTe_docs
        print(f'{self.classification_type}, nD={nD}=({nTr_docs}+{nTe_docs}), nF={nfeats}, nC={nC}')
        return self

    def _load_reuters(self):
        data_path = os.path.join(get_data_home(), 'reuters21578')
        devel = fetch_reuters21578(subset='train', data_path=data_path)
        test = fetch_reuters21578(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(devel.target, test.target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix

    def _load_rcv1(self):
        data_path = '../datasets/RCV1-v2/unprocessed_corpus' #TODO: check when missing
        devel = fetch_RCV1(subset='train', data_path=data_path)
        test = fetch_RCV1(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(devel.target, test.target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix

    def _load_jrc(self, version):
        assert version in ['300','all'], 'allowed versions are "300" or "all"'
        data_path = "../datasets/JRC_Acquis_v3"
        tr_years=list(range(1986, 2006))
        te_years=[2006]
        if version=='300':
            training_docs, tr_cats = fetch_jrcacquis(data_path=data_path, years=tr_years, cat_threshold=1,most_frequent=300)
            test_docs, te_cats = fetch_jrcacquis(data_path=data_path, years=te_years, cat_filter=tr_cats)
        else:
            training_docs, tr_cats = fetch_jrcacquis(data_path=data_path, years=tr_years, cat_threshold=1)
            test_docs, te_cats = fetch_jrcacquis(data_path=data_path, years=te_years, cat_filter=tr_cats)
        print(f'load jrc-acquis (English) with {len(tr_cats)} tr categories ({len(te_cats)} te categories)')

        devel_data = JRCAcquis_Document.get_text(training_docs)
        test_data = JRCAcquis_Document.get_text(test_docs)
        devel_target = JRCAcquis_Document.get_target(training_docs)
        test_target = JRCAcquis_Document.get_target(test_docs)

        self.classification_type = 'multilabel'
        self.devel_raw, self.test_raw = mask_numbers(devel_data), mask_numbers(test_data)
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(devel_target, test_target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix

    def _load_ohsumed(self):
        data_path = os.path.join(get_data_home(), 'ohsumed50k')
        devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        test = fetch_ohsumed50k(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(devel.target, test.target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix

    def _load_20news(self):
        metadata = ('headers', 'footers', 'quotes')
        devel = fetch_20newsgroups(subset='train', remove=metadata)
        test = fetch_20newsgroups(subset='test', remove=metadata)
        self.classification_type = 'singlelabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_target, self.test_target = devel.target, test.target
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))

    def _load_fasttext_data(self,name):
        data_path='../datasets/fastText'
        self.classification_type = 'singlelabel'
        name=name.replace('-','_')
        train_file = join(data_path,f'{name}.train')
        assert os.path.exists(train_file), f'file {name} not found, please place the fasttext data in {data_path}' #' or specify the path' #todo
        self.devel_raw, self.devel_target = load_fasttext_format(train_file)
        self.test_raw, self.test_target = load_fasttext_format(join(data_path, f'{name}.test'))
        self.devel_raw = mask_numbers(self.devel_raw)
        self.test_raw = mask_numbers(self.test_raw)
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(self.devel_target.reshape(-1, 1), self.test_target.reshape(-1, 1))

    def _load_wipo(self, classmode, classlevel):
        assert classmode in {'singlelabel', 'multilabel'}, 'available class_mode are sl (single-label) or ml (multi-label)'
        data_path = '../datasets/WIPO/wipo-gamma/en'
        data_proc = '../datasets/WIPO-extracted'

        devel = fetch_WIPOgamma(subset='train', classification_level=classlevel, data_home=data_path, extracted_path=data_proc, text_fields=['abstract'])
        test  = fetch_WIPOgamma(subset='test', classification_level=classlevel, data_home=data_path, extracted_path=data_proc, text_fields=['abstract'])

        devel_data = [d.text for d in devel]
        test_data  = [d.text for d in test]
        self.devel_raw, self.test_raw = mask_numbers(devel_data), mask_numbers(test_data)

        self.classification_type = classmode
        if classmode== 'multilabel':
            devel_target = [d.all_labels for d in devel]
            test_target  = [d.all_labels for d in test]
            self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(devel_target, test_target)
            self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix
        else:
            devel_target = [d.main_label for d in devel]
            test_target  = [d.main_label for d in test]
            # only for labels with at least one training document
            class_id = {labelname:index for index,labelname in enumerate(sorted(set(devel_target)))}
            devel_target = np.array([class_id[id] for id in devel_target]).astype(int)
            test_target  = np.array([class_id.get(id,None) for id in test_target])
            if None in test_target:
                print(f'deleting {(test_target==None).sum()} test documents without valid categories')
                keep_pos = test_target!=None
                self.test_raw = (np.asarray(self.test_raw)[keep_pos]).tolist()
                test_target = test_target[keep_pos]
            test_target=test_target.astype(int)
            self.devel_target, self.test_target = devel_target, test_target
            self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(self.devel_target.reshape(-1, 1), self.test_target.reshape(-1, 1))


    def vectorize(self):
        print("vectorizing dataset...")
        if not hasattr(self, 'Xtr') or not hasattr(self, 'Xte'):
            self.Xtr = self._vectorizer.transform(self.devel_raw)
            self.Xte = self._vectorizer.transform(self.test_raw)
            self.Xtr.sort_indices()
            self.Xte.sort_indices()
        return self.Xtr, self.Xte

    """
    def analyzer(self):
        return self._vectorizer.build_analyzer()
    """

    def analyzer(self):
        """
        Returns the appropriate analyzer/tokenizer for the dataset.

        For word-based models (e.g., GloVe, Word2Vec, FastText), this method
        uses the default analyzer from the `TfidfVectorizer` or `CountVectorizer`.
        
        For token-based models (e.g., BERT, RoBERTa, DistilBERT), it uses the
        custom tokenizer provided during initialization.

        Returns:
        -------
        callable
            A tokenizer or analyzer function.
        """
        if hasattr(self._vectorizer, 'tokenizer') and self._vectorizer.tokenizer is not None:
            print("Using custom tokenizer from vectorizer.")
            return self._vectorizer.tokenizer  # Use the custom tokenizer (e.g., for BERT)
        else:
            print("Using default analyzer from vectorizer.")
            return self._vectorizer.build_analyzer()  # Use the default word-based analyzer



    @classmethod
    def load(cls, dataset_name, vtype='tfidf', pt_model=None, pickle_path=None):

        print(f'\n\tloading dataset: {dataset_name}, vtype: {vtype}, pt_type: {pt_model.get_type()}, pt_model: {pt_model.get_model()}, pickle_path: {pickle_path}')

        toke = pt_model.get_tokenizer()
        print("tokenizer:\n", toke)

        if pickle_path:
            if os.path.exists(pickle_path):
                print(f'loading pickled dataset from {pickle_path}')
                dataset = pickle.load(open(pickle_path, 'rb'))
            else:
                print(f'fetching dataset and dumping it into {pickle_path}')
                dataset = Dataset(
                    dataset=dataset_name, 
                    vtype=vtype, 
                    custom_tokenizer=toke
                )
                print('dumping')
                pickle.dump(dataset, open(pickle_path, 'wb', pickle.HIGHEST_PROTOCOL))
        else:
            print(f'loading dataset {dataset_name}')
            dataset = Dataset(
                dataset=dataset_name,
                vtype=vtype,
                custom_tokenizer=toke
            )

        return dataset


def _label_matrix(tr_target, te_target):
    mlb = MultiLabelBinarizer(sparse_output=True)
    ytr = mlb.fit_transform(tr_target)
    yte = mlb.transform(te_target)
    print(mlb.classes_)
    return ytr, yte


def load_fasttext_format(path):
    print(f'loading {path}')
    labels,docs=[],[]
    for line in tqdm(open(path, 'rt').readlines()):
        space = line.strip().find(' ')
        label = int(line[:space].replace('__label__',''))-1
        labels.append(label)
        docs.append(line[space+1:])
    labels=np.asarray(labels,dtype=int)
    return docs,labels


def mask_numbers(data, number_mask='numbermask'):
    mask = re.compile(r'\b[0-9][0-9.,-]*\b')
    masked = []
    for text in tqdm(data, desc='masking numbers'):
        masked.append(mask.sub(number_mask, text))
    return masked