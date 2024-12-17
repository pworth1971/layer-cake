import os,sys

from sklearn.datasets import get_data_home, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

from data.jrcacquis_reader import fetch_jrcacquis, JRCAcquis_Document
from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1
from data.wipo_reader import fetch_WIPOgamma, WipoGammaDocument

import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from os.path import join
import re


TEST_SIZE = 0.2
VAL_SIZE = 0.2

MIN_DF_COUNT = 5                    # minimum document frequency count for a term to be included in the vocabulary


DATASET_DIR = '../datasets/'                        # dataset directory


"""
def init_vectorizer():
    return TfidfVectorizer(min_df=5, sublinear_tf=True)
"""



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

    """
    dataset_available = {'bbc-news', 'reuters21578', '20newsgroups', 'ohsumed', 'rcv1', 'ohsumed', 'jrcall',
                         'wipo-sl-mg','wipo-ml-mg','wipo-sl-sc','wipo-ml-sc'}
    """

    dataset_available = {'bbc-news', 'reuters21578', '20newsgroups', 'ohsumed', 'rcv1', 'ohsumed'}

    def __init__(self, name, vtype='tfidf', custom_tokenizer=None):

        print(f'loading dataset {name}')

        assert name in Dataset.dataset_available, f'dataset {name} is not available'

        if name=='bbc-news':
            self._load_bbc_news()
        elif name=='reuters21578':
            self._load_reuters()
        elif name == '20newsgroups':
            self._load_20news()
        elif name == 'rcv1':
            self._load_rcv1()
        elif name == 'ohsumed':
            self._load_ohsumed()
        elif name == 'jrcall':
            self._load_jrc(version='all')
        elif name == 'wipo-sl-mg':
            self._load_wipo('singlelabel', 'maingroup')
        elif name == 'wipo-ml-mg':
            self._load_wipo('multilabel', 'maingroup')
        elif name == 'wipo-sl-sc':
            self._load_wipo('singlelabel', 'subclass')
        elif name == 'wipo-ml-sc':
            self._load_wipo('multilabel', 'subclass')

        self.nC = self.devel_labelmatrix.shape[1]

        if (custom_tokenizer is not None):
            self.custom_toke = True
        else:
            self.custom_toke = False

        print("self.custom_toke:", self.custom_toke)
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



    def _load_bbc_news(self):
        for dirname, _, filenames in os.walk(DATASET_DIR + 'bbc-news'):
            for filename in filenames:
                print(os.path.join(dirname, filename))

        self.classification_type = 'singlelabel'

        # Load datasets
        train_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Train.csv')
        #test_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Test.csv')

        print("train_set:", train_set.shape)
        print("train_set columns:\n", train_set.columns)
        # print("train_set:\n", train_set.head())
        
        # we split the train data into train and test here because that is
        # not done for us with the BBC News dataset (Test data is not labeled)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            train_set['Text'], 
            train_set['Category'], 
            train_size = 1-TEST_SIZE, 
            random_state = 1
        )

        # reset indeces
        X_train_raw = X_train_raw.reset_index(drop=True)
        X_test_raw = X_test_raw.reset_index(drop=True)

        """
        print("\t--- unprocessed text ---")
        print("self.X_train_raw:", type(X_train_raw), X_train_raw.shape)
        print("self.X_train_raw[0]:\n", X_train_raw[0])

        print("self.X_test_raw:", type(X_test_raw), X_test_raw.shape)
        print("self.X_test_raw[0]:\n", X_test_raw[0])
        """

        self.devel_raw, self.test_raw = mask_numbers(X_train_raw), mask_numbers(X_test_raw)

        # Convert target labels to 1D arrays
        self.devel_target = np.array(y_train)       # Flattening the training labels into a 1D array
        self.test_target = np.array(y_test)         # Flattening the test labels into a 1D array

        # Use LabelEncoder to encode the labels into label IDs
        label_encoder = LabelEncoder()
        label_encoder.fit(self.devel_target)        # Fit on training labels

        # Transform labels to numeric IDs
        self.devel_target = label_encoder.transform(self.devel_target)
        self.test_target = label_encoder.transform(self.test_target)
        
        # Pass these reshaped arrays to the _label_matrix method
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(self.devel_target.reshape(-1, 1), self.test_target.reshape(-1, 1))      
        """
        print("devel_labelmatrix:", type(self.devel_labelmatrix), self.devel_labelmatrix.shape)
        print("test_labelmatrix:", type(self.test_labelmatrix), self.test_labelmatrix.shape)
        """

        # Save the original label names (classes)
        self.target_names = label_encoder.classes_
        #print("self.target_names (original labels):\n", self.target_names)


    def _load_reuters(self):
        data_path = os.path.join(get_data_home(), 'reuters21578')
        devel = fetch_reuters21578(subset='train', data_path=data_path)
        test = fetch_reuters21578(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(devel.target, test.target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix
        self.target_names = devel.target_names


    def _load_20news(self):
        metadata = ('headers', 'footers', 'quotes')
        devel = fetch_20newsgroups(subset='train', remove=metadata)
        test = fetch_20newsgroups(subset='test', remove=metadata)
        self.classification_type = 'singlelabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_target, self.test_target = devel.target, test.target
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))
        self.target_names = devel.target_names
    

    def _load_ohsumed(self):
        data_path = os.path.join(get_data_home(), 'ohsumed50k')
        devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        test = fetch_ohsumed50k(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(devel.target, test.target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix
        self.target_names = devel.target_names


    def _load_rcv1(self):
        data_path = '../datasets/RCV1-v2/unprocessed_corpus' #TODO: check when missing
        devel = fetch_RCV1(subset='train', data_path=data_path)
        test = fetch_RCV1(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(devel.target, test.target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix
        self.target_names = devel.target_names
        

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
        if (self.custom_toke):
            print("Using custom tokenizer from vectorizer.")
            return self._vectorizer.tokenizer                                   # Use the custom tokenizer (e.g., for BERT)
        else:
            print("Using default analyzer from vectorizer.")
            return self._vectorizer.build_analyzer()                            # Use the default word-based analyzer


    @classmethod
    def load(cls, name, vtype='tfidf', pt_model=None, pickle_dir=None):
        """
        Load or create the dataset, and serialize it with a model-specific pickle file.

        Parameters:
        ----------
        dataset_name : str
            The name of the dataset to load.
        vtype : str
            The type of vectorizer to use ('tfidf' or 'count').
        pt_model : PretrainedEmbeddings
            The pretrained model object, providing the tokenizer and model name.
        pickle_dir : str
            Directory to store or retrieve the pickle file.

        Returns:
        -------
        Dataset
            The loaded or newly created Dataset object.
        """
        model_type = pt_model.get_type()
        model_name = pt_model.get_model()
        pickle_filename = f"{name}.{vtype}.{model_type}.{model_name}.pickle"
        pickle_path = os.path.join(pickle_dir, pickle_filename) if pickle_dir else None

        print(f'\n\tloading dataset: {name}, vtype: {vtype}, pt_type: {model_type}, pt_model: {model_name}, pickle_path: {pickle_path}')

        # Get the tokenizer from the pretrained model
        tokenizer = pt_model.get_tokenizer()
        print("tokenizer:\n", tokenizer)

        if pickle_path:
            if os.path.exists(pickle_path):
                print(f'loading pickled dataset from {pickle_path}')
                with open(pickle_path, 'rb') as file:
                    dataset = pickle.load(file)
            else:
                print(f'fetching dataset and dumping it into {pickle_path}')
                dataset = Dataset(
                    name=name,
                    vtype=vtype,
                    custom_tokenizer=tokenizer
                )
                print('dumping...')
                with open(pickle_path, 'wb') as file:
                    pickle.dump(dataset, file, pickle.HIGHEST_PROTOCOL)
        else:
            print(f'loading dataset {name}')
            dataset = Dataset(
                name=name,
                vtype=vtype,
                custom_tokenizer=tokenizer
            )

        return dataset


    """
    @classmethod
    def load(cls, dataset_name, pickle_path=None):

        if pickle_path:
            if os.path.exists(pickle_path):
                print(f'loading pickled dataset from {pickle_path}')
                dataset = pickle.load(open(pickle_path, 'rb'))
            else:
                print(f'fetching dataset and dumping it into {pickle_path}')
                dataset = Dataset(name=dataset_name)
                print('vectorizing for faster processing')
                dataset.vectorize()
                print('dumping')
                pickle.dump(dataset, open(pickle_path, 'wb', pickle.HIGHEST_PROTOCOL))
        else:
            print(f'loading dataset {dataset_name}')
            dataset = Dataset(name=dataset_name)

        print('[Done]')
        return dataset
    """


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