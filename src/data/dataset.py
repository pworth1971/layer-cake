import os,sys

from sklearn.datasets import get_data_home, fetch_20newsgroups
#from sklearn.datasets import fetch_rcv1

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

# for file download and extraction (rcv1 dataset)
from pathlib import Path
from urllib import request
import tarfile
import gzip
import shutil

from scipy.sparse import coo_matrix, csr_matrix



def init_tfidf_vectorizer():
    """
    Initializes and returns a sklearn TFIDFVectorizer with specific configuration.
    """
    print("init_tfidf_vectorizer()")
    return TfidfVectorizer(min_df=5, sublinear_tf=True)



def init_count_vectorizer():
    """
    Initializes and returns a sklearn CountVectorizer with specific configuration.
    """
    print("init_count_vectorizer()")
    return CountVectorizer(stop_words='english', min_df=5)


class Dataset:
    """
    A class to handle loading and preparing datasets for text classification.
    Supports multiple datasets including Reuters, 20 Newsgroups, Ohsumed, RCV1, and WIPO.
    """

    dataset_available = {'reuters21578', '20newsgroups', 'ohsumed', 'rcv1', 'ohsumed'}

    """
    dataset_available = {'reuters21578', '20newsgroups', 'ohsumed', 'rcv1', 'ohsumed', 'jrcall',
                         'wipo-sl-mg','wipo-ml-mg','wipo-sl-sc','wipo-ml-sc'}
    """
    
    def __init__(self, name=None, vectorization_type='tfidf'):
        """
        Initializes the Dataset object by loading the appropriate dataset.
        """

        print("initializing dataset with name and vectorization_type:", name, vectorization_type)
        
        assert name in Dataset.dataset_available, f'dataset {name} is not available'

        if name=='reuters21578':
            self._load_reuters()
        elif name == '20newsgroups':
            self._load_20news()
        elif name == 'rcv1':
            self._load_rcv1()
            #self._load_rcv1_skl()
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
        print("nC:", self.nC)

        if (vectorization_type=='tfidf'):
            print("initializing tfidf vectors...")
            self._vectorizer = init_tfidf_vectorizer()
        elif (vectorization_type=='count'):
            print("initializing count vectors...")
            self._vectorizer = init_count_vectorizer()
        else:
            print("WARNING: unknown vectorization_type, initializing tf-idf vectorizer...")
            self._vectorizer = init_tfidf_vectorizer()

        print("fitting training data... devel_raw type and length:", type(self.devel_raw), len(self.devel_raw))
        self._vectorizer.fit(self.devel_raw)

        print("setting vocabulary...")
        self.vocabulary = self._vectorizer.vocabulary_


    def show(self):
        nTr_docs = len(self.devel_raw)
        nTe_docs = len(self.test_raw)
        nfeats = len(self._vectorizer.vocabulary_)
        nC = self.devel_labelmatrix.shape[1]
        nD=nTr_docs+nTe_docs
        print(f'{self.classification_type}, nD={nD}=({nTr_docs}+{nTe_docs}), nF={nfeats}, nC={nC}')
        return self

    def _load_reuters(self):

        print("-- Dataset::_load_reuters() --")
        data_path = os.path.join(get_data_home(), 'reuters21578')
        
        devel = fetch_reuters21578(subset='train', data_path=data_path)
        test = fetch_reuters21578(subset='test', data_path=data_path)

        print("dev target names:", type(devel), devel.target_names)
        print("test target names:", type(test), test.target_names)

        self.classification_type = 'multilabel'

        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(devel.target, test.target)

        self.label_names = devel.target_names           # set self.labels to the class label names

        print("num labels:", len(self.labels))
        print("num label names:", len(self.label_names))

        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix


    def _load_20news(self):
        metadata = ('headers', 'footers', 'quotes')
        
        devel = fetch_20newsgroups(subset='train', remove=metadata)
        test = fetch_20newsgroups(subset='test', remove=metadata)

        self.classification_type = 'singlelabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_target, self.test_target = devel.target, test.target
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1,1), self.test_target.reshape(-1,1))


    def _load_rcv1(self):

        data_path = '../datasets/RCV1-v2/rcv1/'               

        print("loading rcv1 Dataset (_load_rcv1) from path:", data_path)

        """
        print('Downloading rcv1v2-ids.dat.gz...')
        self.download_file(
            'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a07-rcv1-doc-ids/rcv1v2-ids.dat.gz', 
            data_path)

        print('Downloading rcv1-v2.topics.qrels.gz...')
        self.download_file(
            'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz', 
            data_path)
        """

        # ----------------------------------------------------------------
        # we presume tar balls are downloaded into this directory already
        # NB: we only do this once
        """
        print("extracting files...")
        self.extract_gz(data_path + '/' +  'rcv1v2-ids.dat.gz')
        self.extract_gz(data_path + '/' + 'rcv1-v2.topics.qrels.gz')
        self.extract_tar(data_path + '/' + 'rcv1.tar.xz')
        self.extract_tar(data_path + '/' + 'RCV1.tar.bz2')
        """
        # ----------------------------------------------------------------

        print("fetching training and test data...")
        devel = fetch_RCV1(subset='train', data_path=data_path, debug=False)
        test = fetch_RCV1(subset='test', data_path=data_path, debug=False)

        print("training data:", type(devel))
        print("training data:", type(devel.data), len(devel.data))
        print("training targets:", type(devel.target), len(devel.target))

        print("testing data:", type(test))
        print("testing data:", type(test.data), len(test.data))
        print("testing targets:", type(test.target), len(test.target))

        self.classification_type = 'multilabel'

        print("masking numbers...")
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(devel.target, test.target)
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
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(devel_target, test_target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix


    def _load_ohsumed(self):
        data_path = os.path.join(get_data_home(), 'ohsumed50k')
        devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        test = fetch_ohsumed50k(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(devel.target, test.target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix


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
        self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1, 1), self.test_target.reshape(-1, 1))


    def _load_wipo(self, classmode, classlevel):
        print("_load_wipo()")

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
            print("--multilabel--")
            devel_target = [d.all_labels for d in devel]
            test_target  = [d.all_labels for d in test]
            self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(devel_target, test_target)
            self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix
        else:
            print("--single label--")
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
            self.devel_labelmatrix, self.test_labelmatrix, self.labels = _label_matrix(self.devel_target.reshape(-1, 1), self.test_target.reshape(-1, 1))


    def get_label_names(self):
        """
        Returns the labels and their associated names (catgeories) associated with t
        he dataset. Useful for plotting confusion matrices and more.
        """
        if hasattr(self, 'label_names'):
            return self.label_names


    def vectorize(self):

        print("vectorizing dataset...")
        
        if not hasattr(self, 'Xtr') or not hasattr(self, 'Xte'):

            print("self does not have Xtr or Xte attributes, transforming and sorting...")

            self.Xtr = self._vectorizer.transform(self.devel_raw)
            self.Xte = self._vectorizer.transform(self.test_raw)
            self.Xtr.sort_indices()
            self.Xte.sort_indices()
        
        print("self.Xtr:", type(self.Xtr), self.Xtr.shape)
        print("self.Xte:", type(self.Xte), self.Xte.shape)
        
        return self.Xtr, self.Xte


    def analyzer(self):
        return self._vectorizer.build_analyzer()


    def download_file(self, url, path):
        file_name = url.split('/')[-1]
        filename_with_path = path + "/" + file_name

        print("file: ", {filename_with_path})

        file = Path(filename_with_path)

        if not file.exists():
            print('File %s does not exist. Downloading ...\n', file_name)
            file_data = request.urlopen(url)
            data_to_write = file_data.read()

            with file.open('wb') as f:
                f.write(data_to_write)
        else:
            print('File %s already existed.\n', file_name)


    def extract_tar(self, path):
        path = Path(path)
        dir_name = '.'.join(path.name.split('.')[:-2])
        dir_output = path.parent/dir_name
        if not dir_output.exists():
            if path.exists():
                tf = tarfile.open(str(path))
                tf.extractall(path.parent)
            else:
                print('ERROR: File %s is required. \n', path.name)


    def extract_gz(self, path):
        path = Path(path)
        file_output_name = '.'.join(path.name.split('.')[:-1])
        file_name = path.name
        if not (path.parent/file_output_name).exists():
            print('Extracting %s ...\n', file_name)

            with gzip.open(str(path), 'rb') as f_in:
                with open(str(path.parent/file_output_name), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


    @classmethod
    def load(cls, dataset_name, vectorization_type='tfidf', base_pickle_path=None):

        print("Dataset::load():", dataset_name, base_pickle_path)

        # Create a pickle path that includes the vectorization type
        # NB we assume the /pickles directory exists already
        if base_pickle_path:
            full_pickle_path = f"{base_pickle_path}{'/'}{dataset_name}_{vectorization_type}.pkl"
            pickle_file_name = f"{dataset_name}_{vectorization_type}.pkl"
        else:
            full_pickle_path = None
            pickle_file_name = None

        print("pickle_file_name:", pickle_file_name)

        # not None so we are going to create the pickle file, 
        # by dataset and vectorization type
        if full_pickle_path:
            print("full_pickle_path: ", {full_pickle_path})

            if os.path.exists(full_pickle_path):                                        # pickle file exists, load it
                print(f'loading pickled dataset from {full_pickle_path}')
                dataset = pickle.load(open(full_pickle_path, 'rb'))
            else:                                                                       # pickle file does not exist, create it, load it, and dump it
                print(f'fetching dataset and dumping it into {full_pickle_path}')
                dataset = Dataset(name=dataset_name, vectorization_type=vectorization_type)
                print('vectorizing for faster processing')
                dataset.vectorize()
                
                print('dumping')
                #pickle.dump(dataset, open(pickle_path, 'wb', pickle.HIGHEST_PROTOCOL))
                # Open the file for writing and write the pickle data
                try:
                    with open(full_pickle_path, 'wb', pickle.HIGHEST_PROTOCOL) as file:
                        pickle.dump(dataset, file)
                    print("data successfully pickled at:", full_pickle_path)
                except Exception as e:
                    print("Exception raised, failed to pickle data: {e}")

        else:
            print(f'loading dataset {dataset_name}')
            dataset = Dataset(name=dataset_name, vectorization_type=vectorization_type)

        return dataset


def _label_matrix(tr_target, te_target):
    
    print("_label_matrix...")

    mlb = MultiLabelBinarizer(sparse_output=True)
    ytr = mlb.fit_transform(tr_target)
    yte = mlb.transform(te_target)

    print("MultiLabelBinarizer.classes_:", mlb.classes_)

    return ytr, yte, mlb.classes_


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
    """
    Masks numbers in the given text data with a placeholder.
    """
    #print("masking numbers...")

    mask = re.compile(r'\b[0-9][0-9.,-]*\b')
    masked = []
    for text in tqdm(data, desc='masking numbers'):
        masked.append(mask.sub(number_mask, text))
    return masked

