import os,sys
from sklearn.datasets import get_data_home, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
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


def init_vectorizer():
    print("init_vectorizer()")
    return TfidfVectorizer(min_df=5, sublinear_tf=True)


class Dataset:

    dataset_available = {'reuters21578', '20newsgroups', 'ohsumed', 'rcv1', 'ohsumed', 'jrcall',
                         'wipo-sl-mg','wipo-ml-mg','wipo-sl-sc','wipo-ml-sc'}

    def __init__(self, name):

        print("Dataset::__init__():", name)
        
        assert name in Dataset.dataset_available, f'dataset {name} is not available'

        if name=='reuters21578':
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
        self._vectorizer = init_vectorizer()
        self._vectorizer.fit(self.devel_raw)
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
        print("Dataset::_load_reuters()")
        data_path = os.path.join(get_data_home(), 'reuters21578')
        devel = fetch_reuters21578(subset='train', data_path=data_path)
        test = fetch_reuters21578(subset='test', data_path=data_path)

        self.classification_type = 'multilabel'
        self.devel_raw, self.test_raw = mask_numbers(devel.data), mask_numbers(test.data)
        #self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(devel.target, test.target)
        self.devel_labelmatrix, self.test_labelmatrix = _label_matrix_non_sparse(devel.target, test.target)
        self.devel_target, self.test_target = self.devel_labelmatrix, self.test_labelmatrix

    def _load_rcv1(self):

        #
        # TODO: this code doesnt work, data set not readily available
        #
        data_path = '../datasets/RCV1-v2/unprocessed_corpus'               

        print('Downloading rcv1v2-ids.dat.gz...')
        self.download_file(
            'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a07-rcv1-doc-ids/rcv1v2-ids.dat.gz', 
            data_path)

        print('Downloading rcv1-v2.topics.qrels.gz...')
        self.download_file(
            'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz', 
            data_path)

        print("extractng files...")
        self.extract_gz(data_path + '/' +  'rcv1v2-ids.dat.gz')
        self.extract_gz(data_path + '/' + 'rcv1-v2.topics.qrels.gz')
        self.extract_tar(data_path + '/' + 'rcv1.tar.xz')

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
            self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(devel_target, test_target)
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
            self.devel_labelmatrix, self.test_labelmatrix = _label_matrix(self.devel_target.reshape(-1, 1), self.test_target.reshape(-1, 1))

    def vectorize(self):
        print("Dataset::vectorize()")
        if not hasattr(self, 'Xtr') or not hasattr(self, 'Xte'):
            self.Xtr = self._vectorizer.transform(self.devel_raw)
            self.Xte = self._vectorizer.transform(self.test_raw)
            self.Xtr.sort_indices()
            self.Xte.sort_indices()
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
    def load(cls, dataset_name, pickle_path=None):

        print("Dataset::load():", dataset_name, pickle_path)

        if pickle_path:
            print("picklepath: ", {pickle_path})

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

        return dataset


def _label_matrix_non_sparse(tr_target, te_target):
    mlb = MultiLabelBinarizer(sparse_output=False)
    ytr = mlb.fit_transform(tr_target)
    yte = mlb.transform(te_target)
    print(mlb.classes_)
    return ytr, yte


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


