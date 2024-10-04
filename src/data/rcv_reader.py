from zipfile import ZipFile
import xml.etree.ElementTree as ET
from data.labeled import LabelledDocuments
from util.file import list_files
from os.path import join, exists
from util.file import download_file_if_not_exists
import re
import os
from collections import Counter

RCV1_TOPICHIER_URL = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig"
RCV1_BASE_URL = "http://www.daviddlewis.com/resources/testcollections/rcv1/"

rcv1_test_data_gz = ['lyrl2004_tokens_test_pt0.dat.gz',
             'lyrl2004_tokens_test_pt1.dat.gz',
             'lyrl2004_tokens_test_pt2.dat.gz',
             'lyrl2004_tokens_test_pt3.dat.gz']

rcv1_train_data_gz = ['lyrl2004_tokens_train.dat.gz']

rcv1_doc_cats_data_gz = 'rcv1-v2.topics.qrels.gz'

class RCV_Document:
    def __init__(self, id, text, categories, date=''):
        self.id = id
        self.date = date
        self.text = text
        self.categories = categories

class IDRangeException(Exception): pass

nwords = []

def parse_document(xml_content, valid_id_range=None):

    #print("parsing XML doc ...")

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None

    doc_id = root.attrib['itemid']
    if valid_id_range is not None:
        if not valid_id_range[0] <= int(doc_id) <= valid_id_range[1]:
            raise IDRangeException

    doc_categories = [cat.attrib['code'] for cat in
                      root.findall('.//metadata/codes[@class="bip:topics:1.0"]/code')]

    doc_date = root.attrib['date']
    doc_title = root.find('.//title').text
    doc_headline = root.find('.//headline').text
    doc_body = '\n'.join([p.text for p in root.findall('.//text/p')])

    if not doc_body:
        raise ValueError('Empty document')

    if doc_title is None: doc_title = ''
    if doc_headline is None or doc_headline in doc_title: doc_headline = ''
    text = '\n'.join([doc_title, doc_headline, doc_body]).strip()

    return RCV_Document(id=doc_id, text=text, categories=doc_categories, date=doc_date)


def fetch_RCV1_raw(data_path, subset='all'):

    print("fetch_RVCV1(data_path):", data_path)
    print("subset:", subset)

    assert subset in ['train', 'test', 'all'], 'split should either be "train", "test", or "all"'

    request = []
    labels = set()
    read_documents = 0

    training_documents = 23149
    test_documents = 781265

    if subset == 'all':
        split_range = (2286, 810596)
        expected = training_documents+test_documents
    elif subset == 'train':
        split_range = (2286, 26150)
        expected = training_documents
    else:
        split_range = (26151, 810596)
        expected = test_documents

    # Scan for directories matching the required pattern
    for dir_name in os.listdir(data_path):                                          # list all the directories 
        if re.match('199\d{5}$', dir_name):                                         # if they start with '199*' and are 8 digits long
            target_dir = os.path.join(data_path, dir_name)
            print("parsing xml files in target_dir...")
            for file_name in os.listdir(target_dir):                                # for each file in directory
                if file_name.endswith('.xml'):
                    xmlfile_path = os.path.join(target_dir, file_name)
                    with open(xmlfile_path, 'r') as file:
                        #print("\nparsing ", xmlfile_path, "...")
                        xmlcontent = file.read()
                        try:
                            doc = parse_document(xmlcontent, valid_id_range=split_range)
                            labels.update(doc.categories)
                            request.append(doc)
                            read_documents += 1
                        except (IDRangeException, ValueError) as e:
                            pass
                        #print('\r[{}] read {} documents'.format(dir_name, len(request)), end='')
                        if read_documents == expected:
                            break
            if read_documents == expected:
                break

    return LabelledDocuments(data=[d.text for d in request], target=[d.categories for d in request], target_names=list(labels))


def fetch_RCV1(data_path, subset='all', debug=False):
    """
    Fetches the RCV1 dataset by parsing zipped XML files from the specified directory.

    This function loads and parses a portion of the Reuters Corpus Volume 1 (RCV1) dataset
    based on the provided subset. The function expects the dataset to be stored in zipped 
    XML format within the `data_path` directory. Depending on the `subset` parameter, it 
    fetches the 'train', 'test', or 'all' portions of the dataset.

    Parameters:
    ----------
    data_path : str
        The path to the directory where the zipped XML files of the RCV1 dataset are stored.
    
    subset : str, optional
        Specifies which subset of the dataset to load. It can be one of the following:
        - 'train': Loads the training subset.
        - 'test': Loads the testing subset.
        - 'all' : Loads the entire dataset (both training and testing).
        Default is 'all'.
    
    debug : bool, optional
        If True, prints debugging information such as file paths and ZIP contents. Default is False.

    Returns:
    -------
    LabelledDocuments
        A collection of the parsed documents, with their associated text, labels, and label names.
        - `data`: List of document texts.
        - `target`: List of document categories (labels).
        - `target_names`: List of all unique categories (labels) found.

    Raises:
    -------
    AssertionError
        If the `subset` argument is not one of ['train', 'test', 'all'].

    Notes:
    ------
    The function assumes that the XML files are stored in a zipped format, where the parent directory
    contains multiple zip files, each corresponding to a part of the dataset.
    """

    print('fetch_RCV1(data_path):', data_path)
    print("subset:", subset)

    assert subset in ['train', 'test', 'all'], 'split should either be "train", "test", or "all"'

    request = []
    labels = set()
    read_documents = 0

    training_documents = 23149
    test_documents = 781265

    if subset == 'all':
        split_range = (2286, 810596)
        expected = training_documents+test_documents
    elif subset == 'train':
        split_range = (2286, 26150)
        expected = training_documents
    else:
        split_range = (26151, 810596)
        expected = test_documents

    #
    # NB: we assume that all of the XML files and their associated parent directories
    # are zipped up in the root folder, ie the parent directory under which
    # all of the XML file directories live
    #
    for part in list_files(data_path):
        if not re.match('\d+\.zip', part): continue
        target_file = join(data_path, part)
        #print("target_file:", target_file)
        assert exists(target_file), \
            "You don't seem to have the file "+part+" in " + data_path + ", and the RCV1 corpus can not be downloaded"+\
            " w/o a formal permission. Please, refer to " + RCV1_BASE_URL + " for more information."
        zipfile = ZipFile(target_file)
        #print("zipfile:", zipfile)
        for xmlfile in zipfile.namelist():
            #print("\nxmlfile:", xmlfile)
            xmlcontent = zipfile.open(xmlfile).read()
            try:
                doc = parse_document(xmlcontent, valid_id_range=split_range)
                labels.update(doc.categories)
                request.append(doc)
                read_documents += 1
            except (IDRangeException,ValueError) as e:
                pass
            #print('\r[{}] read {} documents'.format(part, len(request)), end='')
            #print("\n")
            if read_documents == expected: break
        if read_documents == expected: break

    return LabelledDocuments(data=[d.text for d in request], target=[d.categories for d in request], target_names=list(labels))





def fetch_topic_hierarchy(path, topics='all'):
    assert topics in ['all', 'leaves']

    download_file_if_not_exists(RCV1_TOPICHIER_URL, path)
    hierarchy = {}
    for line in open(path, 'rt'):
        parts = line.strip().split()
        parent,child = parts[1],parts[3]
        if parent not in hierarchy:
            hierarchy[parent]=[]
        hierarchy[parent].append(child)

    del hierarchy['None']
    del hierarchy['Root']
    print(hierarchy)

    if topics=='all':
        topics = set(hierarchy.keys())
        for parent in hierarchy.keys():
            topics.update(hierarchy[parent])
        return list(topics)
    elif topics=='leaves':
        parents = set(hierarchy.keys())
        childs = set()
        for parent in hierarchy.keys():
            childs.update(hierarchy[parent])
        return list(childs.difference(parents))


if __name__=='__main__':

    # example

    RCV1_PATH = '../../datasets/RCV1-v2/unprocessed_corpus'

    rcv1_train = fetch_RCV1(RCV1_PATH, subset='train')
    rcv1_test = fetch_RCV1(RCV1_PATH, subset='test')

    print('read {} documents in rcv1-train, and {} labels'.format(len(rcv1_train.data), len(rcv1_train.target_names)))
    print('read {} documents in rcv1-test, and {} labels'.format(len(rcv1_test.data), len(rcv1_test.target_names)))

    cats = Counter()
    for cats in rcv1_train.target: cats.update(cats)
    print('RCV1', cats)
