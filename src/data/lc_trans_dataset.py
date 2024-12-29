import os
import numpy as np
from tqdm import tqdm
import re
import pandas as pd
import string
import random

from scipy.sparse import csr_matrix

import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from joblib import Parallel, delayed

from data.ohsumed_reader import fetch_ohsumed50k
from data.reuters21578_reader import fetch_reuters21578
from data.rcv_reader import fetch_RCV1


#SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news", "ohsumed", "imdb", "arxiv", "cmu_movie_corpus"]
SUPPORTED_DATASETS = ["20newsgroups", "rcv1", "reuters21578", "bbc-news", "ohsumed", "imdb", "arxiv"]

DATASET_DIR = '../datasets/'                        # dataset directory
PICKLE_DIR = "../pickles/"


#
# Disable Hugging Face tokenizers parallelism to avoid fork issues
#
os.environ["TOKENIZERS_PARALLELISM"] = "false"


TEST_SIZE = 0.175                   # test size for train/test split
NUM_DL_WORKERS = 3                  # number of workers to handle DataLoader tasks

RANDOM_SEED = 29



def get_dataset_data(dataset_name, seed=RANDOM_SEED, pickle_dir=PICKLE_DIR):
    """
    Load dataset data from a pickle file if it exists; otherwise, call the dataset loading method,
    save the returned data to a pickle file, and return the data.

    Parameters:
    - dataset_name (str): Name of the dataset.
    - seed (int): Random seed for reproducibility.
    - pickle_dir (str): Directory where the pickle file is stored.

    Returns:
    - train_data: Training data.
    - train_target: Training labels.
    - test_data: Test data.
    - labels_test: Test labels.
    - num_classes: Number of classes in the dataset.
    - target_names: Names of the target classes.
    - class_type: Classification type (e.g., 'multi-label', 'single-label').
    """

    print(f'get_dataset_data()... dataset_name: {dataset_name}, pickle_dir: {pickle_dir}, seed: {seed}')

    pickle_file = os.path.join(pickle_dir, f"trans_lc.{dataset_name}.pickle")
    
    # Ensure the pickle directory exists
    os.makedirs(pickle_dir, exist_ok=True)

    if os.path.exists(pickle_file):
        print(f"Loading dataset from pickle file: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
    else:
        print(f"Pickle file not found. Loading dataset using `trans_lc_load_dataset` for {dataset_name}...")
        
        data = trans_lc_load_dataset(
            name=dataset_name, 
            seed=seed) 
        
        # Save the dataset to a pickle file
        print(f"Saving dataset to pickle file: {pickle_file}")
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)

    # Unpack and return the data
    (train_data, train_target), (test_data, labels_test), num_classes, target_names, class_type = data

    return train_data, train_target, test_data, labels_test, num_classes, target_names, class_type




def preprocess(text_series: pd.Series, remove_punctuation=True, lowercase=False, remove_stopwords=False):
    """
    Preprocess a pandas Series of texts by removing punctuation, optionally lowercasing, and optionally removing stopwords.
    Numbers are not masked, and text remains in its original form unless modified by these options.

    Parameters:
    - text_series: A pandas Series containing text data (strings).
    - remove_punctuation: Boolean indicating whether to remove punctuation.
    - lowercase: Boolean indicating whether to convert text to lowercase.
    - remove_stopwords: Boolean indicating whether to remove stopwords.

    Returns:
    - processed_texts: A list containing processed text strings.
    """

    print("preprocessing...")
    print("text_series:", type(text_series), text_series.shape)
    
    # Load stop words once outside the loop
    stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    punctuation_table = str.maketrans('', '', string.punctuation)  # Translation table to remove punctuation

    # Function to process each text
    def process_text(text):
        if lowercase:
            text = text.lower()

        if remove_punctuation:
            text = text.translate(punctuation_table)

        if remove_stopwords:
            for stopword in stop_words:
                text = re.sub(r'\b' + re.escape(stopword) + r'\b', '', text)

        # Ensure extra spaces are removed after stopwords are deleted
        return ' '.join(text.split())

    # Use Parallel processing with multiple cores
    processed_texts = Parallel(n_jobs=-1)(delayed(process_text)(text) for text in text_series)

    # Return as a list
    return list(processed_texts)



def _mask_numbers(data, number_mask='[NUM]'):
    """
    Masks numbers in the given text data with a placeholder.
    """
    mask = re.compile(r'\b[0-9][0-9.,-]*\b')
    return [mask.sub(number_mask, text) for text in data]


def preprocess_text(data):
    """
    Preprocess the text data by converting to lowercase, masking numbers, 
    removing punctuation, and removing stopwords.
    """
    import re
    from nltk.corpus import stopwords
    from string import punctuation

    stop_words = set(stopwords.words('english'))
    punct_table = str.maketrans("", "", punctuation)

    def _remove_punctuation_and_stopwords(data):
        """
        Removes punctuation and stopwords from the text data.
        """
        cleaned = []
        for text in data:
            # Remove punctuation and lowercase text
            text = text.translate(punct_table).lower()
            # Remove stopwords
            tokens = text.split()
            tokens = [word for word in tokens if word not in stop_words]
            cleaned.append(" ".join(tokens))
        return cleaned

    # Apply preprocessing steps
    masked = _mask_numbers(data)
    cleaned = _remove_punctuation_and_stopwords(masked)
    return cleaned




def _label_matrix(tr_target, te_target):
    """
    Converts multi-label target data into a binary matrix format using MultiLabelBinarizer.
    
    Input:
    - tr_target: A list (or iterable) of multi-label sets for the training data. 
                 Each element is a list, tuple, or set of labels assigned to a sample.
                 Example: [["label1", "label2"], ["label3"], ["label2", "label4"]]
    - te_target: A list (or iterable) of multi-label sets for the test data.
                 Each element is a list, tuple, or set of labels assigned to a sample.
                 Example: [["label1"], ["label3", "label4"]]
    
    Output:
    - ytr: A binary label matrix for the training data where each column represents a label.
           The matrix has shape (n_samples, n_classes).
    - yte: A binary label matrix for the test data where each column represents a label.
           The matrix has shape (n_samples, n_classes).
    - mlb.classes_: A list of all unique classes (labels) across the training data.
    """
    
    """
    print("_label_matrix...")
    print("tr_target:", tr_target)
    print("te_target:", te_target)
    """

    mlb = MultiLabelBinarizer(sparse_output=True)
    
    ytr = mlb.fit_transform(tr_target)
    yte = mlb.transform(te_target)

    """
    print("ytr:", type(ytr), ytr.shape)
    print("yte:", type(yte), yte.shape)

    print("MultiLabelBinarizer.classes_:\n", mlb.classes_)
    """
    
    return ytr, yte, mlb.classes_





# ------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Load dataset method for transformer based neural models
#
def trans_lc_load_dataset(name, seed):

    print(f'trans_lc_load_dataset(): dataset: {name}, seed: {seed}...')

    if name == "20newsgroups":

        """
        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
        """

        metadata = ('headers', 'footers', 'quotes')
        train_data = fetch_20newsgroups(subset='train', remove=metadata)
        test_data = fetch_20newsgroups(subset='test', remove=metadata)

        target_names = list(set(train_data.target_names))  # Ensures unique class names
        num_classes = len(target_names)
  
        # Preprocess text data
        """
        train_data_processed = preprocess_text(train_data.data)
        test_data_processed = preprocess_text(test_data.data)
        """

        train_data_processed = preprocess(
            pd.Series(train_data.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False
            )

        test_data_processed = preprocess(
            pd.Series(test_data.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False
            )

        class_type = 'single-label'

        return (train_data_processed, train_data.target), (test_data_processed, test_data.target), num_classes, target_names, class_type
        

    elif name == "reuters21578":
        
        import os

        data_path = os.path.join(DATASET_DIR, 'reuters21578')    
        print("data_path:", data_path)  

        class_type = 'multi-label'

        train_labelled_docs = fetch_reuters21578(subset='train', data_path=data_path)
        test_labelled_docs = fetch_reuters21578(subset='test', data_path=data_path)

        train_data = preprocess(
            pd.Series(train_labelled_docs.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)
        
        test_data = preprocess(
            pd.Series(test_labelled_docs.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)
        
        """
        train_data = preprocess_text(train_labelled_docs.data)
        test_data = preprocess_text(list(test_labelled_docs.data))
        """

        train_target = train_labelled_docs.target
        test_target = test_labelled_docs.target
        
        train_target, test_target, labels = _label_matrix(train_target, test_target)

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        target_names = train_labelled_docs.target_names
        num_classes = len(target_names)
        
        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type
        
    elif name == "ohsumed":

        import os
        
        data_path = os.path.join(DATASET_DIR, 'ohsumed50k')

        devel = fetch_ohsumed50k(subset='train', data_path=data_path)
        test = fetch_ohsumed50k(subset='test', data_path=data_path)

        """
        train_data = preprocess_text(devel.data)
        test_data = preprocess_text(test.data)
        """
        
        """
        train_data = _preprocess(pd.Series(devel.data), remove_punctuation=False)
        test_data = _preprocess(pd.Series(test.data), remove_punctuation=False)
        """

        train_data = preprocess(
            pd.Series(devel.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)
        
        test_data = preprocess(
            pd.Series(test.data), 
            remove_punctuation=False, 
            lowercase=True, 
            remove_stopwords=False)

        train_target, test_target = devel.target, test.target
        class_type = 'multi-label'
        
        train_target, test_target, target_names = _label_matrix(train_target, test_target)

        target_names = devel.target_names

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        num_classes = len(target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type
    
    elif name == "bbc-news":

        import os

        for dirname, _, filenames in os.walk(DATASET_DIR + 'bbc-news'):
            for filename in filenames:
                print(os.path.join(dirname, filename))

        class_type = 'single-label'

        # Load datasets
        train_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Train.csv')
        #test_set = pd.read_csv(DATASET_DIR + 'bbc-news/BBC News Test.csv')
        
        target_names = train_set['Category'].unique()
        num_classes = len(train_set['Category'].unique())
        #print(f"num_classes: {len(target_names)}")
        #print("class_names:", target_names)
        
        # we split the train data into train and test here because that is
        # not done for us with the BBC News dataset (Test data is not labeled)
        train_data, test_data, train_target, test_target = train_test_split(
            train_set['Text'], 
            train_set['Category'], 
            train_size = 1-TEST_SIZE, 
            random_state = seed,
        )

        """
        train_data = preprocess_text(train_data.tolist())
        test_data = preprocess_text(test_data.tolist())
        """

        train_data = preprocess(
            train_data, 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )

        test_data = preprocess(
            test_data, 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )
        
        #
        # set up label targets
        # Convert target labels to 1D arrays
        train_target_arr = np.array(train_target)  # Flattening the training labels into a 1D array
        test_target_arr = np.array(test_target)    # Flattening the test labels into a 1D array

        # Use LabelEncoder to encode the labels into label IDs
        label_encoder = LabelEncoder()
        label_encoder.fit(train_target_arr)  # Fit on training labels

        # Transform labels to numeric IDs
        train_target_encoded = label_encoder.transform(train_target_arr)
        test_target_encoded = label_encoder.transform(test_target_arr)

        return (train_data, train_target_encoded), (test_data, test_target_encoded), num_classes, target_names, class_type

    elif name == "rcv1":

        import os
        
        data_path = os.path.join(DATASET_DIR, 'rcv1')
        
        class_type = 'multi-label'

        """
        from sklearn.datasets import fetch_rcv1

        devel_data, devel_target = fetch_rcv1(
            subset='train', 
            data_home=data_path, 
            download_if_missing=True,
            return_X_y=True,
            #shuffle-True
            )

        test_data, test_target = fetch_rcv1(
            subset='test', 
            data_home=data_path, 
            download_if_missing=True,
            return_X_y=True,
            #shuffle-True
            )
       
        train_data = preprocess(
            pd.DataFrame(devel_data.toarray()).apply(lambda x: ' '.join(x.astype(str)), axis=1),
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
        )

        test_data = preprocess(
            pd.DataFrame(test_data.toarray()).apply(lambda x: ' '.join(x.astype(str)), axis=1),
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
        )
 
        """

        devel = fetch_RCV1(subset='train', data_path=data_path)
        test = fetch_RCV1(subset='test', data_path=data_path)
        
        train_data = preprocess(
            pd.Series(devel.data),
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
        )

        test_data = preprocess(
            pd.Series(test.data),
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
        )

        train_target, test_target = devel.target, test.target
                
        train_target, test_target, target_names = _label_matrix(train_target, test_target)

        train_target = train_target.toarray()                                     # Convert to dense
        test_target = test_target.toarray()                                       # Convert to dense

        num_classes = len(target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type

    elif name == 'imdb':

        from datasets import load_dataset
        import os

        class_type = 'single-label'

        data_path = os.path.join(DATASET_DIR, 'imdb')

        # Load IMDB dataset using the Hugging Face Datasets library
        imdb_dataset = load_dataset('imdb', cache_dir=data_path)

        #train_data = preprocess_text(imdb_dataset['train']['text'])
        #train_data = _preprocess(imdb_dataset['train']['text'], remove_punctuation=False)
        train_data = imdb_dataset['train']['text']

        # Split dataset into training and test data
        #train_data = imdb_dataset['train']['text']
        train_target = np.array(imdb_dataset['train']['label'], dtype=np.int64)  # Convert to numpy array of type int64

        #test_data = imdb_dataset['test']['text']
        #test_data = preprocess_text(imdb_dataset['test']['text'])
        #test_data = _preprocess(imdb_dataset['test']['text'], remove_punctuation=False)
        test_data = imdb_dataset['test']['text']

        test_target = np.array(imdb_dataset['test']['label'], dtype=np.int64)  # Convert to numpy array of type int64

        train_data = preprocess(
            pd.Series(train_data), 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )

        test_data = preprocess(
            pd.Series(test_data), 
            remove_punctuation=False,
            lowercase=True,
            remove_stopwords=False
            )

        # Define target names
        target_names = ['negative', 'positive']
        num_classes = len(target_names)

        return (train_data, train_target), (test_data, test_target), num_classes, target_names, class_type

    elif name == 'arxiv':

        import os, json, re

        class_type = 'multi-label'

        sci_field_map = {'astro-ph': 'Astrophysics',
                'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
                'astro-ph.EP': 'Earth and Planetary Astrophysics',
                'astro-ph.GA': 'Astrophysics of Galaxies',
                'astro-ph.HE': 'High Energy Astrophysical Phenomena',
                'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
                'astro-ph.SR': 'Solar and Stellar Astrophysics',
                'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
                'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
                'cond-mat.mtrl-sci': 'Materials Science',
                'cond-mat.other': 'Other Condensed Matter',
                'cond-mat.quant-gas': 'Quantum Gases',
                'cond-mat.soft': 'Soft Condensed Matter',
                'cond-mat.stat-mech': 'Statistical Mechanics',
                'cond-mat.str-el': 'Strongly Correlated Electrons',
                'cond-mat.supr-con': 'Superconductivity',
                'cs.AI': 'Artificial Intelligence',
                'cs.AR': 'Hardware Architecture',
                'cs.CC': 'Computational Complexity',
                'cs.CE': 'Computational Engineering, Finance, and Science',
                'cs.CG': 'Computational Geometry',
                'cs.CL': 'Computation and Language',
                'cs.CR': 'Cryptography and Security',
                'cs.CV': 'Computer Vision and Pattern Recognition',
                'cs.CY': 'Computers and Society',
                'cs.DB': 'Databases',
                'cs.DC': 'Distributed, Parallel, and Cluster Computing',
                'cs.DL': 'Digital Libraries',
                'cs.DM': 'Discrete Mathematics',
                'cs.DS': 'Data Structures and Algorithms',
                'cs.ET': 'Emerging Technologies',
                'cs.FL': 'Formal Languages and Automata Theory',
                'cs.GL': 'General Literature',
                'cs.GR': 'Graphics',
                'cs.GT': 'Computer Science and Game Theory',
                'cs.HC': 'Human-Computer Interaction',
                'cs.IR': 'Information Retrieval',
                'cs.IT': 'Information Theory',
                'cs.LG': 'Machine Learning',
                'cs.LO': 'Logic in Computer Science',
                'cs.MA': 'Multiagent Systems',
                'cs.MM': 'Multimedia',
                'cs.MS': 'Mathematical Software',
                'cs.NA': 'Numerical Analysis',
                'cs.NE': 'Neural and Evolutionary Computing',
                'cs.NI': 'Networking and Internet Architecture',
                'cs.OH': 'Other Computer Science',
                'cs.OS': 'Operating Systems',
                'cs.PF': 'Performance',
                'cs.PL': 'Programming Languages',
                'cs.RO': 'Robotics',
                'cs.SC': 'Symbolic Computation',
                'cs.SD': 'Sound',
                'cs.SE': 'Software Engineering',
                'cs.SI': 'Social and Information Networks',
                'cs.SY': 'Systems and Control',
                'econ.EM': 'Econometrics',
                'eess.AS': 'Audio and Speech Processing',
                'eess.IV': 'Image and Video Processing',
                'eess.SP': 'Signal Processing',
                'gr-qc': 'General Relativity and Quantum Cosmology',
                'hep-ex': 'High Energy Physics - Experiment',
                'hep-lat': 'High Energy Physics - Lattice',
                'hep-ph': 'High Energy Physics - Phenomenology',
                'hep-th': 'High Energy Physics - Theory',
                'math.AC': 'Commutative Algebra',
                'math.AG': 'Algebraic Geometry',
                'math.AP': 'Analysis of PDEs',
                'math.AT': 'Algebraic Topology',
                'math.CA': 'Classical Analysis and ODEs',
                'math.CO': 'Combinatorics',
                'math.CT': 'Category Theory',
                'math.CV': 'Complex Variables',
                'math.DG': 'Differential Geometry',
                'math.DS': 'Dynamical Systems',
                'math.FA': 'Functional Analysis',
                'math.GM': 'General Mathematics',
                'math.GN': 'General Topology',
                'math.GR': 'Group Theory',
                'math.GT': 'Geometric Topology',
                'math.HO': 'History and Overview',
                'math.IT': 'Information Theory',
                'math.KT': 'K-Theory and Homology',
                'math.LO': 'Logic',
                'math.MG': 'Metric Geometry',
                'math.MP': 'Mathematical Physics',
                'math.NA': 'Numerical Analysis',
                'math.NT': 'Number Theory',
                'math.OA': 'Operator Algebras',
                'math.OC': 'Optimization and Control',
                'math.PR': 'Probability',
                'math.QA': 'Quantum Algebra',
                'math.RA': 'Rings and Algebras',
                'math.RT': 'Representation Theory',
                'math.SG': 'Symplectic Geometry',
                'math.SP': 'Spectral Theory',
                'math.ST': 'Statistics Theory',
                'math-ph': 'Mathematical Physics',
                'nlin.AO': 'Adaptation and Self-Organizing Systems',
                'nlin.CD': 'Chaotic Dynamics',
                'nlin.CG': 'Cellular Automata and Lattice Gases',
                'nlin.PS': 'Pattern Formation and Solitons',
                'nlin.SI': 'Exactly Solvable and Integrable Systems',
                'nucl-ex': 'Nuclear Experiment',
                'nucl-th': 'Nuclear Theory',
                'physics.acc-ph': 'Accelerator Physics',
                'physics.ao-ph': 'Atmospheric and Oceanic Physics',
                'physics.app-ph': 'Applied Physics',
                'physics.atm-clus': 'Atomic and Molecular Clusters',
                'physics.atom-ph': 'Atomic Physics',
                'physics.bio-ph': 'Biological Physics',
                'physics.chem-ph': 'Chemical Physics',
                'physics.class-ph': 'Classical Physics',
                'physics.comp-ph': 'Computational Physics',
                'physics.data-an': 'Data Analysis, Statistics and Probability',
                'physics.ed-ph': 'Physics Education',
                'physics.flu-dyn': 'Fluid Dynamics',
                'physics.gen-ph': 'General Physics',
                'physics.geo-ph': 'Geophysics',
                'physics.hist-ph': 'History and Philosophy of Physics',
                'physics.ins-det': 'Instrumentation and Detectors',
                'physics.med-ph': 'Medical Physics',
                'physics.optics': 'Optics',
                'physics.plasm-ph': 'Plasma Physics',
                'physics.pop-ph': 'Popular Physics',
                'physics.soc-ph': 'Physics and Society',
                'physics.space-ph': 'Space Physics',
                'q-bio.BM': 'Biomolecules',
                'q-bio.CB': 'Cell Behavior',
                'q-bio.GN': 'Genomics',
                'q-bio.MN': 'Molecular Networks',
                'q-bio.NC': 'Neurons and Cognition',
                'q-bio.OT': 'Other Quantitative Biology',
                'q-bio.PE': 'Populations and Evolution',
                'q-bio.QM': 'Quantitative Methods',
                'q-bio.SC': 'Subcellular Processes',
                'q-bio.TO': 'Tissues and Organs',
                'q-fin.CP': 'Computational Finance',
                'q-fin.EC': 'Economics',
                'q-fin.GN': 'General Finance',
                'q-fin.MF': 'Mathematical Finance',
                'q-fin.PM': 'Portfolio Management',
                'q-fin.PR': 'Pricing of Securities',
                'q-fin.RM': 'Risk Management',
                'q-fin.ST': 'Statistical Finance',
                'q-fin.TR': 'Trading and Market Microstructure',
                'quant-ph': 'Quantum Physics',
                'stat.AP': 'Applications',
                'stat.CO': 'Computation',
                'stat.ME': 'Methodology',
                'stat.ML': 'Machine Learning',
                'stat.OT': 'Other Statistics',
                'stat.TH': 'Statistics Theory'}
        #
        # code from
        # https://www.kaggle.com/code/jampaniramprasad/arxiv-abstract-classification-using-roberta
        #
        data_path = os.path.join(DATASET_DIR, 'arxiv')

        file_path = data_path + '/arxiv-metadata-oai-snapshot.json'
        #print("file_path:", file_path)

        # Preprocessing function for text cleaning
        def clean_text(text):
            text = text.lower()                                                       # Convert text to lowercase
            text = re.sub(r'\d+', '<NUM>', text)                                      # Mask numbers
            text = re.sub(r'\$\{[^}]*\}|\$|\\[a-z]+|[{}]', '', text)                  # Remove LaTeX-like symbols
            text = re.sub(r'\s+', ' ', text).strip()                                  # Remove extra spaces
            return text

        # Generator function with progress bar
        def get_data(file_path, preprocess=False):
            with open(file_path, 'r') as f:
                # Use tqdm to wrap the file iterator
                for line in tqdm(f, desc="Loading dataset", unit="line"):
                    paper = json.loads(line)
                    if preprocess:
                        # Apply text cleaning to relevant fields
                        paper['title'] = clean_text(paper.get('title', ''))
                        paper['abstract'] = clean_text(paper.get('abstract', ''))
                    yield paper

        paper_metadata = get_data(file_path, preprocess=True)
        #print("paper_metadata:", type(paper_metadata))

        """
        def load_dataset(file_path):
            data = []
            with open(file_path, 'r') as f:
                for line in tqdm(f, desc="Loading dataset", unit="line", total=2626136):            # Approximate total
                    data.append(json.loads(line))
            return data

        dataset = load_dataset(file_path)
        """

        # Using `yield` to load the JSON file in a loop to prevent 
        # Python memory issues if JSON is loaded directly
        def get_raw_data():
            with open(file_path, 'r') as f:
                for thing in f:
                    yield thing

        #paper_metadata = get_data()

        """
        for paper in paper_metadata:
            for k, v in json.loads(paper).items():
                print(f'{k}: {v} \n')
            break
        """

        paper_titles = []
        paper_intro = []
        paper_type = []

        paper_categories = np.array(list(sci_field_map.keys())).flatten()

        metadata_of_paper = get_raw_data()
        for paper in tqdm(metadata_of_paper):
            papers_dict = json.loads(paper)
            category = papers_dict.get('categories')
            try:
                try:
                    year = int(papers_dict.get('journal-ref')[-4:])
                except:
                    year = int(papers_dict.get('journal-ref')[-5:-1])

                if category in paper_categories and 2010<year<2021:
                    paper_titles.append(papers_dict.get('title'))
                    paper_intro.append(papers_dict.get('abstract'))
                    paper_type.append(papers_dict.get('categories'))
            except:
                pass 

        """
        print("paper_titles:", paper_titles[:5])
        print("paper_intro:", paper_intro[:5])
        print("paper_type:", paper_type[:5])
        print(len(paper_titles), len(paper_intro), len(paper_type))
        """
        papers_dataframe = pd.DataFrame({
            'title': paper_titles,
            'abstract': paper_intro,
            'categories': paper_type
        })

        # preprocess text
        papers_dataframe['abstract'] = papers_dataframe['abstract'].apply(lambda x: x.replace("\n",""))
        papers_dataframe['abstract'] = papers_dataframe['abstract'].apply(lambda x: x.strip())
        papers_dataframe['text'] = papers_dataframe['title'] + '. ' + papers_dataframe['abstract']

        papers_dataframe['categories'] = papers_dataframe['categories'].apply(lambda x: tuple(x.split()))
        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())
        """

        # Ensure the 'categories' column value counts are calculated and indexed properly
        categories_counts = papers_dataframe['categories'].value_counts().reset_index(name="count")
        """
        print("categories_counts:", categories_counts.shape)
        print(categories_counts.head())
        """

        # Filter for categories with a count greater than 250
        shortlisted_categories = categories_counts.query("count > 250")["categories"].tolist()
        print("shortlisted_categories:", shortlisted_categories)

        # Choosing paper categories based on their frequency & eliminating categories with very few papers
        #shortlisted_categories = papers_dataframe['categories'].value_counts().reset_index(name="count").query("count > 250")["index"].tolist()
        papers_dataframe = papers_dataframe[papers_dataframe["categories"].isin(shortlisted_categories)].reset_index(drop=True)
        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())
        """

        # clean the text, remove special chracters, etc
        def clean_text(text):
            # Mask numbers
            text = re.sub(r'\d+', '<NUM>', text)
            # Remove special LaTeX-like symbols and tags
            text = re.sub(r'\$\{[^}]*\}|\$|\\[a-z]+|[{}]', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # Apply cleaning to dataset texts
        papers_dataframe['text'] = papers_dataframe['text'].apply(clean_text)
        
        """
        print("papers_dataframe:", papers_dataframe.shape)
        print(papers_dataframe.head())
        """

        # Shuffle DataFrame
        papers_dataframe = papers_dataframe.sample(frac=1).reset_index(drop=True)

        # Sample roughtly equal number of texts from different paper categories (to reduce class imbalance issues)
        papers_dataframe = papers_dataframe.groupby('categories').head(250).reset_index(drop=True)

        # encode categories using MultiLabelBinarizer
        multi_label_encoder = MultiLabelBinarizer()
        multi_label_encoder.fit(papers_dataframe['categories'])
        papers_dataframe['categories_encoded'] = papers_dataframe['categories'].apply(lambda x: multi_label_encoder.transform([x])[0])

        papers_dataframe = papers_dataframe[["text", "categories", "categories_encoded"]]
        del paper_titles, paper_intro, paper_type
        #print(papers_dataframe.head())

        # Convert encoded labels to a 2D array
        y = np.vstack(papers_dataframe['categories_encoded'].values)
        #y = papers_dataframe['categories_encoded'].values

        # Retrieve target names and number of classes
        target_names = multi_label_encoder.classes_
        num_classes = len(target_names)

        # split dataset into training and test set
        xtrain, xtest, ytrain, ytest = train_test_split(papers_dataframe['text'], y, test_size=TEST_SIZE, random_state=seed)

        return (xtrain.tolist(), ytrain), (xtest.tolist(), ytest), num_classes, target_names, class_type
    

    elif name == 'cmu_movie_corpus':                # TODO, not working with model, need to fix
        """
        Load and process the CMU Movie Corpus for multi-label classification.

        from 
        https://github.com/prateekjoshi565/movie_genre_prediction/blob/master/Movie_Genre_Prediction.ipynb

        Returns:
            train_data (list): List of movie plots (text) for training.
            train_target (numpy.ndarray): Multi-label binary matrix for training labels.
            test_data (list): List of movie plots (text) for testing.
            test_target (numpy.ndarray): Multi-label binary matrix for testing labels.
            num_classes (int): Number of unique genres (classes).
            target_names (list): List of genre names.
            class_type (str): Classification type ('multi-label').
        """

        import csv
        import json
        import os
        import re

        # Classification type
        class_type = "multi-label"

        data_path = os.path.join(DATASET_DIR, 'cmu_movie_corpus')
        print("data_path:", data_path)

        # Ensure the dataset files exist
        tsv_file = os.path.join(data_path, "movie.metadata.tsv")
        if not os.path.exists(tsv_file):
            raise FileNotFoundError(f"Dataset file not found at {tsv_file}. Please download the dataset as per the article instructions.")

        meta = pd.read_csv(tsv_file, sep = '\t', header = None)
        meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]        # rename columns

        #print("meta:\n", meta.head())

        file_path_2 = data_path + '/plot_summaries.txt'
        plots = []
        with open(file_path_2, 'r') as f:
            reader = csv.reader(f, dialect='excel-tab') 
            for row in tqdm(reader):
                plots.append(row)


        movie_id = []
        plot = []

        for i in tqdm(plots):
            movie_id.append(i[0])
            plot.append(i[1])

        movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})

        #print("movies:\n", movies.head())

        # change datatype of 'movie_id'
        meta['movie_id'] = meta['movie_id'].astype(str)

        # merge meta with movies
        movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

        #print("movies:\n", movies.head())

        # get genre data
        genres = []
        for i in movies['genre']:
            genres.append(list(json.loads(i).values()))
        movies['genre_new'] = genres

        # remove samples with 0 genre tags
        movies_new = movies[~(movies['genre_new'].str.len() == 0)]

        """
        print("movies:", movies.shape)
        print("movies_new:", movies_new.shape)
        
        print("movies_new:\n", movies_new.head())
        """

        # get all genre tags in a list
        all_genres = sum(genres,[])
        len(set(all_genres))
        #print("all_genres:", all_genres)

        # function for text cleaning
        def clean_text(text):
            # remove backslash-apostrophe
            text = re.sub("\'", "", text)
            # remove everything alphabets
            text = re.sub("[^a-zA-Z]"," ",text)
            # remove whitespaces
            text = ' '.join(text.split())
            # convert text to lowercase
            text = text.lower()
            
            return text

        movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))
        #print("movies_new:", movies_new.head())
        
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        # function to remove stopwords
        def remove_stopwords(text):
            no_stopword_text = [w for w in text.split() if not w in stop_words]
            return ' '.join(no_stopword_text)
        
        movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))
        
        mlb = MultiLabelBinarizer()
        mlb.fit(movies_new['genre_new'])

        # transform target variable
        y = mlb.transform(movies_new['genre_new'])

        # Retrieve target names and number of classes
        target_names = mlb.classes_
        num_classes = len(target_names)

        # split dataset into training and test set
        xtrain, xtest, ytrain, ytest = train_test_split(movies_new['clean_plot'], y, test_size=TEST_SIZE, random_state=seed)

        """
        xtrain = _preprocess(pd.Series(xtrain), remove_punctuation=False)
        xtest = _preprocess(pd.Series(xtest), remove_punctuation=False)  
        """

        return (xtrain.tolist(), ytrain), (xtest.tolist(), ytest), num_classes, target_names, class_type
    else:
        raise ValueError("Unsupported dataset:", name)




import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

def show_class_distribution(labels, target_names, class_type, dataset_name, display_mode='text'):
    """
    Visualize the class distribution and compute class weights for single-label or multi-label datasets.
    Supports graphical display or text-based summary for remote sessions.

    Parameters:
    - labels: The label matrix (numpy array or csr_matrix for multi-label) or 1D array for single-label.
    - target_names: A list of class names corresponding to the labels.
    - class_type: A string, either 'single-label' or 'multi-label', to specify the classification type.
    - dataset_name: A string representing the name of the dataset.
    - display_mode: A string, 'both', 'text', or 'graph'. Controls whether to display a graph, text, or both.

    Returns:
    - class_weights: A list of computed weights for each class, useful for loss functions.
    """
    # Handle sparse matrix for multi-label case
    if isinstance(labels, csr_matrix):
        labels = labels.toarray()

    # Calculate class counts differently for single-label vs multi-label
    if class_type == 'single-label':
        # For single-label, count occurrences of each class
        unique_labels, class_counts = np.unique(labels, return_counts=True)
    elif class_type == 'multi-label':
        # For multi-label, sum occurrences across the columns
        class_counts = np.sum(labels, axis=0)
        unique_labels = np.arange(len(class_counts))
    else:
        raise ValueError(f"Unsupported class_type: {class_type}")

    # Total number of samples
    total_samples = labels.shape[0]

    # Calculate class weights (inverse frequency)
    class_weights = [total_samples / (len(class_counts) * count) if count > 0 else 0 for count in class_counts]

    # Normalize weights
    max_weight = max(class_weights) if max(class_weights) > 0 else 1
    class_weights = [w / max_weight for w in class_weights]

    # Display graphical output if requested
    if display_mode in ('both', 'graph'):
        plt.figure(figsize=(14, 8))
        plt.bar(target_names, class_counts, color='blue', alpha=0.7)
        plt.xlabel('Classes', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title(f'Class Distribution in {dataset_name} ({len(target_names)} Classes)', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        try:
            plt.show()
        except:
            print("Unable to display the graph (e.g., no GUI backend available). Switching to text-only output.")

    # Display text-based summary if requested
    if display_mode in ('both', 'text'):
        print(f"\n\tClass Distribution and Weights in {dataset_name}:")
        for idx, (class_name, count, weight) in enumerate(zip(target_names, class_counts, class_weights)):
            print(f"{idx:2d}: {class_name:<20} Count: {count:5d}, Weight: {weight:.4f}")

    print("\tclass weights:\n", class_weights)

    return class_weights




def check_empty_docs(data, name):
    """
    Check for empty docs (strings) in a list of data and print details for debugging.

    Args:
        data: List of strings (e.g., train_data or test_data).
        name: Name of the dataset (e.g., "Train", "Test").
    """
    empty_indices = [i for i, doc in enumerate(data) if not doc.strip()]
    if empty_indices:
        print(f"[WARNING] {name} dataset contains {len(empty_indices)} empty strings (docs).")
        for idx in empty_indices[:10]:  # Print details for up to 10 empty rows
            print(f"Empty String at Index {idx}: Original Document: '{data[idx]}'")
    else:
        print(f"[INFO] No empty strings (docs) found in {name} dataset.")



def spot_check_documents(documents, vectorizer, tokenizer, vectorized_data, num_docs=5, debug=False):
    """
    Spot-check random documents in the dataset for their TF-IDF calculations.

    Args:
        documents: List of original documents (strings).
        vectorizer: Fitted LCTFIDFVectorizer object.
        tokenizer: Custom tokenizer object.
        vectorized_data: Sparse matrix of TF-IDF features.
        num_docs: Number of random documents to check.
    """
    vocab = vectorizer.vocabulary_
    idf_values = vectorizer.idf_
    reverse_vocab = {idx: token for token, idx in vocab.items()}
    
    print("\n[INFO] Spot-checking random documents...")
    
    if (debug):
        print("documents:", type(documents), len(documents))
        print("documents[0]:", type(documents[0]), documents[0])

        print("vectorized_data:", type(vectorized_data))
        print(f"[DEBUG] Vocabulary size: {len(vocab)}, IDF array size: {len(idf_values)}\n")

    # Randomly select `num_docs` indices from the document list
    doc_indices = random.sample(range(len(documents)), min(num_docs, len(documents)))

    for doc_id in doc_indices:
        doc = documents[doc_id]

        if (debug):
            print(f"[INFO] Document {doc_id}:")
            print(f"Original Text: {doc}\n")

        tokens = tokenizer(doc)
        if (debug):
            print(f"Tokens: {tokens}\n")

        vectorized_row = vectorized_data[doc_id]
        mismatches = []

        for token in tokens:
            if token in vocab:
                idx = vocab[token]
                if idx < len(idf_values):
                    tfidf_value = vectorized_row[0, idx]
                    expected_idf = idf_values[idx]
                    if tfidf_value == 0:
                        print(f"[DEBUG] Token '{token}' in tokenizer vocabulary: {token in tokenizer.get_vocab()}")
                        print(f"[ERROR] Token '{token}' has IDF {expected_idf} but TF-IDF is 0.")
                else:
                    print(f"[WARNING] Token '{token}' has out-of-bounds index {idx}.")
            else:
                print(f"[ERROR] Token '{token}' not in vectorizer vocabulary.")

    print("finished spot checking docs.\n")