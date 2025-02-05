from abc import ABC, abstractmethod
import gensim
import os
import numpy as np
from tqdm import tqdm

import torch
from torchtext.vocab import GloVe as TorchTextGloVe

from model.LCRepresentationModel import VECTOR_CACHE, GLOVE_MODEL, WORD2VEC_MODEL, FASTTEXT_MODEL

# ----------------------------------------------------------------------------------------------------------------------------
#
# pretrained models we are using for legacy Neural Model (supports word based models)
#
AVAILABLE_PRETRAINED = ['glove', 'word2vec', 'fasttext']        

GLOVE_840B_300d_URL = 'https://nlp.stanford.edu/data/glove.840B.300d.zip'

GLOVE_SET = '840B'                                          # GloVe set to use
#
# ----------------------------------------------------------------------------------------------------------------------------






class PretrainedEmbeddings(ABC):
    """
    Python module for handling pretrained word embeddings, structured using an object-oriented approach. Includes classes for 
    managing embeddings from GloVe, Word2Vec, and FastText, each subclassing an abstract base class defined for generalized 
    pretrained embeddings.

    General Usage and Functionality
    These classes are designed to make it easy to integrate different types of word embeddings into various NLP models.
    They provide a uniform interface to access the embeddings’ vocabulary, dimensionality, and actual vector representations.
    The design follows the principle of encapsulation, keeping embedding-specific loading and access mechanisms 
    internal to each class while presenting a consistent external interface. This modular and abstracted approach 
    allows for easy extensions and modifications, such as adding support for new types of embeddings or changing 
    the underlying library used for loading the embeddings.

    PretrainedEmbeddings Abstract Base Class: Serves as an abstract base class for all 
    specific embedding classes, ensuring they implement required methods.

    Methods:
    - vocabulary(): An abstract method that subclasses must implement to return their vocabulary.
    - dim(): An abstract method to return the dimensionality of the embeddings.
    - reindex(cls, words, word2index): A class method to align indices of words in a given list 
    with their indices in a pretrained model's vocabulary. This helps in efficiently extracting 
    embedding vectors for a specific subset of words.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def vocabulary(self): pass

    @abstractmethod
    def dim(self): pass

    @abstractmethod
    def extract(self, words): pass

    @classmethod
    def reindex(cls, words, word2index):
        source_idx, target_idx = [], []
        for i, word in enumerate(words):
            if word not in word2index: continue
            j = word2index[word]
            source_idx.append(i)
            target_idx.append(j)
        source_idx = np.asarray(source_idx)
        target_idx = np.asarray(target_idx)
        return source_idx, target_idx
    
   


class GloVeEmbeddings(PretrainedEmbeddings):
    """
    GloVeEmbeddings Class: Manages GloVe embeddings by loading them using torchtext.

    - Initialization: Takes parameters for the GloVe set to use (e.g., '840B'), the path 
    where the vectors are cached, and optionally the maximum number of vectors to load.

    Methods:
    - extract(words): Extracts embeddings for a list of words, using the reindexing mechanism 
    to handle words not found in GloVe’s vocabulary and replacing them with zeros.
    """

    def __init__(self, setname=GLOVE_SET, model_name=GLOVE_MODEL, path=VECTOR_CACHE, max_vectors=None):         # presumes running from bin directory
        
        super().__init__()

        print(f'GloVeEmbeddings::__init__() setname={setname}, model_name: {model_name}, path={path}, max_vectors={max_vectors}')

        self.type = 'glove'
        self.tokenizer = None
        dir = path
        path = path + '/' + model_name
        print(f'loading GloVe embeddings from {path}...')

        print(f'Initializing GloVe class, loading GloVe pretrained vectors...')
        try:    
            embeddings_file = self.get_embeddings_file(path, GLOVE_840B_300d_URL)                                 # check to make sure GloVe embeddings are downloaded
            if (embeddings_file):
                 # Initialize GloVe model from torchtext
                print(f'embeddings file found at {embeddings_file}, loading embeddings using torchtext...')
                #self.embed = TorchTextGloVe(name=setname, cache=path, max_vectors=max_vectors)    
                self.embed = TorchTextGloVe(name=setname, cache=dir, max_vectors=max_vectors)     
            else:
                print(f'Error loading GloVe embeddings, cannot find embeddings file at {path}]')
                return
        except Exception as e:
            print(f"Error when trying to load GloVe embeddings from {path}. Error: {e}")
            raise


    def get_embeddings_file(self, target_dir, glove_url):

        import requests
        import zipfile
                
        print("GloVe::get_embeddings_file...")

        #os.makedirs(target_dir, exist_ok=True)
        
        #local_filename = os.path.join(target_dir, glove_url.split('/')[-1])
        local_filename = target_dir
        print("local_filename:", local_filename)
        
        if not os.path.exists(local_filename.replace('.zip', '.txt')):

            print(f"GloVe embeddings not found locally. Downloading from {glove_url}...")
            
            response = requests.get(glove_url, stream=True)

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

            with open(local_filename, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                print("ERROR, something went wrong")
                return None
        
            print("Download complete. Extracting files...")
            with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                zip_ref.extractall(target_dir)

            print("Files extracted.")
            
        else:
            print("GloVe embeddings (txt file) found locally.")

        #return local_filename.replace('.zip', '.txt')           # return text file, unzipped embeddings
        return local_filename

    def get_model(self):
        return GLOVE_MODEL

    def get_type(self):
        return self.type

    def get_tokenizer(self):
        return self.tokenizer
    
    def vocabulary(self):
        # accessing the vocabulary
        return set(self.embed.stoi.keys())

    def dim(self):
        # getting the dimension of the embeddings
        return self.embed.dim

    def extract(self, words):
        # Extract embeddings for a list of words
        print("GloVe::extract()...")

        source_idx, target_idx = self.reindex(words, self.embed.stoi)
        extraction = torch.zeros((len(words), self.dim()), dtype=torch.float)
        valid_source_idx = torch.tensor(source_idx, dtype=torch.long)
        valid_target_idx = torch.tensor(target_idx, dtype=torch.long)
        extraction[valid_source_idx] = self.embed.vectors[valid_target_idx]
        
        return extraction

    

class Word2VecEmbeddings(PretrainedEmbeddings):
    """
    Word2VecEmbeddings Class: Handles Word2Vec embeddings, loading them using gensim.

    Initialization: Requires a path to the binary file containing Word2Vec embeddings, and 
    optionally limits the number of vectors to load.

    Methods:
    extract(words): Similar to the GloVe class's method, it extracts embeddings, handling 
    out-of-vocabulary words by zero-padding their vectors.
    """
    def __init__(self, path, limit=None, binary=True):
        
        super().__init__()
        
        print(f'Word2VecEmbeddings::__init__... path={path}, limit={limit}, binary={binary}')

        self.type = 'word2vec'
        self.tokenizer = None

        print(f'loading embeddings from {path}...')

        assert os.path.exists(path), print(f'pre-trained keyed vectors not found in {path}')
        self.embed = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary, limit=limit)
        
        #self.word2index = {w: i for i,w in enumerate(self.embed.index2word)}
        self.word2index = {w: i for i,w in enumerate(self.embed.index_to_key)}      # gensim 4.0
        
        #print('Done')

    def get_model(self):
        return WORD2VEC_MODEL
    
    def get_type(self):
        return self.type

    def get_tokenizer(self):
        return self.tokenizer

    def vocabulary(self):
        """
        Returns the vocabulary for the Word2Vec model.
        """
        return set(self.embed.index_to_key)  # Updated for gensim 4.x
    
    def dim(self):
        return self.embed.vector_size

    def extract(self, words):
        print("Word2Vec::extract()...")
        source_idx, target_idx = PretrainedEmbeddings.reindex(words, self.word2index)
        extraction = np.zeros((len(words), self.dim()))
        extraction[source_idx] = self.embed.vectors[target_idx]
        extraction = torch.from_numpy(extraction).float()

        return extraction




# ----------------------------------------------------------------------------------------------------------------
# 
# Fasttext embeddings download URL:
# https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
#
#


class FastTextEmbeddings(Word2VecEmbeddings):

    def __init__(self, path, limit=None):
        
        #pathbin = path+'.bin'
                
        print(f'loading FastText embeddings from {path}...')

        self.type = 'fasttext'

        if path.endswith('.bin'):  # Check for binary format
         
            if os.path.exists(path):
                print('open binary file')
                super().__init__(path, limit, binary=True)

        elif path.endswith('.vec'):  # vector file format

            if os.path.exists(path):
                print('open textual file')
                super().__init__(path, limit, binary=False)            
                print('saving as binary file')
                self.save_binary(pathbin)
                
        else:
            raise ValueError('FastText embeddings must be in .vec or .bin format')

    def save_binary(self, path):
        self.embed.save_word2vec_format(path, binary=True)

from gensim.models.fasttext import FastText
import torch

class FastTextEmbeddings310(Word2VecEmbeddings):
    """
    FastTextEmbeddings Class: Handles FastText embeddings with subword support.
    Inherits from PretrainedEmbeddings and aligns its interface with Word2VecEmbeddings.
    """
    
    def __init__(self, path, limit=None):
        """
        Initializes the FastText embeddings. If a binary version exists, it loads it;
        otherwise, it loads the text version and saves it as binary for future use.

        Parameters:
        ----------
        path : str
            Path to the FastText embedding file (either .vec or .bin).
        limit : int, optional
            Limit on the number of embeddings to load.
        """

        self.type = 'fasttext'

        print(f'loading FastText embeddings from {path}...')

        if path.endswith('.bin'):  # Check for binary format
            print('Binary format detected. Loading using FastText binary loader.')
            self.embed = FastText.load_fasttext_format(path)
        else:
            print('Text format detected. Loading using gensim KeyedVectors...')
            self.embed = FastText.load(path)  # Using gensim's FastText loading mechanism for .vec files
            print('saving as binary file')
            self.save_binary(pathbin)
            print('Binary file saved.')

        # Build the word2index mapping
        print("Building word2index mapping...")
        self.word2index = {word: idx for idx, word in enumerate(self.embed.wv.index_to_key)}

        # Initialize the tokenizer with a subword tokenizer using a fixed n-gram size
        self.n = 3                                          # Default n-gram size

        # we use gensim default tokenizer 
        self.tokenizer = self._subword_tokenizer


        
    def _subword_tokenizer(self, text):
        """
        Tokenizes text into character n-grams for subword embeddings.

        Parameters:
        ----------
        text : str
            Input text to tokenize.
        n : int
            Length of character n-grams.

        Returns:
        -------
        list of str
            List of character n-grams.
        """
        tokens = []
        words = text.split()
        for word in words:
            word = f"<{word}>"
            tokens.extend([word[i:i+self.n] for i in range(len(word) - self.n + 1)])
        return tokens


    def get_tokenizer(self):
        #return self.tokenizer
        return None


    def get_model(self):
        return FASTTEXT_MODEL


    def get_type(self):
        return self.type


    def save_binary(self, path):
        self.embed.save_word2vec_format(path, binary=True)


    def vocabulary(self):
        """
        Returns the vocabulary for the FastText model.
        """
        return set(self.embed.wv.index_to_key)


    def dim(self):
        """
        Returns the dimensionality of the embeddings.
        """
        return self.embed.wv.vector_size

    def extract(self, words):
        print("FastText::extract()...")
        extraction = np.zeros((len(words), self.dim()), dtype=np.float32)
        for i, word in enumerate(words):
            extraction[i] = self.embed.wv.get_vector(word)
        return torch.from_numpy(extraction).float()



import fasttext
import fasttext.util
from gensim.models import KeyedVectors


class FastTextEmbeddings312(Word2VecEmbeddings):

    def __init__(self, path, limit=None):

        self.type = 'fasttext'

        #
        # Load the FastText model using fasttext package
        #
        print(f'loading fastText embeddings from {path}...')

        if path.endswith('.bin'):  # Check for binary format
            print('Binary format detected. Loading using FastText binary loader.')
            super().__init__(path, limit, binary=True)
        elif path.endswith('.vec'):  # Check for text format
            print('Text format detected. Loading using gensim KeyedVectors...')

            if os.path.exists(pathbin):
                print('open textual file')
                super().__init__(path, limit, binary=False)
                print('saving as binary file')
                self.save_binary(pathbin)

         # Initialize the tokenizer with a subword tokenizer using a fixed n-gram size
        self.n = 3              # Default n-gram size
        #self.tokenizer = lambda text: self._subword_tokenizer(text, n)


    def _subword_tokenizer(self, text):
        """
        Tokenizes text into character n-grams for subword embeddings.

        Parameters:
        ----------
        text : str
            Input text to tokenize.
        n : int
            Length of character n-grams.

        Returns:
        -------
        list of str
            List of character n-grams.
        """
        tokens = []
        words = text.split()
        for word in words:
            word = f"<{word}>"
            tokens.extend([word[i:i+self.n] for i in range(len(word) - self.n + 1)])
        return tokens

    def get_tokenizer(self):
        #return self.tokenizer
        return self._subword_tokenizer

    def get_model(self):
        return FASTTEXT_MODEL
    
    def get_type(self):
        return self.type

    def vocabulary(self):
        return set(self.embed.key_to_index.keys())
    
    def dim(self):
        return self.embed.vector_size


    def extract(self, words):
        print("FastText::extract()...")

        # Initialize an array for embeddings
        extraction = np.zeros((len(words), self.dim()), dtype=np.float32)

        for i, word in enumerate(words):
            try:
                # FastText handles subwords internally if the word is not directly in the vocabulary
                extraction[i] = self.embed.get_vector(word)
            except KeyError:
                # This should not occur with FastText as it generates embeddings even for OOV words
                print(f"FastText::extract()... word '{word}' not found in vocabulary")
                continue

        return torch.from_numpy(extraction).float()


    def save_binary(self, path):
        self.embed.save_word2vec_format(path, binary=True)