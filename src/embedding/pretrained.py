from abc import ABC, abstractmethod

import gensim
import os
import numpy as np
import requests
import zipfile

import torch, torchtext
from torchtext.vocab import GloVe as TorchTextGloVe
from transformers import BertModel, BertTokenizer
from transformers import  logging as transformers_logging

import joblib

AVAILABLE_PRETRAINED = ['glove', 'word2vec', 'fasttext', 'bert']

VECTOR_CACHE = '../.vector_cache'





# --- PJW Comments added ---
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
"""


"""
PretrainedEmbeddings Abstract Base Class: Serves as an abstract base class for all 
specific embedding classes, ensuring they implement required methods.

Methods:
- vocabulary(): An abstract method that subclasses must implement to return their vocabulary.

- dim(): An abstract method to return the dimensionality of the embeddings.

- reindex(cls, words, word2index): A class method to align indices of words in a given list 
with their indices in a pretrained model's vocabulary. This helps in efficiently extracting 
embedding vectors for a specific subset of words.
"""
class PretrainedEmbeddings(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def vocabulary(self): pass

    @abstractmethod
    def dim(self): pass

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

"""
GloVe Class: Manages GloVe embeddings by loading them using torchtext.

- Initialization: Takes parameters for the GloVe set to use (e.g., '840B'), the path 
where the vectors are cached, and optionally the maximum number of vectors to load.

Methods:
- extract(words): Extracts embeddings for a list of words, using the reindexing mechanism 
to handle words not found in GloVe’s vocabulary and replacing them with zeros.
"""
class GloVe(PretrainedEmbeddings):

    def __init__(self, setname='840B', path=VECTOR_CACHE, max_vectors=None):         # presumes running from bin directory
        
        super().__init__()
        
        print(f'Loading GloVe pretrained vectors from torchtext')
        # Initialize GloVe model from torchtext
        self.embed = TorchTextGloVe(name=setname, cache=path, max_vectors=max_vectors)
        print('Done')

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

    @staticmethod
    def reindex(words, word2index):
        source_idx, target_idx = [], []
        for i, word in enumerate(words):
            if word not in word2index:
                continue
            j = word2index[word]
            source_idx.append(i)
            target_idx.append(j)
        return source_idx, target_idx


"""
Word2Vec Class: Handles Word2Vec embeddings, loading them using gensim.

Initialization: Requires a path to the binary file containing Word2Vec embeddings, and 
optionally limits the number of vectors to load.

Methods:
extract(words): Similar to the GloVe class's method, it extracts embeddings, handling 
out-of-vocabulary words by zero-padding their vectors.
"""
class Word2Vec(PretrainedEmbeddings):

    def __init__(self, path, limit=None, binary=True):
        super().__init__()
        
        print(f'Loading word2vec format pretrained vectors from {path}')
        
        assert os.path.exists(path), print(f'pre-trained keyed vectors not found in {path}')
        self.embed = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary, limit=limit)
        
        #self.word2index = {w: i for i,w in enumerate(self.embed.index2word)}
        self.word2index = {w: i for i,w in enumerate(self.embed.index_to_key)}      # gensim 4.0
        
        print('Done')

    def vocabulary(self):
        return set(self.word2index.keys())

    def dim(self):
        return self.embed.vector_size

    def extract(self, words):
        print("Word2Vec::extract()...")
        source_idx, target_idx = PretrainedEmbeddings.reindex(words, self.word2index)
        extraction = np.zeros((len(words), self.dim()))
        extraction[source_idx] = self.embed.vectors[target_idx]
        extraction = torch.from_numpy(extraction).float()
        return extraction


"""
FastTextEmbeddings Class

Inheritance: Inherits from Word2Vec since FastText can be loaded in a similar way when the binary format is used.

Purpose: Adjusts loading mechanisms based on whether the embeddings are stored in a binary format or not.

Special Handling:
If a binary version of the file exists, it uses that directly. If only a textual version is available, it 
loads the text format and then saves it as a binary file for faster future loading.
"""
class FastText(Word2Vec):

    def __init__(self, path=VECTOR_CACHE, limit=None):

        pathvec = path
        pathbin = path+'.bin'
        pathzip = pathvec+'.zip'

        ft_emb_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"

        if os.path.exists(pathbin):
            print('open binary file')
            super().__init__(pathbin, limit, binary=True)
        elif os.path.exists(pathvec):
            print('open textual (.vec) file')
            super().__init__(path, limit, binary=False)
            print('saving as binary file')
            self.save_binary(pathbin)
            print('done')
        else:
            print("downlading embeddings from ", {ft_emb_url})
            self.ensure_emb_binary_exists(pathvec, ft_emb_url, pathzip)
             # After ensuring the file exists, initialize the embeddings
            if os.path.exists(pathbin):
                super().__init__(pathbin, limit, binary=True)
            else:
                super().__init__(pathvec, limit, binary=False)


    def ensure_emb_binary_exists(self, filename, url, zip_filename):
        """
        Checks if the specified .vec file exists.
        If it doesn't exist, download a .zip file from the given URL and unzip it.
        """
        # Check if the .vec file exists
        if not os.path.exists(filename):
            # If the .vec file does not exist, check if the .zip file exists
            if not os.path.exists(zip_filename):
                print(f"{zip_filename} not found, downloading...")
                # Download the .zip file
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(zip_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print("Download completed!")
            
            # Unzip the file in the directory of 'filename' without including the file name
            emb_dir = os.path.dirname(filename)
            print(f"Unzipping the file to {emb_dir}...")
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(path=emb_dir)
            print(f"{filename} is ready for use!")
        else:
            print(f"{filename} already exists!")


    def save_binary(self, path):
        self.embed.save_word2vec_format(path, binary=True)


"""
BERTEmbeddings class: class that inherits from the PretrainedEmbeddings abstract base class.

Initialization: Load a pre-trained BERT model and its tokenizer. BERT models work 
with tokens that might correspond to subwords or full words, and the tokenizer converts 
text to the format expected by the model.

Methods Implementation: Implement the required methods like vocabulary, dim, extract, etc., 
keeping in mind that BERT generates embeddings differently, typically by processing 
input through its transformer network.
"""
class BERT(PretrainedEmbeddings):

    def __init__(self, model_name='bert-base-uncased', cache_dir=VECTOR_CACHE, dataset_name='20newsgroups'):

        super().__init__()
        
        transformers_logging.set_verbosity_warning()                      # Set up transformers logging      

        print(f'Initializing', {model_name})

        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


        self.model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()                   # Set the model to inference mode
       
        self.cache_path = os.path.join(cache_dir)           # Define a path to store embeddings
        
        self.dataset = dataset_name
        self.model_name = model_name

        os.makedirs(self.cache_path, exist_ok=True)
        
        print('Done: BERT model and tokenizer are ready for use!')
        
    def vocabulary(self):
        # Returns the tokenizer's vocabulary as a set
        return set(self.tokenizer.get_vocab().keys())

    def dim(self):
        # Returns the hidden size of the BERT model
        return self.model.config.hidden_size

    def extract(self, words):

        print("Bert::extract()...")

        cache_file_name = self.model_name + '-' + self.dataset + '.pkl'
        cache_file = os.path.join(self.cache_path, cache_file_name)

        print("attempting to load embeddings from cache_file ", {cache_file})
        if os.path.exists(cache_file):
            return joblib.load(cache_file)

        inputs = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        joblib.dump(embeddings, cache_file)                 # Save the embeddings to disk
        return embeddings

    @staticmethod
    def reindex(words, word2index):
        # This method might be less relevant for BERT, depending on how you choose to handle subword tokens
        return super().reindex(words, word2index)
