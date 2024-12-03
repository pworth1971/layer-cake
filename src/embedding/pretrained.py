from abc import ABC, abstractmethod

import gensim
import os
import numpy as np
import requests
import zipfile
from tqdm import tqdm

import torch
from torchtext.vocab import GloVe as TorchTextGloVe

from transformers import BertModel, BertTokenizer
from transformers import  logging as transformers_logging


from transformers import BertModel, RobertaModel, GPT2Model, XLNetModel, DistilBertModel
from transformers import BertTokenizerFast, RobertaTokenizerFast, GPT2TokenizerFast, XLNetTokenizer, DistilBertTokenizerFast


# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 8
DEFAULT_GPU_BATCH_SIZE = 8
DEFAULT_MPS_BATCH_SIZE = 8

VECTOR_CACHE = "../.vector_cache"                               # cache directory for pretrained models


# Setup device prioritizing CUDA, then MPS, then CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    BATCH_SIZE = DEFAULT_GPU_BATCH_SIZE
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    BATCH_SIZE = DEFAULT_MPS_BATCH_SIZE
else:
    DEVICE = torch.device("cpu")
    BATCH_SIZE = DEFAULT_CPU_BATCH_SIZE



# -------------------------------------------------------------------------------------------------------
# default pretrained models.
#
# NB: these models are all case sensitive, ie no need to lowercase the input text (see _preprocess)
#
#GLOVE_MODEL = 'glove.6B.300d.txt'                          # dimension 300, case insensensitve
#GLOVE_MODEL = 'glove.42B.300d.txt'                          # dimensiomn 300, case sensitive
GLOVE_MODEL = 'glove.840B.300d.txt'                          # dimensiomn 300, case sensitive

WORD2VEC_MODEL = 'GoogleNews-vectors-negative300.bin'       # dimension 300, case sensitive

#FASTTEXT_MODEL = 'cc.en.300.bin'                            # dimension 300, case sensitive
FASTTEXT_MODEL = 'crawl-300d-2M-subword.bin'                # dimension 300, case sensitive

#BERT_MODEL = 'bert-base-cased'                              # dimension = 768, case sensitive
#BERT_MODEL = 'bert-base-uncased'                             # dimension = 1024, case insensitive
BERT_MODEL = 'bert-large-cased'                             # dimension = 1024, case sensitive

ROBERTA_MODEL = 'roberta-base'                             # dimension = 768, case insensitive
#ROBERTA_MODEL = 'roberta-large'                             # dimension = 1024, case sensitive

DISTILBERT_MODEL = 'distilbert-base-uncased'                 # dimension = 768, case insensitive

#XLNET_MODEL = 'xlnet-base-cased'                            # dimension = 768, case sensitive
XLNET_MODEL = 'xlnet-base-uncased'                            # dimension = 768, case insensitive
#XLNET_MODEL = 'xlnet-large-cased'                           # dimension = 1024, case sensitive

GPT2_MODEL = 'gpt2'                                          # dimension = 768, case sensitive
#GPT2_MODEL = 'gpt2-medium'                                   # dimension = 1024, case sensitive
#GPT2_MODEL = 'gpt2-large'                                    # dimension = 1280, case sensitive

LLAMA_MODEL = 'llama-7b-hf'                                  # dimension = 4096, case sensitive
# -------------------------------------------------------------------------------------------------------


# Define a default model mapping (optional) to avoid invalid identifiers
MODEL_MAP = {
    "glove": "glove.840B.300d",
    "word2vec": "GoogleNews-vectors-negative300",
    "fasttext": "crawl-300d-2M.vec",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "xlnet": "xlnet-base-uncased",
    "gpt2": "gpt2",
    "llama": "meta-llama/Llama-2-7b-chat-hf"  # Example for a possible LLaMA identifier
}


MAX_LENGTH = 512  # Max sequence length for the transformer models





from huggingface_hub import login

HF_TOKEN = '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'
HF_TOKEN2 = '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'


GLOVE_840B_300d_URL = 'https://nlp.stanford.edu/data/glove.840B.300d.zip'



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

    def __init__(self, setname='840B', path=VECTOR_CACHE, max_vectors=None):         # presumes running from bin directory
        
        super().__init__()

        self.type = 'glove'
        self.tokenizer = None

        print(f'Initializing GloVe class, loading GloVe pretrained vectors...')
        try:    
            embeddings_file = self.get_embeddings_file(path, GLOVE_840B_300d_URL)                                 # check to make sure GloVe embeddings are downloaded
            if (embeddings_file):
                 # Initialize GloVe model from torchtext
                print(f'embeddings file found at {embeddings_file}, loading embeddings using torchtext...')
                self.embed = TorchTextGloVe(name=setname, cache=path, max_vectors=max_vectors)     
            else:
                print(f'Error loading GloVe embeddings, cannot find embeddings file at {path}]')
                return
        except Exception as e:
            print(f"Error when trying to load GloVe embeddings from {path}. Error: {e}")
            raise


    def get_embeddings_file(self, target_dir, glove_url):
        
        print("GloVe::get_embeddings_file...")

        os.makedirs(target_dir, exist_ok=True)
        
        local_filename = os.path.join(target_dir, glove_url.split('/')[-1])

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
            with ZipFile(local_filename, 'r') as zip_ref:
                zip_ref.extractall(target_dir)

            print("Files extracted.")
            
        else:
            print("GloVe embeddings (txt file) found locally.")

        return local_filename.replace('.zip', '.txt')           # return text file, unzipped embeddings


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
        print("GloVe::extract()...")
        
        source_idx, target_idx = PretrainedEmbeddings.reindex(words, self.embed.stoi)
        extraction = torch.zeros((len(words), self.dim()))
        extraction[source_idx] = self.embed.vectors[target_idx]
        return extraction


    def extract_new(self, words):
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
        
        self.type = 'word2vec'
        self.tokenizer = None

        print(f'Loading word2vec format pretrained vectors from {path}')
        
        assert os.path.exists(path), print(f'pre-trained keyed vectors not found in {path}')
        self.embed = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary, limit=limit)
        
        #self.word2index = {w: i for i,w in enumerate(self.embed.index2word)}
        self.word2index = {w: i for i,w in enumerate(self.embed.index_to_key)}      # gensim 4.0
        
        print('Done')

    def get_model(self):
        return WORD2VEC_MODEL
    
    def get_type(self):
        return self.type

    def get_tokenizer(self):
        return self.tokenizer

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



class FastTextEmbeddings(Word2VecEmbeddings):

    def __init__(self, path, limit=None):

        self.type = 'fasttext'
        self.tokenizer = None

        pathbin = path+'.bin'
        if os.path.exists(pathbin):
            print('open binary file')
            super().__init__(pathbin, limit, binary=True)
        else:
            print('open textual file')
            super().__init__(path, limit, binary=False)
            print('saving as binary file')
            self.save_binary(pathbin)
            print('done')

    def get_model(self):
        return FASTTEXT_MODEL
    
    def get_type(self):
        return self.type
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def save_binary(self, path):
        self.embed.save_word2vec_format(path, binary=True)

        


class BERTEmbeddings(PretrainedEmbeddings):

    def __init__(self, path=None):

        super().__init__()

        print(f"Initializing BERTEmbeddings class with model {BERT_MODEL}, in path: {path}...")

        self.device = DEVICE
        self.batch_size = BATCH_SIZE
        self.type = 'bert'

        # instantiate model and tokenizer
        self.model = BertModel.from_pretrained(BERT_MODEL, cache_dir=path).to(self.device)
        
        self.tokenizer = BertTokenizerFast.from_pretrained(
            BERT_MODEL, 
            cache_dir=path,
            do_lower_case=False                 # keep tokenizer case sensitive
        )

        self.max_length = self.tokenizer.model_max_length
        
        self.mean_embedding = self._compute_mean_embedding()
           

    def get_model(self):
        return BERT_MODEL
    
    def get_type(self):
        return self.type

    def get_tokenizer(self):
        return self._custom_tokenizer
    

    def _custom_tokenizer(self, text):
        """
        Tokenize the text using the tokenizer, returning tokenized strings (not token IDs) for TF-IDF or CountVectorizer.
        This tokenizer works for BERT, RoBERTa, and LLaMA models.
        
        Parameters:
        - text: The input text to be tokenized.
        
        Returns:
        - tokens: A list of tokens with special tokens removed based on the model in use.
        """

        tokens = self.tokenizer.tokenize(text, max_length=self.max_length, truncation=True)
        
        # Retrieve special tokens from the tokenizer object
        special_tokens = self.tokenizer.all_special_tokens                  # Dynamically fetch special tokens like [CLS], [SEP], <s>, </s>, etc.
        
        # Optionally, remove special tokens
        tokens = [token for token in tokens if token not in special_tokens]

        return tokens
    

    def _tokenize(self, texts):
        """
        Tokenize a batch of texts using `encode_plus` to ensure truncation and padding.
        """
        input_ids = []
        attention_masks = []

        for text in texts:
            # Use encode_plus to handle truncation, padding, and return attention mask
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,                                # Add special tokens like [CLS] and [SEP]
                max_length=self.max_length,                             # Truncate sequences to this length
                padding='max_length',                                   # Pad sequences to the max_length
                return_attention_mask=True,                             # Generate attention mask
                return_tensors='pt',                                    # Return PyTorch tensors
                truncation=True                                         # Ensure truncation to max_length
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        # Convert lists of tensors to a single tensor
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks
         

    def _compute_mean_embedding(self):
        """
        Compute the mean embedding vector using all tokens in the model's vocabulary.
        This vector will be used for handling OOV tokens.
        """
        vocab = list(self.tokenizer.get_vocab().keys())
        batch_size = self.batch_size
        total_embeddings = []

        # Process the tokens in batches with a progress bar
        with tqdm(total=len(vocab), desc="Computing mean embedding for model vocabulary", unit="token") as pbar:
            for i in range(0, len(vocab), batch_size):
                batch_tokens = vocab[i:i + batch_size]
                
                # Tokenize the batch of tokens, ensuring attention_mask is created
                inputs = self.tokenizer(batch_tokens, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)  # Ensure attention_mask is passed

                # Pass through the model to get token embeddings
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)  # Pass attention_mask
                token_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                total_embeddings.append(token_embeddings)

                # Update the progress bar with the actual batch size
                pbar.update(len(batch_tokens))

        # Concatenate all embeddings and compute the mean embedding
        total_embeddings = np.concatenate(total_embeddings, axis=0)
        mean_embedding = total_embeddings.mean(axis=0)  # Compute the mean embedding across all tokens
        print(f"Mean embedding shape: {mean_embedding.shape}")

        return mean_embedding

    
    def vocabulary(self):
        return set(self.tokenizer.get_vocab().keys())

    def dim(self):
        return self.model.config.hidden_size


    def extract(self, words, batch_size=BATCH_SIZE):
        """
        Extracts embeddings for a list of words, aggregating subword embeddings if necessary.

        Parameters:
        ----------
        words : list of str
            List of words for which embeddings are to be extracted.
        batch_size : int, optional, default=64
            Number of words to process in a single batch for efficiency.

        Returns:
        -------
        torch.Tensor
            A tensor of shape (len(words), embedding_dim) containing the word embeddings.
            Aggregates embeddings for multi-token words.
        """
        print("BERT::extract()...")
        embeddings = []
        
        # Process words in batches
        with tqdm(total=len(words), desc="extracting word embeddings from BERT model", unit="word") as pbar:
            for i in range(0, len(words), batch_size):
                batch_words = words[i:i + batch_size]

                # Tokenize batch into subwords
                tokenized = self.tokenizer(
                    batch_words,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)

                # Pass through the model to get embeddings
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Get the embeddings for all subwords
                token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

                # Aggregate subword embeddings for each word (e.g., mean pooling)
                batch_embeddings = token_embeddings.mean(dim=1)  # Shape: (batch_size, hidden_size)

                # Append to the results
                embeddings.append(batch_embeddings)

                # Update progress bar
                pbar.update(len(batch_words))

        # Concatenate all batch embeddings
        embeddings = torch.cat(embeddings, dim=0)  # Shape: (len(words), hidden_size)

        return embeddings





    @staticmethod
    def reindex(words, word2index):
        source_idx, target_idx = [], []
        for i, word in enumerate(words):
            if word not in word2index:
                continue
            j = word2index[word]
            source_idx.append(i)
            target_idx.append(j)
        source_idx = np.asarray(source_idx)
        target_idx = np.asarray(target_idx)
        return source_idx, target_idx
    





"""
FastTextEmbeddings Class

Inheritance: Inherits from Word2VecEmbeddings since FastText can be loaded in a similar way when the binary format is used.

Purpose: Adjusts loading mechanisms based on whether the embeddings are stored in a binary format or not.

Special Handling:
If a binary version of the file exists, it uses that directly. If only a textual version is available, it 
loads the text format and then saves it as a binary file for faster future loading.
"""
class FastText(Word2VecEmbeddings):

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
            #print(f"{filename} is ready for use!")
        else:
            print(f"{filename} already exists!")


    def save_binary(self, path):
        self.embed.save_word2vec_format(path, binary=True)

