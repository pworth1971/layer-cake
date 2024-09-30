import numpy as np
import torch
from tqdm import tqdm
import os
import requests

from abc import ABC, abstractmethod

from simpletransformers.language_representation import RepresentationModel
from transformers import BertModel, RobertaModel, GPT2Model, XLNetModel
from transformers import BertTokenizerFast, RobertaTokenizerFast, GPT2TokenizerFast, XLNetTokenizer

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.fasttext import load_facebook_model

from joblib import Parallel, delayed

from util.common import VECTOR_CACHE



NUM_JOBS = -1           # number of jobs for parallel processing

# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 16
DEFAULT_GPU_BATCH_SIZE = 8
MPS_BATCH_SIZE = 16


# Setup device prioritizing CUDA, then MPS, then CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    BATCH_SIZE = DEFAULT_GPU_BATCH_SIZE
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    BATCH_SIZE = MPS_BATCH_SIZE
else:
    DEVICE = torch.device("cpu")
    BATCH_SIZE = DEFAULT_CPU_BATCH_SIZE



# -------------------------------------------------------------------------------------------------------
# default pretrained models.
#
# NB: these models are all case sensitive, ie no need to lowercase the input text (see _preprocess)
#
GLOVE_MODEL = 'glove.6B.300d.txt'                          # dimension 300, case insensensitve
#GLOVE_MODEL = 'glove.42B.300d.txt'                          # dimensiomn 300, case sensitive
#GLOVE_MODEL = 'glove.840B.300d.txt'                          # dimensiomn 300, case sensitive

WORD2VEC_MODEL = 'GoogleNews-vectors-negative300.bin'       # dimension 300, case sensitive

FASTTEXT_MODEL = 'crawl-300d-2M-subword.bin'                # dimension 300, case sensitive

BERT_MODEL = 'bert-base-cased'                              # dimension = 768, case sensitive
#BERT_MODEL = 'bert-large-cased'                             # dimension = 1024, case sensitive

ROBERTA_MODEL = 'roberta-base'                             # dimension = 768, case insensitive
#ROBERTA_MODEL = 'roberta-large'                             # dimension = 1024, case sensitive

GPT2_MODEL = 'gpt2'                                          # dimension = 768, case sensitive
#GPT2_MODEL = 'gpt2-medium'                                   # dimension = 1024, case sensitive
#GPT2_MODEL = 'gpt2-large'                                    # dimension = 1280, case sensitive

XLNET_MODEL = 'xlnet-base-cased'                            # dimension = 768, case sensitive
#XLNET_MODEL = 'xlnet-large-cased'                           # dimension = 1024, case sensitive

# -------------------------------------------------------------------------------------------------------

MAX_VOCAB_SIZE = 15000                                      # max feature size for TF-IDF vectorization

#
# tokens for LLAMA model access, must be requested from huggingface
#
from huggingface_hub import login

HF_TOKEN = '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'
HF_TOKEN2 = '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'



class LCRepresentationModel(RepresentationModel, ABC):
    """
    class LCRepresentationModel(RepresentationModel): 
    inherits from the transformer RepresentationModel class and acts as an abstract base class which
    computes representations of dataset text for different pretrained language models. This version 
    implements an encoding method that returns a summary vector for the different models as well, with 
    some performance optimizations.
    """
    
    def __init__(self, model_name=None, model_dir='../.vector_cache'):
        """
        Initialize the representation model.
        
        Args:
        - model_type (str): Type of the model ('bert', 'roberta', etc.).
        - model_name (str): Hugging Face Model Hub ID or local path.
        - model_dir (str, optional): Directory to save and load model files.
        - device (str, optional): Device to use for encoding ('cuda', 'mps', or 'cpu').
        """
        
        print("initializing LCRepresentationModel...")

        self.device = DEVICE
        self.batch_size = BATCH_SIZE
        print("self.device:", self.device)
        print("self.batch_size:", self.batch_size)

        self.model_name = model_name
        print("self.model_name:", self.model_name)
        
        self.model_dir = model_dir
        print("self.model_dir:", model_dir)

        self.combine_strategy = 'mean'          # default combine strategy  

        self.path_to_embeddings = model_dir + '/' + model_name
        print("self.path_to_embeddings:", self.path_to_embeddings)

        self.model = None
        self.tokenizer = None

        self.initialized = True


    def _download_file(self, url, destination):
        """
        Helper function to download a file from a URL.
        """
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(1024):
                file.write(data)
                bar.update(len(data))


    def _unzip_embeddings(self, zip_file_path):
        """
        Unzip embeddings.
        """
        from zipfile import ZipFile, BadZipFile

        print(f'unzipping embeddings... zip_file: {zip_file_path}')

        try:        
            with ZipFile(zip_file_path, 'r') as zip_ref:
                print(f"Extracting GloVe embeddings from {zip_file_path}...")
                zip_ref.extractall(os.path.dirname(zip_file_path))

            # Delete the zip file after extraction
            os.remove(zip_file_path)
            print(f"Deleted zip file {zip_file_path} after extraction.")

        except BadZipFile:
            print(f"Error: {zip_file_path} is not a valid ZIP file or is corrupted.")
            raise


    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    

    @abstractmethod
    def encode_docs(self, text_list, embedding_vocab_matrix):
        """
        Abstract method to be implemented by all subclasses.
        """
        pass




class WordLCRepresentationModel(LCRepresentationModel):
    """
    WordBasedRepresentationModel handles word-based embeddings (e.g., Word2Vec, GloVe).
    It computes sentence embeddings by averaging, summing, or computing TF-IDF weighted embeddings.
    """

    def __init__(self, model_name, model_dir, vtype='tfidf', model_type='word2vec'):
        """
        Initialize the word-based representation model (e.g., Word2Vec, GloVe).
        
        Parameters:
        ----------
        model_name : str
            Name of the pre-trained word embedding model (e.g., 'word2vec', 'glove').
        embedding_path : str
            Path to the pre-trained embedding file (e.g., 'GoogleNews-vectors-negative300.bin').
        device : str, optional
            Device to use for encoding ('cpu' for word embeddings since it's lightweight).
        """
        print("Initializing WordBasedRepresentationModel...")

        super().__init__(model_name, model_dir=model_dir)  # parent constructor

        # Automatically download embeddings if not present
        if not os.path.exists(self.path_to_embeddings):
            print(f"Embedding file {self.path_to_embeddings} not found. Downloading...")
            self._download_embeddings(model_name, model_dir, model_type)

        # Load the approproate model type
        if (model_type == 'word2vec'):
            print("Using Word2Vec pretrained embeddings...")
            self.model = KeyedVectors.load_word2vec_format(self.path_to_embeddings, binary=True)
        elif (model_type == 'glove'):
            print("Using GloVe pretrained embeddings...")
            from gensim.scripts.glove2word2vec import glove2word2vec
            glove_input_file = self.path_to_embeddings
            word2vec_output_file = glove_input_file + '.word2vec'
            glove2word2vec(glove_input_file, word2vec_output_file)
            self.model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        else:
            raise ValueError("Invalid model type. Use 'word2vec' or 'glove'.")
        
        self.vtype = vtype
        print(f"Vectorization type: {vtype}")

        # Get embedding size (dimensionality)
        self.embedding_dim = self.model.vector_size
        print(f"Embedding dimension: {self.embedding_dim}")
        
        #
        # vectorize the text, note that the Word2Vec and GloVe models we use are case sensitive
        #
        if vtype == 'tfidf':
            print("using TF-IDF vectorization...")
            
            self.vectorizer = TfidfVectorizer(
                #min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                sublinear_tf=True,                              # use sublinear TF scaling
                lowercase=False                                 # dont lowercase the tokens
            )              
        elif vtype == 'count':
            print("using Count vectorization...")

            self.vectorizer = CountVectorizer(
                #min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                lowercase=False                                 # dont lowercase the tokens
            )
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")


    def _download_embeddings(self, model_name, model_dir, model_type):
        """
        Download pre-trained embeddings (Word2Vec or GloVe) from a URL and save them to the specified path.
        """

        print(f'downloading embeddings... model:{model_name}, model_type:{model_type}, path:{model_dir}')

        #
        # TODO: This URL is not correct, pls dowwnload these embeddings offline 
        # from kaggle here: https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300
        #
        if model_type == 'word2vec':
            # Word2Vec Google News embeddings (Commonly hosted on Google Drive)
            url = "https://drive.usercontent.google.com/download?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download&authuser=1"
        elif model_type == 'glove':
            # GloVe embeddings (Stanford)
            if (GLOVE_MODEL == 'glove.6B.300d.txt'):
                url = f"https://nlp.stanford.edu/data/glove.6B.zip"
                zipfile = 'glove.6B.zip'
            elif (GLOVE_MODEL == 'glove.42B.300d.txt'):    
                url = f"https://nlp.stanford.edu/data/glove.42B.300d.zip"
                zipfile = 'glove.42B.zip'
            elif (GLOVE_MODEL == 'glove.840B.300d.txt'):
                url = f"https://nlp.stanford.edu/data/glove.840B.300d.zip"
                zipfile = 'glove.840B.zip'
        else:
            raise ValueError(f"Unsupported model_type {model_type} for download.")

        # Download the file
        if model_type == 'glove':
            dest_zip_file = model_dir + '/' + zipfile
            print(f"Downloading embeddings from {url} to {dest_zip_file}...")
            self._download_file(url, dest_zip_file)

            # Unzip GloVe embeddings
            self._unzip_embeddings(dest_zip_file)

        elif model_type == 'word2vec':
            print(f"Downloading embeddings from {url} to {self.path_to_embeddings}...")
            self._download_file(url, self.path_to_embeddings)


    def build_embedding_vocab_matrix(self):
        
        print("building [word based] embedding representation (matrix) of dataset vocabulary...")

        print("model:", type(self.model))
        print("model.vector_size:", self.model.vector_size)

        self.embedding_dim = self.model.vector_size
        self.vocab_size = len(self.vectorizer.vocabulary_)
        print("embedding_dim:", self.embedding_dim)
        print("vocab_size:", self.vocab_size)

        self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        # Dictionary to store the index-token mapping
        self.token_to_index_mapping = {}

        # Calculate the mean of all embeddings in the model as a fallback for OOV tokens
        mean_vector = np.mean(self.model.vectors, axis=0)
    
        oov_tokens = 0

        # Loop through the dataset vocabulary and fill the embedding matrix
        for word, idx in self.vectorizer.vocabulary_.items():
            self.token_to_index_mapping[idx] = word  # Store the token at its index

            # Check for the word in the original case first
            if word in self.model.key_to_index:
                self.embedding_vocab_matrix[idx] = self.model[word]
            # If not found, check the lowercase version of the word
            elif word.lower() in self.model.key_to_index:
                self.embedding_vocab_matrix[idx] = self.model[word.lower()]
            # If neither is found, handle it as an OOV token
            else:
                #print(f"Warning: OOV word when building embedding vocab matrix. Word: '{word}'")
                self.embedding_vocab_matrix[idx] = mean_vector  # Use the mean of all vectors as a substitute
                oov_tokens += 1

        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        print("token_to_index_mapping:", type(self.token_to_index_mapping), len(self.token_to_index_mapping))
        print("oov_tokens:", oov_tokens)

        return self.embedding_vocab_matrix, self.token_to_index_mapping


    def encode_docs(self, texts, embedding_vocab_matrix):
        """
        Compute both weighted document embeddings (using TF-IDF) and average document embeddings for each document.
        
        Args:
        - texts: List of input documents (as raw text).
        - embedding_vocab_matrix: Matrix of pre-trained word embeddings (e.g., Word2Vec, GloVe).

        Returns:
        - weighted_document_embeddings: Numpy array of weighted document embeddings for each document.
        - avg_document_embeddings: Numpy array of average document embeddings for each document.
        """
        
        print(f"encoding docs...")
        
        print("texts:", type(texts), len(texts))
        print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

        weighted_document_embeddings = []
        avg_document_embeddings = []

        unk_token_id = self.vectorizer.vocabulary_.get('<unk>', None)  # Check for the existence of an UNK token
        print("unk_token_id:", unk_token_id)
        
        # Compute the mean embedding for OOV tokens across the entire embedding matrix
        # NB self.embedding_vocab_matrix is computed in build_embedding_vocab_matrix
        self.mean_embedding = np.mean(self.embedding_vocab_matrix, axis=0)
        print(f"Mean embedding vector for OOV tokens calculated: {self.mean_embedding.shape}")

        oov_tokens = 0

        for doc in texts:
            # Tokenize the document using the vectorizer (ensures consistency in tokenization)
            tokens = self.vectorizer.build_analyzer()(doc)

            # Calculate TF-IDF weights for the tokens
            tfidf_vector = self.vectorizer.transform([doc]).toarray()[0]

            weighted_sum = np.zeros(embedding_vocab_matrix.shape[1])
            total_weight = 0.0
            valid_embeddings = []    

            for token in tokens:
                # Get the token's index in the vocabulary
                token_id = self.vectorizer.vocabulary_.get(token, None)

                if token_id is not None and 0 <= token_id < embedding_vocab_matrix.shape[0]:
                    # Get the embedding for the token from the embedding matrix
                    embedding = embedding_vocab_matrix[token_id]
                    # Get the TF-IDF weight for the token
                    weight = tfidf_vector[token_id]
                    
                    # Accumulate the weighted embedding
                    weighted_sum += embedding * weight
                    total_weight += weight
                elif unk_token_id is not None:
                    # Fallback to <unk> embedding if available
                    embedding = embedding_vocab_matrix[unk_token_id]
                else:
                    # Use the mean embedding for OOV tokens
                    embedding = self.mean_embedding
                    oov_tokens += 1

                valid_embeddings.append(embedding)
                    
            # Compute the weighted embedding for the document
            if total_weight > 0:
                weighted_doc_embedding = weighted_sum / total_weight
            else:
                weighted_doc_embedding = np.zeros(embedding_vocab_matrix.shape[1])  # Handle empty or OOV cases

            # Compute the average embedding for the document
            if valid_embeddings:
                avg_doc_embedding = np.mean(valid_embeddings, axis=0)
            else:
                avg_doc_embedding = np.zeros(embedding_vocab_matrix.shape[1])  # Handle empty or OOV cases

            weighted_document_embeddings.append(weighted_doc_embedding)
            avg_document_embeddings.append(avg_doc_embedding)

        print("weighted_document_embeddings:", type(weighted_document_embeddings), len(weighted_document_embeddings))
        print("avg_document_embeddings:", type(avg_document_embeddings), len(avg_document_embeddings))

        print("oov_tokens:", oov_tokens)

        return np.array(weighted_document_embeddings), np.array(avg_document_embeddings)
    




class SubWordLCRepresentationModel(LCRepresentationModel):
    """
    SubWordBasedRepresentationModel handles subword-based embeddings (e.g. fastText).
    It computes sentence embeddings by averaging, summing, or computing TF-IDF weighted embeddings.
    """

    def __init__(self, model_name=FASTTEXT_MODEL, model_dir=VECTOR_CACHE+'/fastText', vtype='tfidf'):
        """
        Initialize the word-based representation model (e.g., Word2Vec, GloVe).
        
        Parameters:
        ----------
        model_name : str
            Name of the pre-trained word embedding model (e.g. 'crawl-300d-2M-subword.bin').
        model_dir : str
            Path to the pre-trained embedding file (e.g., '../.vector_cache/fastText').
        vtype : str, optional
            vectorization type, either 'tfidf' or 'count'.
        """

        print("Initializing SubWordBasedRepresentationModel...")

        super().__init__(model_name, model_dir=model_dir)  # parent constructor

        # Automatically download embeddings if not present
        if not os.path.exists(self.path_to_embeddings):
            print(f"Embedding file {self.path_to_embeddings} not found. Downloading...")
            self._download_extract_embeddings(model_name, model_dir)

        # Use load_facebook_model to load FastText model from Facebook's pre-trained binary files
        self.model = load_facebook_model(self.path_to_embeddings)
            
        self.vtype = vtype
        print(f"Vectorization type: {vtype}")

        # Get embedding size (dimensionality)
        self.embedding_dim = self.model.vector_size
        print(f"Embedding dimension: {self.embedding_dim}")

        #
        # vectorize the text, note that the Word2Vec and GloVe models we use are case sensitive
        #
        if vtype == 'tfidf':
            print("using TF-IDF vectorization...")
            
            self.vectorizer = TfidfVectorizer(
                #min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                sublinear_tf=True,                              # use sublinear TF scaling
                lowercase=False                                 # dont lowercase the tokens
            )              
        elif vtype == 'count':
            print("using Count vectorization...")

            self.vectorizer = CountVectorizer(
                #min_df=MIN_DF_COUNT,                            # ignore terms that have a document frequency strictly lower than the given threshold
                lowercase=False                                 # dont lowercase the tokens
            )
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")


    def _download_extract_embeddings(self, model_name, model_dir):
        """
        Download pre-trained embeddings (Word2Vec or GloVe) from a URL and save them to the specified path.
        """
        
        from zipfile import ZipFile, BadZipFile

        print(f'downloading embeddings... model:{model_name}, path:{model_dir}')

        if (FASTTEXT_MODEL == 'crawl-300d-2M-subword.bin'):
            url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip'
            archive = 'crawl-300d-2M-subword.zip'
        else:
            print("EROR: unsupported fasttext model")
            return

        # Download the file
        dest_archive = model_dir + '/' + archive
        print(f"Downloading embeddings from {url} to {dest_archive}...")
        self._download_file(url, dest_archive)

        # Handle ZIP files
        try:
            with ZipFile(dest_archive, 'r') as zip_ref:
                print(f"Extracting ZIP file from {dest_archive}...")
                zip_ref.extractall(os.path.dirname(dest_archive))
            os.remove(dest_archive)
            print(f"Deleted ZIP file {dest_archive} after extraction.")
        except BadZipFile:
            print(f"Error: {dest_archive} is not a valid ZIP file or is corrupted.")
            os.remove(dest_archive)  # Optionally delete the corrupted file
            raise


    def build_embedding_vocab_matrix(self):

        print("building [subword based] embedding representation (matrix) of dataset vocabulary...")

        print("model:", type(self.model))
        print("model.vector_size:", self.model.vector_size)

        self.embedding_dim = self.model.vector_size
        self.vocab_size = len(self.vectorizer.vocabulary_)
        print("embedding_dim:", self.embedding_dim)
        print("vocab_size:", self.vocab_size)
        
        self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        # Dictionary to store the index-token mapping
        self.token_to_index_mapping = {}
        
        # Calculate the mean of all embeddings in the FastText model as a fallback for OOV tokens
        mean_vector = np.mean(self.model.wv.vectors, axis=0)

        oov_tokens = 0
        
        # Loop through the dataset vocabulary
        for word, idx in self.vectorizer.vocabulary_.items():
            self.token_to_index_mapping[idx] = word  # Store the token at its index

            # Check for the word in the original case first
            if word in self.model.wv.key_to_index:
                self.embedding_vocab_matrix[idx] = self.model.wv[word]
            # If not found, check the lowercase version of the word
            elif word.lower() in self.model.wv.key_to_index:
                self.embedding_vocab_matrix[idx] = self.model.wv[word.lower()]
            # If neither is found, handle it as an OOV token (FastText can generate subword embeddings)
            else:
                #print(f"Warning: OOV word when building embedding vocab matrix. Word: '{word}'")
                self.embedding_vocab_matrix[idx] = mean_vector  # Use the mean of all vectors as a substitute
                oov_tokens += 1
                
        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        print("token_to_index_mapping:", type(self.token_to_index_mapping), len(self.token_to_index_mapping))
        print("oov_tokens:", oov_tokens)

        return self.embedding_vocab_matrix, self.token_to_index_mapping


    def encode_docs(self, texts, embedding_vocab_matrix):
        """
        Compute both weighted document embeddings (using TF-IDF) and average document embeddings for each document.
        
        Args:
        - texts: List of input documents (as raw text).
        - embedding_vocab_matrix: Matrix of pre-trained word embeddings for Word2Vec/GloVe/FastText.

        Returns:
        - weighted_document_embeddings: Numpy array of weighted document embeddings for each document.
        - avg_document_embeddings: Numpy array of average document embeddings for each document.
        """
        
        print(f"encoding docs..")
        
        print("texts:", type(texts), len(texts))
        print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

        # Compute the mean embedding for FastText (for handling OOV tokens)
        self.mean_embedding = np.mean(self.model.wv.vectors, axis=0)
        print(f"Mean embedding vector for OOV tokens calculated: {self.mean_embedding.shape}")

        
        weighted_document_embeddings = []
        avg_document_embeddings = []

        oov_tokens = 0

        for doc in texts:
            # Tokenize the document using the vectorizer (ensures consistency in tokenization)
            tokens = self.vectorizer.build_analyzer()(doc)

            # Calculate TF-IDF weights for the tokens
            tfidf_vector = self.vectorizer.transform([doc]).toarray()[0]

            weighted_sum = np.zeros(embedding_vocab_matrix.shape[1])
            total_weight = 0.0
            valid_embeddings = []

            for token in tokens:
                # Get the embedding from FastText
                try:
                    embedding = self.model.wv.get_vector(token)  # FastText handles subword embeddings here
                except KeyError:
                    # If the token is OOV, use the mean embedding instead of a zero vector
                    embedding = self.mean_embedding
                    oov_tokens += 1

                # Get the TF-IDF weight if available, else assign a default weight of 1 (or 0 if not in vocabulary)
                weight = tfidf_vector[self.vectorizer.vocabulary_.get(token, 0)]

                # Accumulate the weighted sum for the weighted embedding
                weighted_sum += weight * embedding
                total_weight += weight

                # Collect valid embeddings for average embedding calculation
                valid_embeddings.append(embedding)

            # Compute the weighted embedding for the document
            if total_weight > 0:
                weighted_doc_embedding = weighted_sum / total_weight
            else:
                weighted_doc_embedding = np.zeros(embedding_vocab_matrix.shape[1])

            # Compute the average embedding for the document
            if valid_embeddings:
                avg_doc_embedding = np.mean(valid_embeddings, axis=0)
            else:
                avg_doc_embedding = np.zeros(embedding_vocab_matrix.shape[1])  # Handle empty or OOV cases

            # Append the document embeddings to the list
            weighted_document_embeddings.append(weighted_doc_embedding)
            avg_document_embeddings.append(avg_doc_embedding)

        print("weighted_document_embeddings:", type(weighted_document_embeddings), len(weighted_document_embeddings))
        print("avg_document_embeddings:", type(avg_document_embeddings), len(avg_document_embeddings))

        print("oov_tokens:", oov_tokens)

        return np.array(weighted_document_embeddings), np.array(avg_document_embeddings)




class TransformerLCRepresentationModel(LCRepresentationModel):
    """
    Base class for Transformer based architcteure langugae models such as BERT, RoBERTa, XLNet and GPT2
    """
    
    def __init__(self, model_name, model_dir, vtype='tfidf'):

        super().__init__(model_name, model_dir)
        
    
    
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
        special_tokens = self.tokenizer.all_special_tokens  # Dynamically fetch special tokens like [CLS], [SEP], <s>, </s>, etc.
        
        # Optionally, remove special tokens
        tokens = [token for token in tokens if token not in special_tokens]

        return tokens
    

    def _tokenize(self, texts):
        """
        Tokenize a batch of texts using the tokenizer, returning token IDs and attention masks.
        """
        
        tokenized_inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,  # Use the tokenizer's maximum length (typically 512 for BERT)
                return_attention_mask=True
        )

        return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask']
         
    
    def build_embedding_vocab_matrix(self):
        """
        Build the embedding vocabulary matrix for BERT and RoBERTa models.
        """
        print("building [BERT/RoBERTa based] embedding representation (matrix) of dataset vocabulary...")

        self.model.eval()  # Set model to evaluation mode            
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")  # To confirm which device is being used
        print("model:", type(self.model))
        
        #
        # NB we use different embedding vocab matrices here depending upon the pretrained model
        #
        self.embedding_dim = self.model.config.hidden_size  # Get the embedding dimension size
        self.vocab_size = len(self.vectorizer.vocabulary_)

        print("embedding_dim:", self.embedding_dim)
        print("dataset vocab size:", self.vocab_size)   

        self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        # Dictionary to store the index-token mapping
        self.token_to_index_mapping = {}

        # Initialize OOV token tracking
        oov_tokens = 0
        oov_list = []  # Keep a list of OOV words
        
        # Mean vector for OOV tokens
        mean_vector = np.zeros(self.embedding_dim)
        
        # -------------------------------------------------------------------------------------------------------------
        def _bert_embedding_vocab_matrix(vocab, batch_size=DEFAULT_CPU_BATCH_SIZE, max_len=512):
            """
            Construct the embedding vocabulary matrix for BERT and RoBERTa models.
            """

            print("_bert_embedding_vocab_matrix...")
            print("batch_size:", batch_size)
            print("max_len:", max_len)

            embedding_vocab_matrix = np.zeros((len(vocab), self.model.config.hidden_size))
            batch_words = []
            word_indices = []

            def process_batch(batch_words, max_len):
                inputs = self.tokenizer(batch_words, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)  # Ensure all inputs are on the same device

                # Perform inference, ensuring that the model is on the same device as the inputs
                self.model.to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Return the embeddings and ensure they are on the CPU for further processing
                return outputs.last_hidden_state[:, 0, :].cpu().numpy()

            with tqdm(total=len(vocab), desc="processing BERT/RoBERTa embedding vocab matrix construction batches") as pbar:
                for word, idx in vocab.items():
                    batch_words.append(word)
                    word_indices.append(idx)

                    if len(batch_words) == batch_size:
                        embeddings = process_batch(batch_words, max_len)
                        for i, embedding in zip(word_indices, embeddings):
                            if i < len(embedding_vocab_matrix):
                                embedding_vocab_matrix[i] = embedding
                            else:
                                print(f"IndexError: Skipping index {i} as it's out of bounds for embedding_vocab_matrix.")
                                oov_tokens += 1
                                oov_list.append(word)  # Track OOV words
                                embedding_vocab_matrix[i] = mean_vector  # Assign mean vector for OOV
                        batch_words = []
                        word_indices = []
                        pbar.update(batch_size)

                if batch_words:
                    embeddings = process_batch(batch_words, max_len)
                    for i, embedding in zip(word_indices, embeddings):
                        if i < len(embedding_vocab_matrix):
                            embedding_vocab_matrix[i] = embedding
                        else:
                            print(f"IndexError: Skipping index {i} as it's out of bounds for the embedding_vocab_matrix.")
                            oov_tokens += 1
                            oov_list.append(batch_words[i])  # Track OOV words
                            embedding_vocab_matrix[i] = mean_vector  # Assign mean vector for OOV
                    pbar.update(len(batch_words))

            return embedding_vocab_matrix
        # -------------------------------------------------------------------------------------------------------------
        
        #tokenize and prepare inputs
        max_length = self.tokenizer.model_max_length
        #print("max_length:", max_length)

        self.embedding_vocab_matrix = _bert_embedding_vocab_matrix(self.vectorizer.vocabulary_, batch_size=BATCH_SIZE, max_len=max_length)
        
        # Add to token-to-index mapping for BERT and RoBERTa
        for word, idx in self.vectorizer.vocabulary_.items():
            self.token_to_index_mapping[idx] = word  # Store the token at its index
            
        print("embedding_vocab_matrix:", type(self.embedding_vocab_matrix), self.embedding_vocab_matrix.shape)
        print("token_to_index_mapping:", type(self.token_to_index_mapping), len(self.token_to_index_mapping))

        # Final OOV tracking output
        print(f"Total OOV tokens: {oov_tokens}")
        """
        if oov_tokens > 0:
            print("List of OOV tokens:", oov_list)
        """    
        
        return self.embedding_vocab_matrix, self.token_to_index_mapping


    def encode_docs(self, text_list, embedding_vocab_matrix=None):
        """
        Generates both the mean and first token embeddings for a list of text sentences using RoBERTa.
        
        RoBERTa does not use a [CLS] token, but the first token often serves a similar purpose.
        This function computes:
        - The mean of all token embeddings.
        - The first token embedding (position 0).

        Parameters:
        ----------
        text_list : list of str
            List of docs to encode.

        Returns:
        -------
        mean_embeddings : np.ndarray
            Array of mean sentence embeddings (mean of all tokens).
        first_token_embeddings : np.ndarray
            Array of sentence embeddings using the first token.
        """

        print("encoding docs using BERT/RoBERTa...")

        self.model.eval()
        mean_embeddings = []
        first_token_embeddings = []

        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]  # Token-level embeddings from RoBERTa

                # Compute the mean of all token embeddings
                batch_mean_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                mean_embeddings.append(batch_mean_embeddings)

                # Compute the first token embedding (similar to [CLS] in BERT)
                batch_first_token_embeddings = token_vectors[:, 0, :].cpu().detach().numpy()
                first_token_embeddings.append(batch_first_token_embeddings)

        # Concatenate all batch results
        mean_embeddings = np.concatenate(mean_embeddings, axis=0)
        first_token_embeddings = np.concatenate(first_token_embeddings, axis=0)


        return mean_embeddings, first_token_embeddings



class BERTLCRepresentationModel(TransformerLCRepresentationModel):
    """
    BERTRepresentation subclass implementing sentence encoding using BERT
    """
    
    def __init__(self, model_name=BERT_MODEL, model_dir=VECTOR_CACHE+'/BERT', vtype='tfidf'):

        print("initializing BERT representation model...")

        super().__init__(model_name, model_dir)                             # parent constructor

        # instantiate model and tokenizer
        self.model = BertModel.from_pretrained(model_name, cache_dir=model_dir)
        
        self.tokenizer = BertTokenizerFast.from_pretrained(
            model_name, 
            cache_dir=model_dir,
            do_lower_case=False                 # keep tokenizer case sensitive
        )

        self.max_length = self.tokenizer.model_max_length
        print("self.max_length:", self.max_length)

        # NB BertTokenizerFast has a pad_token
            
        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                #min_df=MIN_DF_COUNT, 
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                #min_df=MIN_DF_COUNT, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")
        
        self.model.to(self.device)      # put the model on the appropriate device
    


class RoBERTaLCRepresentationModel(TransformerLCRepresentationModel):
    """
    RoBERTaRepresentationModel subclass implementing sentence encoding using RoBERTa
    """

    def __init__(self, model_name=ROBERTA_MODEL, model_dir=VECTOR_CACHE+'/RoBERTa', vtype='tfidf'):

        print("initializing RoBERTa representation model...")

        super().__init__(model_name, model_dir)                             # parent constructor

        self.model = RobertaModel.from_pretrained(model_name, cache_dir=model_dir)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name, cache_dir=model_dir)
    
        self.max_length = self.tokenizer.model_max_length
        print("self.max_length:", self.max_length)

        # NB RoBERTaTokenizerFast has a pad_token

        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                #min_df=MIN_DF_COUNT, 
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                #min_df=MIN_DF_COUNT, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")
        
        self.model.to(self.device)      # put the model on the appropriate device



class XLNetLCRepresentationModel(TransformerLCRepresentationModel):
    """
    XLNet representation model implementing sentence encoding using XLNet.
    """

    def __init__(self, model_name='xlnet-base-cased', model_dir=VECTOR_CACHE+'/XLNet', vtype='tfidf'):
        print("Initializing XLNet representation model...")

        super().__init__(model_name, model_dir)  # parent constructor

        # Load the XLNet model and tokenizer
        self.model = XLNetModel.from_pretrained(model_name, cache_dir=model_dir)
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name, cache_dir=model_dir)

        # Ensure padding token is available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id                   # Reuse the end-of-sequence token for padding

        """
        self.max_length = self.tokenizer.model_max_length
        print("self.max_length:", self.max_length)
        """

        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")

        self.model.to(self.device)  # Put the model on the appropriate device

    
    def _custom_tokenizer(self, text):
        """
        Tokenize the text using the tokenizer, returning tokenized strings (not token IDs) for TF-IDF or CountVectorizer.
        This tokenizer works for BERT, RoBERTa, and LLaMA models.
        
        Parameters:
        - text: The input text to be tokenized.
        
        Returns:
        - tokens: A list of tokens with special tokens removed based on the model in use.
        """

        #tokens = self.tokenizer.tokenize(text, max_length=self.max_length, truncation=True)
        tokens = self.tokenizer.tokenize(text)
        
        # Retrieve special tokens from the tokenizer object
        special_tokens = self.tokenizer.all_special_tokens  # Dynamically fetch special tokens like [CLS], [SEP], <s>, </s>, etc.
        
        # Optionally, remove special tokens
        tokens = [token for token in tokens if token not in special_tokens]

        return tokens
    

    def _tokenize(self, texts):
        """
        Tokenize a batch of texts using the tokenizer, returning token IDs and attention masks.
        """
        
        tokenized_inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                #truncation=True,
                #max_length=self.tokenizer.model_max_length,  # Use the tokenizer's maximum length (typically 512 for BERT)
                return_attention_mask=True
        )

        return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask']
    


    def build_embedding_vocab_matrix(self):
        """
        Builds the embedding vocabulary matrix using XLNet embeddings and tracks OOV tokens.
        """
        print("Building XLNet embedding vocab matrix...")

        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        self.vocab_size = len(self.vectorizer.vocabulary_)

        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Vocab size: {self.vocab_size}")

        self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        mean_vector = np.zeros(self.embedding_dim)

        oov_tokens = 0
        oov_list = []
        batch_words = []
        word_indices = []

        def process_batch(batch_words):
            #inputs = self.tokenizer(batch_words, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
            inputs = self.tokenizer(batch_words, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(self.device)

            # Get token embeddings from XLNet
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Tokenize the vocabulary and build embedding matrix
        with tqdm(total=len(self.vectorizer.vocabulary_), desc="Processing XLNet embedding vocab matrix") as pbar:
            for word, idx in self.vectorizer.vocabulary_.items():
                if word in self.tokenizer.get_vocab():
                    batch_words.append(word)
                    word_indices.append(idx)
                else:
                    # Track OOV tokens
                    oov_tokens += 1
                    oov_list.append(word)
                    self.embedding_vocab_matrix[idx] = mean_vector

                if len(batch_words) == self.batch_size:
                    embeddings = process_batch(batch_words)
                    for i, embedding in zip(word_indices, embeddings):
                        self.embedding_vocab_matrix[i] = embedding
                    batch_words = []
                    word_indices = []
                    pbar.update(self.batch_size)

            if batch_words:
                embeddings = process_batch(batch_words)
                for i, embedding in zip(word_indices, embeddings):
                    self.embedding_vocab_matrix[i] = embedding
                pbar.update(len(batch_words))

        print(f"Embedding vocab matrix built with shape {self.embedding_vocab_matrix.shape}")
        print(f"OOV tokens: {oov_tokens}")
        print(f"List of OOV tokens: {oov_list}")
    
        return self.embedding_vocab_matrix, self.vectorizer.vocabulary_
    


    def encode_docs(self, text_list, embedding_vocab_matrix=None):
        """
        Generates both the mean and [CLS] token embeddings for a list of text sentences using XLNet.

        The [CLS] token is used as the sentence representation for classification tasks, similar to BERT.

        Parameters:
        ----------
        text_list : list of str
            List of documents to encode.

        Returns:
        -------
        mean_embeddings : np.ndarray
            Array of mean sentence embeddings (mean of all tokens).
        cls_embeddings : np.ndarray
            Array of sentence embeddings using the [CLS] token.
        """
        print("Encoding docs using XLNet and extracting [CLS] token embeddings...")

        self.model.eval()
        mean_embeddings = []
        cls_embeddings = []

        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]  # Token-level embeddings from XLNet

                # XLNet's [CLS] token is at the end of the sequence, not the beginning like BERT
                batch_cls_embeddings = token_vectors[:, -1, :].cpu().detach().numpy()
                cls_embeddings.append(batch_cls_embeddings)

                # Compute the mean of all token embeddings
                batch_mean_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                mean_embeddings.append(batch_mean_embeddings)

        # Concatenate all batch results
        mean_embeddings = np.concatenate(mean_embeddings, axis=0)
        cls_embeddings = np.concatenate(cls_embeddings, axis=0)

        return mean_embeddings, cls_embeddings
    





class GPT2LCRepresentationModel(TransformerLCRepresentationModel):
    """
    GPT-2 representation model implementing sentence encoding using GPT-2.
    """

    def __init__(self, model_name='gpt2', model_dir=VECTOR_CACHE+'/GPT2', vtype='tfidf'):
        print("Initializing GPT-2 representation model...")

        super().__init__(model_name, model_dir)  # parent constructor

        # Load the GPT-2 model and tokenizer
        self.model = GPT2Model.from_pretrained(model_name, cache_dir=model_dir)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name, cache_dir=model_dir)

        # Ensure padding token is available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id                   # Reuse the end-of-sequence token for padding

        self.max_length = self.tokenizer.model_max_length
        print("self.max_length:", self.max_length)

        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                sublinear_tf=True, 
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(
                lowercase=False, 
                tokenizer=self._custom_tokenizer
            )
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")

        self.model.to(self.device)  # Put the model on the appropriate device
        

    def _custom_tokenizer(self, text):
        """
        Tokenize the text using the GPT-2 tokenizer, ensuring truncation and padding are applied.
        This method returns tokenized strings (subwords) for use with TF-IDF/CountVectorizer.
        """

        # Tokenize the text with GPT-2, applying truncation to limit sequence length
        tokenized_output = self.tokenizer(
            text,
            max_length=self.max_length,  # Limit to model's max length (usually 1024)
            truncation=True,  # Ensure sequences longer than max_length are truncated
            padding='max_length',  # Optional: can pad to max length (if needed)
            return_tensors=None,  # Return token strings, not tensor IDs
            add_special_tokens=False  # Do not add special tokens like <|endoftext|> for TF-IDF use
        )

        # The tokenizer will return a dictionary, so we extract the tokenized sequence
        tokens = tokenized_output['input_ids']

        # Convert token IDs back to token strings for use with TF-IDF/CountVectorizer
        tokens = self.tokenizer.convert_ids_to_tokens(tokens)

        return tokens



    def build_embedding_vocab_matrix(self):
        """
        Builds the embedding vocabulary matrix using GPT-2 embeddings and tracks OOV tokens.
        """
        print("Building GPT-2 embedding vocab matrix...")

        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        self.vocab_size = len(self.vectorizer.vocabulary_)

        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Vocab size: {self.vocab_size}")

        self.embedding_vocab_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        mean_vector = np.zeros(self.embedding_dim)

        oov_tokens = 0
        oov_list = []
        batch_words = []
        word_indices = []

        def process_batch(batch_words):
            inputs = self.tokenizer(batch_words, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
            input_ids = inputs['input_ids'].to(self.device)

            # Get token embeddings from GPT-2
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Tokenize the vocabulary and build embedding matrix
        with tqdm(total=len(self.vectorizer.vocabulary_), desc="Processing GPT-2 embedding vocab matrix") as pbar:
            for word, idx in self.vectorizer.vocabulary_.items():
                if word in self.tokenizer.get_vocab():
                    batch_words.append(word)
                    word_indices.append(idx)
                else:
                    # Track OOV tokens
                    oov_tokens += 1
                    oov_list.append(word)
                    self.embedding_vocab_matrix[idx] = mean_vector

                if len(batch_words) == self.batch_size:
                    embeddings = process_batch(batch_words)
                    for i, embedding in zip(word_indices, embeddings):
                        self.embedding_vocab_matrix[i] = embedding
                    batch_words = []
                    word_indices = []
                    pbar.update(self.batch_size)

            if batch_words:
                embeddings = process_batch(batch_words)
                for i, embedding in zip(word_indices, embeddings):
                    self.embedding_vocab_matrix[i] = embedding
                pbar.update(len(batch_words))

        print(f"Embedding vocab matrix built with shape {self.embedding_vocab_matrix.shape}")
        print(f"OOV tokens: {oov_tokens}")
        print(f"List of OOV tokens: {oov_list}")

        return self.embedding_vocab_matrix, self.vectorizer.vocabulary_


    def encode_docs(self, text_list, embedding_vocab_matrix=None):
        """
        Encode documents using GPT-2 embeddings.
        """

        print("Encoding docs using GPT-2...")
        self.model.eval()

        mean_embeddings = []

        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)  # Use token IDs for GPT-2
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                # Get the token-level embeddings from GPT-2
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]  # Shape: [batch_size, sequence_length, hidden_size]

                # Compute the mean of all token embeddings for each document
                batch_mean_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                mean_embeddings.append(batch_mean_embeddings)

        mean_embeddings = np.concatenate(mean_embeddings, axis=0)

        return mean_embeddings









