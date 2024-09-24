import numpy as np
import os
import torch
from tqdm import tqdm

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.language_representation.representation_model import batch_iterable, mean_across_all_tokens, \
    concat_all_tokens

from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import BertTokenizerFast, LlamaTokenizerFast, RobertaTokenizerFast
from transformers import BertModel, LlamaModel, RobertaModel

from util.common import VECTOR_CACHE
from abc import ABC, abstractmethod

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer



# -------------------------------------------------------------------------------------------------------
# default pretrained models.
#
# NB: these models are all case sensitive, ie no need to lowercase the input text (see _preprocess)
#

BERT_MODEL = 'bert-base-cased'                              # dimension = 768
#BERT_MODEL = 'bert-large-cased'

ROBERTA_MODEL = 'roberta-base'                              # dimension = 768
#ROBERTA_MODEL = 'roberta-large'

LLAMA_MODEL = 'meta-llama/Llama-2-7b-hf'                    # dimension = 4096
#LLAMA_MODEL = 'meta-llama/Llama-2-13b-hf'

FASTTEXT_MODEL = 'crawl-300d-2M-subword.bin'                # dimension 300, case sensitive

WORD2VEC_MODEL = 'GoogleNews-vectors-negative300.bin'       # dimension 300, case sensitive

#GLOVE_MODEL = 'glove.6B.300d.txt'                           # dimension 300, case insensensitve

GLOVE_MODEL = 'glove.42B.300d.txt'                          # dimensiomn 300, case sensitive
# -------------------------------------------------------------------------------------------------------

TOKEN_TOKENIZER_MAX_LENGTH = 512

# batch sizes for pytorch encoding routines
DEFAULT_CPU_BATCH_SIZE = 16
DEFAULT_GPU_BATCH_SIZE = 8
MPS_BATCH_SIZE = 16




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
    
    def __init__(self, model_name=None, model_dir='../.vector_cache', device=None):
        """
        Initialize the representation model.
        
        Args:
        - model_type (str): Type of the model ('bert', 'roberta', etc.).
        - model_name (str): Hugging Face Model Hub ID or local path.
        - model_dir (str, optional): Directory to save and load model files.
        - device (str, optional): Device to use for encoding ('cuda', 'mps', or 'cpu').
        """
        
        print("initializing LCRepresentationModel...")

        if device is None:

            # Setup device prioritizing CUDA, then MPS, then CPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.batch_size = DEFAULT_GPU_BATCH_SIZE
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.batch_size = MPS_BATCH_SIZE
            else:
                self.device = torch.device("cpu")
                self.batch_size = DEFAULT_CPU_BATCH_SIZE
        else:
            self.device = device
            if self.device == "cuda":
                self.batch_size = DEFAULT_GPU_BATCH_SIZE
            elif self.device == "mps":
                self.batch_size = MPS_BATCH_SIZE
            else:   
                self.batch_size = DEFAULT_CPU_BATCH_SIZE

        print("self.device:", self.device)
        print("self.batch_size:", self.batch_size)

        self.model_name = model_name
        print("self.model_name:", self.model_name)
        
        self.model_dir = model_dir
        print("self.model_dir:", model_dir)

        self.combine_strategy = 'mean'          # default combine strategy  

        self.model = None
        self.tokenizer = None


    def _tokenize(self, text_list):
        """Tokenizes a list of text strings."""
        encoded = self.tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,                                            # Add '[CLS]' and '[SEP]'
            return_attention_mask=True,
            padding=True,                                                       # Pad to the longest sequence in the batch
            truncation=True,                                                    # Truncate to model max input length
            max_length=TOKEN_TOKENIZER_MAX_LENGTH,                              # specify max_length for truncation
            return_tensors='pt'                                                 # Return PyTorch tensors
        )
        
        return encoded['input_ids'], encoded['attention_mask']

    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    @abstractmethod
    def encode_sentences(self, text_list):
        """
        Abstract method to be implemented by all subclasses.
        """
        pass





class WordBasedRepresentationModel(LCRepresentationModel):
    """
    WordBasedRepresentationModel handles word-based embeddings (e.g., Word2Vec, GloVe).
    It computes sentence embeddings by averaging, summing, or computing TF-IDF weighted embeddings.
    """

    def __init__(self, model_name, embedding_path, device=None):
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

        super().__init__(model_name, model_dir=embedding_path, device=device)  # parent constructor

        # Load the pre-trained word embedding model
        if model_name == 'word2vec':
            print(f"Loading Word2Vec embeddings from {embedding_path}...")
            self.model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        elif model_name == 'glove':
            print(f"Loading GloVe embeddings from {embedding_path}...")
            self.model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
        else:
            raise ValueError("Invalid model_name. Supported values are 'word2vec' and 'glove'.")

        # Get embedding size (dimensionality)
        self.embedding_dim = self.model.vector_size
        print(f"Embedding dimension: {self.embedding_dim}")

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()

    def _tokenize(self, text_list):
        """
        Tokenizes the input text into words.
        Word-based models (Word2Vec, GloVe) do not require subword tokenization like transformers.

        Parameters:
        ----------
        text_list : list of str
            List of sentences to tokenize.

        Returns:
        -------
        tokenized_texts : list of list of str
            List of tokenized sentences (each sentence is a list of words).
        """
        tokenized_texts = [text.split() for text in text_list]  # Simple word tokenization by splitting on spaces
        return tokenized_texts

    def _fit_tfidf(self, text_list):
        """
        Fits the TF-IDF vectorizer to the input text list to calculate term frequencies.
        
        Parameters:
        ----------
        text_list : list of str
            List of sentences to fit the TF-IDF model.

        Returns:
        -------
        None
        """
        print("Fitting TF-IDF vectorizer...")
        self.vectorizer.fit(text_list)  # Fit the vectorizer on the raw text data (not tokenized)

    def encode_sentences(self, text_list, comp_method='mean', use_tfidf=False):
        """
        Generates embeddings for a list of text sentences using Word2Vec or GloVe.
        Supports averaging, summing, or computing TF-IDF weighted embeddings.

        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.
        comp_method : str, optional
            Strategy for combining word embeddings. Defaults to 'mean' but supports 'sum'.
            - 'mean': Mean of all word embeddings.
            - 'sum': Sum of all word embeddings.
        use_tfidf : bool, optional
            If True, uses TF-IDF weighted embeddings.

        Returns:
        -------
        embeddings : np.ndarray
            Array of sentence embeddings.
        """
        print(f"Encoding sentences using Word2Vec or GloVe with {'TF-IDF' if use_tfidf else comp_method}...")

        # Fit the TF-IDF vectorizer on the text list if TF-IDF weighting is required
        if use_tfidf:
            self._fit_tfidf(text_list)

        # Tokenize the sentences into words
        tokenized_texts = self._tokenize(text_list)

        sentence_embeddings = []

        for i, tokens in enumerate(tokenized_texts):
            word_embeddings = []
            tfidf_weights = []
            for word in tokens:
                if word in self.model.key_to_index:  # Check if the word is in the vocabulary
                    word_embeddings.append(self.model[word])

                    if use_tfidf:
                        # Retrieve the TF-IDF weight for the word
                        tfidf = self.vectorizer.transform([text_list[i]]).toarray()[0]
                        tfidf_weight = tfidf[self.vectorizer.vocabulary_.get(word, 0)]
                        tfidf_weights.append(tfidf_weight)
                    else:
                        tfidf_weights.append(1.0)  # Use a default weight of 1 if not using TF-IDF
                else:
                    word_embeddings.append(np.zeros(self.embedding_dim))  # Handle OOV words with zero vector
                    tfidf_weights.append(0.0)  # No weight for OOV words

            if word_embeddings:
                word_embeddings = np.stack(word_embeddings)

                if comp_method == 'mean':
                    # Compute the mean of word embeddings
                    sentence_embedding = word_embeddings.mean(axis=0)
                elif comp_method == 'sum':
                    # Compute the sum of word embeddings
                    sentence_embedding = word_embeddings.sum(axis=0)
                elif use_tfidf:
                    # Compute TF-IDF weighted sum
                    tfidf_weights = np.array(tfidf_weights).reshape(-1, 1)  # Reshape to align with word_embeddings
                    sentence_embedding = np.sum(word_embeddings * tfidf_weights, axis=0)
                else:
                    raise ValueError("Invalid comp_method. Supported: 'mean', 'sum', or use TF-IDF.")
            else:
                sentence_embedding = np.zeros(self.embedding_dim)  # Empty sentence, return zero vector

            sentence_embeddings.append(sentence_embedding)

        return np.array(sentence_embeddings)

    def encode_sentences_opt(self, text_list):
        """
        Generates both the mean, sum, and TF-IDF weighted embeddings for a list of text sentences using Word2Vec or GloVe.
        
        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.

        Returns:
        -------
        mean_embeddings : np.ndarray
            Array of mean sentence embeddings (mean of all word embeddings).
        sum_embeddings : np.ndarray
            Array of sum sentence embeddings (sum of all word embeddings).
        tfidf_embeddings : np.ndarray
            Array of TF-IDF weighted sentence embeddings.
        """
        print("Encoding sentences using Word2Vec or GloVe (optimized)...")

        # Fit the TF-IDF vectorizer on the text list
        self._fit_tfidf(text_list)

        # Tokenize the sentences into words
        tokenized_texts = self._tokenize(text_list)

        mean_embeddings = []
        sum_embeddings = []
        tfidf_embeddings = []

        for i, tokens in enumerate(tokenized_texts):
            word_embeddings = []
            tfidf_weights = []
            for word in tokens:
                if word in self.model.key_to_index:
                    word_embeddings.append(self.model[word])

                    # Retrieve the TF-IDF weight for the word
                    tfidf = self.vectorizer.transform([text_list[i]]).toarray()[0]
                    tfidf_weight = tfidf[self.vectorizer.vocabulary_.get(word, 0)]
                    tfidf_weights.append(tfidf_weight)
                else:
                    word_embeddings.append(np.zeros(self.embedding_dim))  # Handle OOV words
                    tfidf_weights.append(0.0)

            if word_embeddings:
                word_embeddings = np.stack(word_embeddings)

                # Compute mean and sum embeddings
                mean_embeddings.append(word_embeddings.mean(axis=0))
                sum_embeddings.append(word_embeddings.sum(axis=0))

                # Compute TF-IDF weighted sum embeddings
                tfidf_weights = np.array(tfidf_weights).reshape(-1, 1)
                tfidf_embeddings.append(np.sum(word_embeddings * tfidf_weights, axis=0))
            else:
                mean_embeddings.append(np.zeros(self.embedding_dim))
                sum_embeddings.append(np.zeros(self.embedding_dim))
                tfidf_embeddings.append(np.zeros(self.embedding_dim))

        return np.array(mean_embeddings), np.array(sum_embeddings), np.array(tfidf_embeddings)


# BERTRepresentation subclass implementing sentence encoding using BERT
class BERTLCRepresentationModel(LCRepresentationModel):

    def __init__(self, model_name, model_dir, device):

        print("initializing BERT representation model...")

        super().__init__(model_name, model_dir, device)                             # parent constructor

        # instantiate model and tokenizer
        self.model = BertModel.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT').to(self.device)
        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT')

        self.model.to(self.device)      # put the model on the appropriate device


    def encode_sentences(self, text_list, comp_method='mean'):
        """
        Generates embeddings for a list of text sentences using BERT.

        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.
        comp_method : str, optional
            Strategy for combining word embeddings. Defaults to 'mean' but also supports 'summary' which uses the [CLS] token (index 0).
            - 'mean': Mean of all token embeddings.
            - 'summary': Use the [CLS] token embedding.

        Returns:
        -------
        embeddings : np.ndarray
            Array of sentence embeddings.
        """

        print("encoding sentences (BERT style)...")

        if (comp_method in ['avg', 'average', 'mean']):
            self.combine_strategy = 'mean'
        elif (comp_method in ['cls', 'summary', 'summ', 'cls_token', 'first']):
            self.combine_strategy = 'summary'
        else:
            self.combine_stratgey = 'mean'              # default to mean
        
        print("self.combine_strategy:", self.combine_strategy)

        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]

                if (self.combine_strategy == "mean"):
                    batch_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                elif (self.combine_strategy == "summary"):
                    # Select the [CLS] token embedding, at index 0
                    batch_embeddings = token_vectors[:, 0, :].cpu().detach().numpy()            # should be CLS token
                 
                embeddings.append(batch_embeddings)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings


    def encode_sentences_opt(self, text_list):
        """
        Generates both the mean and first token embeddings for a list of text sentences using RoBERTa.
        
        RoBERTa does not use a [CLS] token, but the first token often serves a similar purpose.
        This function computes:
        - The mean of all token embeddings.
        - The first token embedding (position 0).

        Generates embeddings for a list of text sentences using BERT.

        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.

        Returns:
        -------
        mean_embeddings : np.ndarray
            Array of mean sentence embeddings (mean of all tokens).
        first_token_embeddings : np.ndarray
            Array of sentence embeddings using the first token.
        """

        print("Encoding sentences for RoBERTa...")

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






# BERTRepresentation subclass implementing sentence encoding using BERT
class RoBERTaLCRepresentationModel(LCRepresentationModel):

    def __init__(self, model_name, model_dir, device):

        print("initializing RoBERTa representation model...")

        super().__init__(model_name, model_dir, device)                             # parent constructor

        self.model = RobertaModel.from_pretrained(ROBERTA_MODEL, cache_dir=VECTOR_CACHE+'/RoBERTa').to(self.device)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_MODEL, cache_dir=VECTOR_CACHE+'/RoBERTa')
    
        self.model.to(self.device)      # put the model on the appropriate device



    def encode_sentences(self, text_list, comp_method='mean'):
        """
        Generates embeddings for a list of text sentences using BERT.

        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.
        comp_method : str, optional
            Strategy for combining word embeddings. Defaults to 'mean' but also supports 'summary' which uses the [CLS] token (index 0).
            - 'mean': Mean of all token embeddings.
            - 'summary': Use the [CLS] token embedding.

        Returns:
        -------
        embeddings : np.ndarray
            Array of sentence embeddings.
        """

        print("encoding sentences (BERT style)...")

        if (comp_method in ['avg', 'average', 'mean']):
            self.combine_strategy = 'mean'
        elif (comp_method in ['summary', 'summ', 'first', 'aggregate']):
            self.combine_strategy = 'summary'
        else:
            self.combine_stratgey = 'mean'              # default to mean
        
        print("self.combine_strategy:", self.combine_strategy)

        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]

                if (self.combine_strategy == "mean"):
                    batch_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                elif (self.combine_strategy == "summary"):
                    # Select the first token, not a [CLS] token but still 
                    # contains aggregate information about sentence 
                    batch_embeddings = token_vectors[:, 0, :].cpu().detach().numpy()     
                 
                embeddings.append(batch_embeddings)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings
    


    def encode_sentences_opt(self, text_list):
        """
        Generates both the mean and first token embeddings for a list of text sentences using RoBERTa.
        
        RoBERTa does not use a [CLS] token, but the first token often serves a similar purpose.
        This function computes:
        - The mean of all token embeddings.
        - The first token embedding (position 0).

        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.

        Returns:
        -------
        mean_embeddings : np.ndarray
            Array of mean sentence embeddings (mean of all tokens).
        first_token_embeddings : np.ndarray
            Array of sentence embeddings using the first token.
        """
        print("Encoding sentences for RoBERTa...")

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

    

# LLAMARepresentation subclass implementing sentence encoding using LLAMA
class LlaMaLCRepresentationModel(LCRepresentationModel):

    def __init__(self, model_name, model_dir, device):
        
        print("initializing LLAMA representation model...")

        super().__init__(model_name, model_dir, device)                             # parent constructor
    
        self.model = LlamaModel.from_pretrained(LLAMA_MODEL, cache_dir=VECTOR_CACHE+'/LLaMa').to(self.device)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(LLAMA_MODEL, cache_dir=VECTOR_CACHE+'/LLaMa')

        self.model.to(self.device)      # put the model on the appropriate device



    def encode_sentences(self, text_list, comp_method='mean'):
        """
        Generates embeddings for a list of text sentences using LLaMa.
        
        LLaMa does not use [CLS] tokens. Instead, it can return:
        - The mean of all token embeddings.
        - The last token embedding which is a (representation of the) summary of the sentence.
        
        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.
        comp_method : str, optional
            Strategy for combining word embeddings. Defaults to 'mean' but supports 'last' (use the last token embedding).
            - 'mean': Mean of all token embeddings.
            - 'last': Use the last token embedding.

        Returns:
        -------
        embeddings : np.ndarray
            Array of sentence embeddings.
        """
        print("Encoding sentences for LLaMa...")
        
        if comp_method in ['avg', 'average', 'mean']:
            self.combine_strategy = 'mean'
        elif comp_method in ['summary', 'last', 'summ']:
            self.combine_strategy = 'last'
        else:
            raise ValueError("Invalid combine_strategy. Supported: 'mean' or 'last'.")

        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]  # Token-level embeddings from LLaMa

                if self.combine_strategy == 'last':
                    # Use the last token embedding for each sequence
                    batch_embeddings = token_vectors[:, -1, :].cpu().detach().numpy()
                elif self.combine_strategy == 'mean':
                    # Mean of all token embeddings
                    batch_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                else:
                    raise ValueError("Invalid combine_strategy. Supported: 'mean' or 'last'.")

                embeddings.append(batch_embeddings)

        embeddings = np.concatenate(embeddings, axis=0)
        
        return embeddings
    

    def encode_sentences_opt(self, text_list):
        """
        Generates both the mean and last token embeddings for a list of text sentences using LLaMa.
        
        LLaMa does not use [CLS] tokens. Instead, it can return:
        - The mean of all token embeddings.
        - The last token embedding, which is a (representation of the) summary of the sentence.
        
        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.

        Returns:
        -------
        mean_embeddings : np.ndarray
            Array of mean sentence embeddings (mean of all tokens).
        last_embeddings : np.ndarray
            Array of sentence embeddings using the last token.
        """
        print("Encoding sentences for LLaMa (optimized)...")

        self.model.eval()
        mean_embeddings = []
        last_embeddings = []

        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]  # Token-level embeddings from LLaMa

                # Compute the mean of all token embeddings
                batch_mean_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                mean_embeddings.append(batch_mean_embeddings)

                # Compute the last token embedding
                batch_last_embeddings = token_vectors[:, -1, :].cpu().detach().numpy()
                last_embeddings.append(batch_last_embeddings)

        # Concatenate all batch results
        mean_embeddings = np.concatenate(mean_embeddings, axis=0)
        last_embeddings = np.concatenate(last_embeddings, axis=0)

        return mean_embeddings, last_embeddings

    





























class CustomRepresentationModel(RepresentationModel):
   
    def __init__(self, model_type='bert', model_name='bert-base-uncased', model_dir='../.vector_cache'):
        """
        Initialize the representation model.
        
        Args:
        model_type (str): Type of the model ('bert', 'roberta', etc.).
        model_name (str): Hugging Face Model Hub ID or local path.
        model_dir (str, optional): Directory to save and load model files.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if model_dir is provided and contains the config file
        config_path = os.path.join(model_dir, 'config.json')

        if model_dir and os.path.exists(config_path):
            self.config = AutoConfig.from_pretrained(model_dir)
            self.model = AutoModel.from_pretrained(model_dir, config=self.config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            # If not, fallback to downloading the model
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model.to(self.device)


    def _tokenize(self, text_list):
        """Tokenizes a list of text strings."""
        encoded = self.tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,                                # Add '[CLS]' and '[SEP]'
            return_attention_mask=True,
            padding=True,                                           # Pad to the longest sequence in the batch
            truncation=True,                                        # Truncate to model max input length
            max_length=TOKEN_TOKENIZER_MAX_LENGTH,                  # specify max_length for truncation
            return_tensors='pt'                                     # Return PyTorch tensors
        )
        
        return encoded['input_ids'], encoded['attention_mask']



    def encode_sentences(self, text_list, combine_strategy=None, batch_size=32):
        """
        Generates list of contextual word or sentence embeddings using the model passed to class constructor
        :param text_list: list of text sentences
        :param combine_strategy: strategy for combining word vectors, supported values: None, "mean", "concat", or int value to select a specific embedding (e.g. 0 for [CLS] or -1 for the last one)
        :param batch_size
        :return: list of lists of sentence embeddings(if `combine_strategy=None`) OR list of sentence embeddings(if `combine_strategy!=None`)
        """
        self.model.to(self.device)

        batches = batch_iterable(text_list, batch_size=batch_size)
        embeddings = list()
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(batches, total=np.ceil(len(text_list) / batch_size)):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Assume outputs[0] contains the token-level embeddings
                token_vectors = outputs[0]  # The output tensor containing hidden states

                if combine_strategy is not None:
                    if isinstance(combine_strategy, int):
                        # Select embeddings for a specific token position across all sequences
                        batch_embeddings = token_vectors[:, combine_strategy, :].cpu().detach().numpy()
                    else:
                        embedding_func_mapping = {"mean": mean_across_all_tokens, "concat": concat_all_tokens}
                        try:
                            embedding_func = embedding_func_mapping[combine_strategy]
                            batch_embeddings = embedding_func(token_vectors).cpu().detach().numpy()
                        except KeyError:
                            raise ValueError("Provided combine_strategy is not valid. Supported values are: 'concat', 'mean' and None.")
                    embeddings.append(batch_embeddings)
                else:
                    # Append all token embeddings without any combination
                    embeddings.append(token_vectors.cpu().detach().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings





class BERTRepresentationModel:
    """
    BERTRepresentationModel handles the generation of embeddings for BERT-based models.
    It includes support for sentence and token-level embeddings using BERT's [CLS] token.
    """

    def __init__(self, model_name=BERT_MODEL, model_dir=VECTOR_CACHE+'/BERT'):
        """
        Initialize BERT model, tokenizer, and config.

        Parameters:
        ----------
        model_name : str
            Hugging Face model name or path to pre-trained BERT model.
        model_dir : str, optional
            Path to load the BERT model from if available locally.
        """

        print("initializing BERT representation model...")

        # Setup device prioritizing CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.batch_size = DEFAULT_GPU_BATCH_SIZE
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.batch_size = MPS_BATCH_SIZE
        else:
            self.device = torch.device("cpu")
            self.batch_size = DEFAULT_CPU_BATCH_SIZE

        print("device:", self.device)
        print("batch_size:", self.batch_size)

        self.model_name = model_name
        print("model_name:", self.model_name)
        
        print("model_dir:", model_dir)

        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT')
        self.model = BertModel.from_pretrained(BERT_MODEL, cache_dir=VECTOR_CACHE+'/BERT').to(self.device)


    def _tokenize(self, text_list):
        """Tokenizes a list of text strings."""
        encoded = self.tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            truncation=True,
            max_length=TOKEN_TOKENIZER_MAX_LENGTH,       # specify max_length for truncation
            return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']


    def encode_sentences(self, text_list, combine_strategy='mean'):
        """
        Generates embeddings for a list of text sentences using BERT.

        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.
        combine_strategy : int, str, optional
            Strategy for combining word embeddings. Defaults to 'summary' which uses the [CLS] token (index 0).
            - 'mean': Mean of all token embeddings.
            - 'summary': Use the [CLS] token embedding.

        Returns:
        -------
        embeddings : np.ndarray
            Array of sentence embeddings.
        """

        print("encoding sentences (BERT style)...")
        print("combine_strategy:", combine_strategy)

        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]

                if (combine_strategy == 'summary'):
                    # Select the [CLS] token embedding, at index 0
                    batch_embeddings = token_vectors[:, 0, :].cpu().detach().numpy()            # should be CLS token
                elif combine_strategy == "mean":
                    batch_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                else:
                    raise ValueError("Invalid combine_strategy. Supported: 'summary' or 'mean'.")

                embeddings.append(batch_embeddings)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings


class RoBERTaRepresentationModel:
    """
    RoBERTaRepresentationModel handles the generation of embeddings for RoBERTa-based models.
    RoBERTa doesn't use [CLS] or [SEP] tokens, so the first token (<s>) is used for sentence embeddings.
    """

    def __init__(self, model_name=ROBERTA_MODEL, model_dir=VECTOR_CACHE+'/RoBERTa'):
        """
        Initialize RoBERTa model, tokenizer, and config.

        Parameters:
        ----------
        model_name : str
            Hugging Face model name or path to pre-trained RoBERTa model.
        model_dir : str, optional
            Path to load the RoBERTa model from if available locally.
        """
        # Setup device prioritizing CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.batch_size = DEFAULT_GPU_BATCH_SIZE
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.batch_size = MPS_BATCH_SIZE
        else:
            self.device = torch.device("cpu")
            self.batch_size = DEFAULT_CPU_BATCH_SIZE

        print("device:", self.device)
        print("batch_size:", self.batch_size)
        
        self.model_name = model_name

        self.model = RobertaModel.from_pretrained(ROBERTA_MODEL, cache_dir=VECTOR_CACHE+'/RoBERTa').to(self.device)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_MODEL, cache_dir=VECTOR_CACHE+'/RoBERTa')
        
        self.model.to(self.device)

    def _tokenize(self, text_list):
        """Tokenizes a list of text strings."""
        encoded = self.tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            truncation=True,
            max_length=TOKEN_TOKENIZER_MAX_LENGTH,       # specify max_length for truncation
            return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']

    def encode_sentences(self, text_list, combine_strategy='mean'):
        """
        Generates embeddings for a list of text sentences using RoBERTa.

        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.

        Returns:
        -------
        embeddings : np.ndarray
            Array of sentence embeddings.
        """

        print("encoding sentences (RoBERTA style)...")

        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm([text_list[i:i + self.batch_size] for i in range(0, len(text_list), self.batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]

                batch_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()     # mean of all token embeddings

                embeddings.append(batch_embeddings)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings


class LlaMaRepresentationModel:
    """
    LlaMaRepresentationModel handles the generation of embeddings for LlaMa-based models.
    LlaMa doesn't use specific [CLS] or [SEP] tokens for sentence embeddings.
    """

    def __init__(self, model_name=LLAMA_MODEL, model_dir=VECTOR_CACHE+'/LlaMa'):
        """
        Initialize LLaMa model, tokenizer, and config.

        Parameters:
        ----------
        model_name : str
            Hugging Face model name or path to pre-trained LLaMa model.
        model_dir : str, optional
            Path to load the LLaMa model from if available locally.
        """

        # Setup device prioritizing CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.batch_size = DEFAULT_GPU_BATCH_SIZE
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.batch_size = MPS_BATCH_SIZE
        else:
            self.device = torch.device("cpu")
            self.batch_size = DEFAULT_CPU_BATCH_SIZE

        print("device:", self.device)
        print("batch_size:", self.batch_size)

        self.model_name = model_name

        # must have a valid token to access the LLaMa model
        # by using 'huggingface-cli -login' at terminal and 
        # logging in with autnentciation token
        self.model = LlamaModel.from_pretrained(LLAMA_MODEL, cache_dir=VECTOR_CACHE+'/LLaMa').to(self.device)

        self.tokenizer = LlamaTokenizerFast.from_pretrained(LLAMA_MODEL, cache_dir=VECTOR_CACHE+'/LLaMa')
                
        self.model.to(self.device)

    def _tokenize(self, text_list):
        """Tokenizes a list of text strings."""
        encoded = self.tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,
            return_attention_mask=True,
            padding=True,
            truncation=True,
            max_length=TOKEN_TOKENIZER_MAX_LENGTH,       # specify max_length for truncation
            return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']

    def encode_sentences(self, text_list, combine_strategy='mean'):
        """
        Generates embeddings for a list of text sentences using LlaMa.

        Parameters:
        ----------
        text_list : list of str
            List of sentences to encode.

        Returns:
        -------
        embeddings : np.ndarray
            Array of sentence embeddings.
        """
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm([text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]):
                input_ids, attention_mask = self._tokenize(batch)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_vectors = outputs[0]

                batch_embeddings = token_vectors.mean(dim=1).cpu().detach().numpy()
                
                embeddings.append(batch_embeddings)

        embeddings = np.concatenate(embeddings, axis=0)
        
        return embeddings
