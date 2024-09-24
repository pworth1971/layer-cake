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


from abc import ABC, abstractmethod

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from gensim.models.fasttext import load_facebook_model

from util.common import VECTOR_CACHE, DATASET_DIR


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
#BERT_MODEL = 'bert-base-cased'                              # dimension = 768, case sensitive
BERT_MODEL = 'bert-large-cased'                             # dimension = 1024, case sensitive

#ROBERTA_MODEL = 'roberta-base'                             # dimension = 768, case insensitive
ROBERTA_MODEL = 'roberta-large'

LLAMA_MODEL = 'meta-llama/Llama-2-7b-hf'                    # dimension = 4096, case sensitive
#LLAMA_MODEL = 'meta-llama/Llama-2-13b-hf'

FASTTEXT_MODEL = 'crawl-300d-2M-subword.bin'                # dimension 300, case sensitive

WORD2VEC_MODEL = 'GoogleNews-vectors-negative300.bin'       # dimension 300, case sensitive

#GLOVE_MODEL = 'glove.6B.300d.txt'                          # dimension 300, case insensensitve
GLOVE_MODEL = 'glove.42B.300d.txt'                          # dimensiomn 300, case sensitive
# -------------------------------------------------------------------------------------------------------

TOKEN_TOKENIZER_MAX_LENGTH = 512


MIN_DF_COUNT = 5
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

        self.model = None
        self.tokenizer = None

        self.initialized = True


    def _tokenize(self, text_list):
        """Tokenizes a list of text strings."""
        encoded = self.tokenizer.batch_encode_plus(
            text_list,
            add_special_tokens=True,                                            
            return_attention_mask=True,
            padding=True,                                                       # Pad to the longest sequence in the batch
            truncation=True,                                                    # Truncate to model max input length
            max_length=TOKEN_TOKENIZER_MAX_LENGTH,                              # specify max_length for truncation
            return_tensors='pt'                                                 # Return PyTorch tensors
        )
        
        return encoded['input_ids'], encoded['attention_mask']


    def _custom_tokenizer(self, text):
        """
        Tokenize the text using the tokenizer, returning tokenized strings (not token IDs) for TF-IDF or CountVectorizer.
        This tokenizer works for BERT, RoBERTa, and LLaMA models.
        
        Parameters:
        - text: The input text to be tokenized.
        
        Returns:
        - tokens: A list of tokens with special tokens removed based on the model in use.
        """
        # Tokenize the text into words/subwords
        tokens = self.tokenizer.tokenize(text, max_length=TOKEN_TOKENIZER_MAX_LENGTH, truncation=True)
        
        # Define special tokens based on the model in use
        special_tokens = []
        if self.pretrained == 'bert':
            # BERT special tokens
            special_tokens = ["[CLS]", "[SEP]"]
        elif self.pretrained == 'roberta':
            # RoBERTa special tokens
            special_tokens = ["<s>", "</s>"]  # RoBERTa uses <s> for CLS and </s> for SEP
        elif self.pretrained == 'llama':
            # LLaMA special tokens
            special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]  # Adjust based on the actual tokenizer used for LLaMA
        
        # Optionally, remove special tokens
        tokens = [token for token in tokens if token not in special_tokens]

        return tokens


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





class WordLCRepresentationModel(LCRepresentationModel):
    """
    WordBasedRepresentationModel handles word-based embeddings (e.g., Word2Vec, GloVe).
    It computes sentence embeddings by averaging, summing, or computing TF-IDF weighted embeddings.
    """

    def __init__(self, model_name, embedding_path, vtype='tfidf'):
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

        super().__init__(model_name, model_dir=embedding_path)  # parent constructor

        # Load the pre-trained word embedding model
        
        self.model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        
        self.vtype = vtype
        print(f"Vectorization type: {vtype}")

        # Get embedding size (dimensionality)
        self.embedding_dim = self.model.vector_size
        print(f"Embedding dimension: {self.embedding_dim}")

        if vtype == 'tfidf':
            print("using TF-IDF vectorization...")
            self.vectorizer = TfidfVectorizer(min_df=MIN_DF_COUNT, sublinear_tf=True)              # alignment with 2019 paper params
        elif vtype == 'count':
            print("using Count vectorization...")
            self.vectorizer = CountVectorizer(min_df=MIN_DF_COUNT)
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")



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
        
        print(f"encoding docs...")
        
        print("texts:", type(texts), len(texts))
        print("embedding_vocab_matrix:", type(embedding_vocab_matrix), embedding_vocab_matrix.shape)

        weighted_document_embeddings = []
        avg_document_embeddings = []

        for doc in texts:
            # Tokenize the document
            tokens = doc.split()

            # Calculate TF-IDF weights for the tokens
            tfidf_vector = self.vectorizer.transform([doc]).toarray()[0]

            weighted_sum = np.zeros(embedding_vocab_matrix.shape[1])
            total_weight = 0.0
            valid_embeddings = []

            unk_token_id = self.vectorizer.vocabulary_.get('<unk>', None)  # Check for the existence of an UNK token

            for token in tokens:
                #token_lower = token.lower()

                #token_id = self.vectorizer.vocabulary_.get(token.lower(), None)
                token_id = self.vectorizer.vocabulary_.get(token, None)

                if token_id is not None and 0 <= token_id < embedding_vocab_matrix.shape[0]:
                    # Get the embedding for the token
                    embedding = embedding_vocab_matrix[token_id]
                    
                    # Get the TF-IDF weight for the token
                    weight = tfidf_vector[token_id]

                    # Accumulate the weighted embedding
                    weighted_sum += embedding * weight
                    total_weight += weight    
                    
                    # Add to the valid embeddings list for averaging
                    valid_embeddings.append(embedding)
            
                elif unk_token_id is not None:
                    # Fallback to <unk> embedding if available
                    embedding = embedding_vocab_matrix[unk_token_id]
                
                else:
                    # OOV token handling: Use a zero vector if no embedding is found
                    print(f"Warning: OOV token: {token}")
                    embedding = np.zeros(embedding_vocab_matrix.shape[1])
                
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

        return np.array(weighted_document_embeddings), np.array(avg_document_embeddings)
    




class SubWordLCRepresentationModel(LCRepresentationModel):
    """
    SubWordBasedRepresentationModel handles subword-based embeddings (e.g. fastText).
    It computes sentence embeddings by averaging, summing, or computing TF-IDF weighted embeddings.
    """

    def __init__(self, model_name=FASTTEXT_MODEL, embedding_path=VECTOR_CACHE+'/fastText', vtype='tfidf'):
        """
        Initialize the word-based representation model (e.g., Word2Vec, GloVe).
        
        Parameters:
        ----------
        model_name : str
            Name of the pre-trained word embedding model (e.g. 'crawl-300d-2M-subword.bin').
        embedding_path : str
            Path to the pre-trained embedding file (e.g., '../.vector_cache/fastText').
        vtype : str, optional
            vectorization type, either 'tfidf' or 'count'.
        """

        print("Initializing SubWordBasedRepresentationModel...")

        super().__init__(model_name, model_dir=embedding_path)  # parent constructor

        # Load the pre-trained word embedding model
        # Append the FastText model name to the pretrained_path
        fasttext_model_path = embedding_path + '/' + model_name
        print("fasteext_model_path:", fasttext_model_path)

        # Use load_facebook_model to load FastText model from Facebook's pre-trained binary files
        self.model = load_facebook_model(fasttext_model_path)
            
        self.vtype = vtype
        print(f"Vectorization type: {vtype}")

        # Get embedding size (dimensionality)
        self.embedding_dim = self.model.vector_size
        print(f"Embedding dimension: {self.embedding_dim}")

        if vtype == 'tfidf':
            print("using TF-IDF vectorization...")
            self.vectorizer = TfidfVectorizer(min_df=MIN_DF_COUNT, sublinear_tf=True)              # alignment with 2019 paper params
        elif vtype == 'count':
            print("using Count vectorization...")
            self.vectorizer = CountVectorizer(min_df=MIN_DF_COUNT)
        else:
            raise ValueError("Invalid vectorizer type. Use 'tfidf' or 'count'.")



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

        weighted_document_embeddings = []
        avg_document_embeddings = []

        for doc in texts:
            # Tokenize the document
            tokens = doc.split()

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
                    # If the token or subwords cannot be found (rare with FastText), use a zero vector
                    print(f"Warning: OOV token: {token}")
                    embedding = np.zeros(embedding_vocab_matrix.shape[1])

            # Get the TF-IDF weight if available, else assign a default weight of 1 (or 0 if not in vocabulary)
            weight = tfidf_vector[self.vectorizer.vocabulary_.get(token, 0)]

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

        return np.array(weighted_document_embeddings), np.array(avg_document_embeddings)
    


class BERTLCRepresentationModel(LCRepresentationModel):
    """
    BERTRepresentation subclass implementing sentence encoding using BERT
    """
    
    def __init__(self, model_name=BERT_MODEL, model_dir=VECTOR_CACHE+'/BERT', vtype='tfidf'):

        print("initializing BERT representation model...")

        super().__init__(model_name, model_dir)                             # parent constructor

        # instantiate model and tokenizer
        self.model = BertModel.from_pretrained(model_name, cache_dir=model_dir)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, cache_dir=model_dir)

        # Ensure padding token is available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id                   # Reuse the end-of-sequence token for padding
            
        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(min_df=MIN_DF_COUNT, sublinear_tf=True, tokenizer=self._custom_tokenizer)
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(min_df=MIN_DF_COUNT, tokenizer=self._custom_tokenizer)
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")
        
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







class RoBERTaLCRepresentationModel(LCRepresentationModel):
    """
    RoBERTaRepresentationModel subclass implementing sentence encoding using RoBERTa
    """

    def __init__(self, model_name=ROBERTA_MODEL, model_dir=VECTOR_CACHE+'/RoBERTa', vtype='tfidf'):

        print("initializing RoBERTa representation model...")

        super().__init__(model_name, model_dir)                             # parent constructor

        self.model = RobertaModel.from_pretrained(model_name, cache_dir=model_dir)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name, cache_dir=model_dir)
    
        # Ensure padding token is available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id                   # Reuse the end-of-sequence token for padding
            
        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(min_df=MIN_DF_COUNT, sublinear_tf=True, tokenizer=self._custom_tokenizer)
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(min_df=MIN_DF_COUNT, tokenizer=self._custom_tokenizer)
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")
        
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

    def __init__(self, model_name=LLAMA_MODEL, model_dir=VECTOR_CACHE+'/LlaMa', vtype='tfidf'):
        
        print("initializing LLAMA representation model...")

        super().__init__(model_name, model_dir, vtype)                             # parent constructor
    
        self.model = LlamaModel.from_pretrained(model_name, cache_dir=model_dir)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(model_name, cache_dir=model_dir)

        # Ensure padding token is available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id                   # Reuse the end-of-sequence token for padding
            
        # Use the custom tokenizer for both TF-IDF and CountVectorizer
        if vtype == 'tfidf':
            self.vectorizer = TfidfVectorizer(min_df=MIN_DF_COUNT, sublinear_tf=True, tokenizer=self._custom_tokenizer)
        elif vtype == 'count':
            self.vectorizer = CountVectorizer(min_df=MIN_DF_COUNT, tokenizer=self._custom_tokenizer)
        else:
            raise ValueError("Invalid vectorizer type. Must be in [tfidf, count].")          

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

    












