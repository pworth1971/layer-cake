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


# -------------------------------------------------------------------------------------------------------
# default pretrained models.
#
# NB: these models are all case sensitive, ie no need to lowercase the input text (see _preprocess)
#

BERT_MODEL = 'bert-base-cased'                              # dimension = 768
ROBERTA_MODEL = 'roberta-base'                              # dimension = 768

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
        - The last token embedding.
        
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
