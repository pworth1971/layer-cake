import numpy as np
import os
from tqdm import tqdm
import torch

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.language_representation.representation_model import batch_iterable, mean_across_all_tokens, \
    concat_all_tokens

from transformers import AutoModel, AutoConfig, AutoTokenizer

import torch
import numpy as np
from transformers import AutoModel, AutoConfig, AutoTokenizer
from tqdm import tqdm

from data.lc_dataset import DEFAULT_GPU_BATCH_SIZE, DEFAULT_CPU_BATCH_SIZE, MPS_BATCH_SIZE
from data.lc_dataset import BERT_MODEL, ROBERTA_MODEL, LLAMA_MODEL, VECTOR_CACHE

from transformers import BertTokenizerFast, LlamaTokenizerFast, RobertaTokenizerFast, LlamaTokenizer
from transformers import BertModel, LlamaModel, RobertaModel



#
# tokens for LLAMA model access, must be requested from huggingface
#
from huggingface_hub import login

HF_TOKEN = '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'
HF_TOKEN2 = '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'



"""
class CustomRepresentationModel(RepresentationModel)

This is a customized version of the RepresentationModel class from simpletransformers.
This version implements an encoding method that returns the embedding for the [CLS] token,
plus a few performance optimizations.

Default init() method added by PJW to support auto-loading of BERT model and model type
"""
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
            add_special_tokens=True,                # Add '[CLS]' and '[SEP]'
            return_attention_mask=True,
            padding=True,                           # Pad to the longest sequence in the batch
            truncation=True,                        # Truncate to model max input length
            return_tensors='pt'                     # Return PyTorch tensors
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
