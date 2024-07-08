import numpy as np
import os
from tqdm import tqdm
import torch

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.language_representation.representation_model import batch_iterable, mean_across_all_tokens, \
    concat_all_tokens

from transformers import AutoModel, AutoConfig, AutoTokenizer



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
