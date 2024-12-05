import logging
logging.basicConfig(level=logging.INFO)

import torch
from transformers import BertModel, BertTokenizer
from transformers import BertTokenizerFast, BertModel

from model.layers import *

from embedding.pretrained import BERT_MODEL




class NeuralClassifier(nn.Module):

    ALLOWED_NETS = {'cnn', 'lstm', 'attn'}

    def __init__(self,
                 net_type,
                 output_size,
                 hidden_size,
                 vocab_size,
                 learnable_length,
                 pretrained = None,
                 drop_embedding_range=None,
                 drop_embedding_prop=0):
        
        super(NeuralClassifier, self).__init__()

        self.embed = EmbeddingCustom(vocab_size, learnable_length, pretrained, drop_embedding_range, drop_embedding_prop)
        self.projection = init__projection(net_type)(self.embed.dim(), hidden_size)
        self.label = nn.Linear(self.projection.dim(), output_size)

    def forward(self, input):
        word_emb = self.embed(input)
        doc_emb = self.projection(word_emb)
        logits = self.label(doc_emb)
        return logits

    def finetune_pretrained(self):
        self.embed.finetune_pretrained()

    def xavier_uniform(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    def get_embedding_size(self):
        return self.embed.get_pt_dimensions()

    def get_learnable_embedding_size(self):
        return self.embed.get_lrn_dimensions()
    
    

class Token2BertEmbeddings:

    def __init__(self, pretrained_model_name=BERT_MODEL, device=None):
        """
        Initialize Token2BertEmbeddings with a pretrained BERT model and tokenizer.

        Parameters:
        ----------
        pretrained_model_name : str
            The name of the pretrained BERT model (default: 'bert-base-uncased').
        device : str
            Device to run the model on ('cuda', 'mps', or 'cpu').
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
        self.model = BertModel.from_pretrained(pretrained_model_name).eval().to(device)

        self.max_length = self.model.config.max_position_embeddings  # Dynamically get max length
        self.hidden_size = self.model.config.hidden_size  # Dynamically get embedding dimension
        
        self.device = device

    def train(self, mode=True):
        """
        Sets the model to training or evaluation mode.

        Parameters:
        ----------
        mode : bool
            If True, sets to training mode. If False, sets to evaluation mode.
        """
        self.model.train(mode)

    def eval(self):
        """
        Sets the BERT model in evaluation mode for inference.
        """
        self.model.eval()

    def embeddings(self, tokens):
        """
        Generate contextualized embeddings for the given tokens.

        Parameters:
        ----------
        tokens : list of lists of str or tensor
            Tokenized input. Each sublist is a sequence of tokens.

        Returns:
        -------
        torch.Tensor
            Contextualized embeddings of shape (batch_size, seq_len, embedding_dim).
        """
        if isinstance(tokens, torch.Tensor):
            # Convert tensor to a list of lists of tokens using the tokenizer
            tokens = [
                [self.tokenizer.convert_ids_to_tokens(token_id.item()) for token_id in doc if token_id != self.tokenizer.pad_token_id]
                for doc in tokens
            ]

        max_length = min(self.max_length, max(map(len, tokens)))  # for dynamic padding
        cls_t = self.tokenizer.cls_token
        sep_t = self.tokenizer.sep_token
        pad_idx = self.tokenizer.pad_token_id
        tokens = [[cls_t] + d[:max_length] + [sep_t] for d in tokens]
        index = [
            self.tokenizer.convert_tokens_to_ids(doc) + [pad_idx] * (max_length - (len(doc)-2)) for doc in
            tokens
        ]
        index = torch.tensor(index).to(self.device)

        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = (index != pad_idx).long()

        with torch.no_grad():
            outputs = self.model(input_ids=index, attention_mask=attention_mask)
            contextualized_embeddings = outputs[0]
            contextualized_embeddings = contextualized_embeddings[:, 1:-1, :]  # Ignore [CLS] and last token
            return contextualized_embeddings
        

    def dim(self):
        """
        Get the dimensionality of the embeddings.

        Returns:
        -------
        int
            The embedding dimensionality.
        """
        return self.hidden_size



class Token2WCEmbeddings(nn.Module):

    def __init__(self, WCE, WCE_range, WCE_vocab, drop_embedding_prop=0.5, max_length=500, device='cuda'):
        """
        Initialize Token2WCEmbeddings for Word-Class Embeddings.

        Parameters:
        ----------
        WCE : torch.Tensor
            Pretrained Word-Class Embedding matrix.
        WCE_range : list
            Range of supervised embeddings in the matrix.
        WCE_vocab : dict
            Vocabulary mapping words to indices.
        drop_embedding_prop : float
            Dropout probability for embedding dropout.
        max_length : int
            Maximum sequence length.
        device : str
            Device to use ('cuda', 'cpu', etc.).
        """
        super(Token2WCEmbeddings, self).__init__()
        assert '[PAD]' in WCE_vocab, 'unknown index for special token [PAD] in WCE vocabulary'
    
        self.embed = EmbeddingCustom(len(WCE_vocab), 0, WCE, WCE_range, drop_embedding_prop).to(device)
    
        self.max_length = max_length
        self.device = device
        self.vocab = WCE_vocab
        self.pad_idx = self.vocab['[PAD]']
        self.unk_idx = self.vocab['[UNK]']

        self.training_mode = False

    def forward(self, tokens):
        """
        Generate embeddings for a batch of token sequences.

        Parameters:
        ----------
        tokens : list of lists of str
            Tokenized input.

        Returns:
        -------
        torch.Tensor
            Word-Class Embeddings for the input tokens.
        """
        max_length = min(self.max_length, max(map(len,tokens)))  # for dynamic padding
        tokens = [d[:max_length] for d in tokens]
        index = [
            [self.vocab.get(ti, self.unk_idx) for ti in doc] + [self.pad_idx]*(max_length - len(doc)) for doc in tokens
        ]
        index = torch.tensor(index).to(self.device)
        return self.embed(index)

    def dim(self):
        """
        Get the dimensionality of the embeddings.

        Returns:
        -------
        int
            The embedding dimensionality.
        """
        return self.embed.dim()

    def train(self, mode=True):
        """
        Set the embedding layer to training mode.

        Parameters:
        ----------
        mode : bool
            If True, set to training mode. Otherwise, evaluation mode.
        """
        self.training_mode = mode
        super(Token2WCEmbeddings, self).train(mode)
        self.embed.train(mode)

    def eval(self):
        """
        Set the embedding layer to evaluation mode.
        """
        self.train(False)

    def finetune_pretrained(self):
        """
        Enable fine-tuning for the pretrained embeddings.
        """
        self.embed.finetune_pretrained()



class BertWCEClassifier(nn.Module):
    
    ALLOWED_NETS = {'cnn', 'lstm', 'attn'}

    def __init__(self,
                 net_type,
                 output_size,
                 hidden_size,
                 token2bert_embeddings,
                 token2wce_embeddings):
        """
        Initialize the BertWCEClassifier with optional Word-Class Embeddings (WCE).

        Parameters:
        ----------
        net_type : str
            Type of network architecture ('cnn', 'lstm', or 'attn').
        output_size : int
            Number of output classes.
        hidden_size : int
            Size of the hidden layer.
        token2bert_embeddings : Token2BertEmbeddings
            BERT-based embeddings.
        token2wce_embeddings : Token2WCEmbeddings or None
            Optional Word-Class Embeddings.
        """
        super(BertWCEClassifier, self).__init__()

        emb_dim = token2bert_embeddings.dim() + (0 if token2wce_embeddings is None else token2wce_embeddings.dim())
        print(f'Embedding dimensions {emb_dim}')

        self.token2bert_embeddings = token2bert_embeddings
        self.token2wce_embeddings = token2wce_embeddings
        
        self.projection = init__projection(net_type)(emb_dim, hidden_size)
        
        self.label = nn.Linear(self.projection.dim(), output_size)


    def forward(self, input):                   # list of lists of tokens
        """
        Forward pass of the BertWCEClassifier.

        Parameters:
        ----------
        input : list of lists of tokens
            Input tokenized text.

        Returns:
        -------
        torch.Tensor
            Logits of shape (batch_size, output_size).
        """

        # convert tokens to id for Bert, pad, and get contextualized embeddings
        contextualized_embeddings = self.token2bert_embeddings.embeddings(input)

        # convert tokens to ids for WCE, pad, and get WCEs
        if self.token2wce_embeddings is not None:
            wce_embeddings = self.token2wce_embeddings(input)
            # concatenate Bert embeddings with WCEs
            assert contextualized_embeddings.shape[1] == wce_embeddings.shape[1], 'shape mismatch between Bert and WCE'
            word_emb = torch.cat([contextualized_embeddings, wce_embeddings], dim=-1)
        else:
            word_emb = contextualized_embeddings

        doc_emb = self.projection(word_emb)
        logits = self.label(doc_emb)
        return logits

    def train(self, mode=True):
        """
        Set the model to training mode.

        Parameters:
        ----------
        mode : bool
            If True, set to training mode. Otherwise, evaluation mode.
        """
        super(BertWCEClassifier, self).train(mode)
        self.token2bert_embeddings.train(mode)
        if self.token2wce_embeddings:
            self.token2wce_embeddings.train(mode)

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.train(False)

    def finetune_pretrained(self):
        """
        Enable fine-tuning for the pretrained embeddings in both BERT and WCE.
        """
        self.token2bert_embeddings.finetune_pretrained()
        if self.token2wce_embeddings:
            self.token2wce_embeddings.finetune_pretrained()

    def xavier_uniform(self):
        """
        Apply Xavier uniform initialization to all learnable parameters.
        """
        for model in [self.token2wce_embeddings, self.projection, self.label]:
            if model is None:
                continue
            for p in model.parameters():
                if p.dim() > 1 and p.requires_grad:
                    nn.init.xavier_uniform_(p)



def init__projection(net_type):
    assert net_type in NeuralClassifier.ALLOWED_NETS, 'unknown network'
    if net_type == 'cnn':
        return CNNprojection
    elif net_type == 'lstm':
        return LSTMprojection
    elif net_type == 'attn':
        return ATTNprojection