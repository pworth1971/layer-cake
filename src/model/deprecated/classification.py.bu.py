import logging
logging.basicConfig(level=logging.INFO)

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertModel
from transformers import DistilBertModel, RobertaModel

# custom imports
from src.model.deprecated.layers import *

from embedding.pretrained import BERT_MODEL




# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# legacy neural classifier (CNN, ATTN, LSTM support)
#

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
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
#
# BERT Embeddings functions (TESTING ONLY)
#
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
    


# --------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# TRANS_LAYER_CAKE Classes and Functions
#
# supported operations for transformer classifier
# combination method with TCEs
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------


from data.lc_trans_dataset import LCTokenizer

SUPPORTED_OPS = ["cat", "add", "dot"]


class LCSequenceClassifier(nn.Module):

    def __init__(self, 
                hf_model: nn.Module,
                num_classes: int, 
                lc_tokenizer: LCTokenizer,
                class_type: str = 'single-label', 
                class_weights: torch.Tensor = None, 
                supervised: bool = False, 
                tce_matrix: torch.Tensor = None, 
                finetune: bool = False, 
                normalize_tces: bool = True,
                dropout_rate: float = 0.3, 
                comb_method: str = "cat", 
                debug: bool = False):
        """
        A Transformer-based Sequence Classifier with optional TCE integration and parameters for Layer Cake Text
        Classification testing. Supports both single label and multi-label classification.
        
        Args:
            hf_model: The HuggingFace pre-trained transformer model (e.g., BERT), preloaded.
            num_classes: Number of classes for classification.
            lc_tokenizer: LCTokenizer object for token verification.
            class_type: type of classification problem, either 'single-label' or 'multi-label'
            class_weights: Class weights for loss function.
            supervised: Boolean indicating if supervised embeddings are used.
            tce_matrix: Precomputed TCE matrix (Tensor) with shape [vocab_size, num_classes].
            finetune: Boolean indicating whether or not the Embedding layer is trainable.
            normalize_tce: Boolean indicating if TCE matrix is normalized.
            dropout: Dropout rate for TCE matrix.
            comb-method: Method to integrate WCE embeddings ("add", "dot" or "cat").            
            debug: Debug mode flag.
        """
        super(LCSequenceClassifier, self).__init__()

        print(f'LCSequenceClassifier:__init__()... class_type: {class_type}, num_classes: {num_classes}, finetune: {finetune}, supervised: {supervised}, debug: {debug}')

        if (supervised):
            print(f'normalize_tces: {normalize_tces}, dropout_rate: {dropout_rate}, comb_method: {comb_method}')

        self.debug = debug

        self.transformer = hf_model
        print("self.transformer:\n", self.transformer)

        self.hidden_size = self.transformer.config.hidden_size          
        print("self.hidden_size:", self.hidden_size)

        self.num_classes = num_classes
        self.class_type = class_type

        self.supervised = supervised
        self.comb_method = comb_method
        self.normalize_tces = normalize_tces
        self.finetune = finetune
        self.class_weights = class_weights

        # --------------------------------------------------------------
        #
        # set up the tokenizer and vocab info to use for input_ind 
        # validation in forward method
        # 
        self.tokenizer = lc_tokenizer.tokenizer
        print("self.tokenizer:\n", self.tokenizer)

        self.vocab_size = len(self.tokenizer.get_vocab())
        print("self.vocab_size:", self.vocab_size)
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_set = set(self.vocab.values())
        #
        # --------------------------------------------------------------

        if (self.class_weights is not None):
            print("self.class_weights.shape:", self.class_weights.shape)
            if (self.debug):
                print("self.class_weights:", self.class_weights)

            # Loss functions
            if class_type == 'multi-label':
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            elif class_type == 'single-label':
                self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                raise ValueError("class_type must be 'single-label' or 'multi-label'")
            print("loss_fn:", self.loss_fn)

        else:
            print("self.class_weights is None")

            # Loss functions
            if class_type == 'multi-label':
                self.loss_fn = nn.BCEWithLogitsLoss()
            elif class_type == 'single-label':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                raise ValueError("class_type must be 'single-label' or 'multi-label'")
            print("loss_fn:", self.loss_fn)

        #
        # Optionally unfreeze the embedding layer if finetune is True
        #
        if self.finetune:
            print("finetuning == True, making the embedding layer (only) trainable")    

            # Freeze gradient computation for all transformer parameters
            for param in self.transformer.parameters():
                param.requires_grad = False

            # Enable training of only the embedding layer
            if hasattr(self.transformer, 'bert'):
                embedding_layer = self.transformer.bert.embeddings
            elif hasattr(self.transformer, 'roberta'):
                embedding_layer = self.transformer.roberta.embeddings
            elif hasattr(self.transformer, 'transformer'):
                embedding_layer = self.transformer.transformer.wte  # GPT-2
            else:
                raise AttributeError(f"Embeddings not found for the given model: {type(self.transformer)}")
            print("embedding_layer:", type(embedding_layer), embedding_layer)

            for param in embedding_layer.parameters():
                param.requires_grad = True
        else:            
            print("finetune == False, default model configuration ...")


        # Assert that tce_matrix is provided if supervised is True
        if self.supervised:
            assert tce_matrix is not None, "tce_matrix must be provided when supervised is True."

        self.tce_matrix = tce_matrix
        self.tce_layer = None               # we initialize this only if we are using TCEs 

        # initialize embedding dimensions to model embedding dimension
        # we over-write this only if we are using supervised tces with the 'cat' method
        combined_size = self.hidden_size
        
        print(f'self.hidden_size: {self.hidden_size}')
        print("combined_size:", combined_size)
            
        if (self.supervised and self.tce_matrix is not None):

            print("supervised is True, original tce_matrix:", type(self.tce_matrix), self.tce_matrix.shape)
 
            with torch.no_grad():                           # normalization code should be in this block

                # Normalize TCE matrix if required
                if self.normalize_tces:

                    # compute the mean and std from the core model embeddings
                    embedding_layer = self.transformer.get_input_embeddings()
                    embedding_mean = embedding_layer.weight.mean(dim=0).to(device)
                    embedding_std = embedding_layer.weight.std(dim=0).to(device)
                    if (self.debug):
                        print(f"transformer embeddings mean: {embedding_mean.shape}, std: {embedding_std.shape}")

                    # normalize the TCE matrix
                    self.tce_matrix = self._normalize_tce(self.tce_matrix, embedding_mean, embedding_std)
                    print(f"Normalized TCE matrix: {type(self.tce_matrix)}, {self.tce_matrix.shape}")
            
                # initialize the TCE Embedding layer, freeze the embeddings if trainable_tces == False
                """
                if (trainable_tces):
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=False)
                else:
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=True)
                """

                self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=True)
                print("self.tce_layer:", self.tce_layer)

                # Adapt classifier head based on combination method
                # 'concat' method is dimension_size + num_classes
                if (self.comb_method == 'cat'):
                    combined_size = self.hidden_size + self.tce_matrix.size(1)
                else:
                    combined_size = self.hidden_size                                # redundant but for clarity - good for 'add' or 'dot'

                """
                #
                # initialize the TCE Embedding layer, freeze the embeddings if trainable_tces == False
                # otherwise let the model train the tce embedding layer
                #
                if (finetune):
                    print("finetuning, retraining tce embedding layer...")
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=False)
                else:
                    print("not finetuning, not retraining tce embedding layer...")
                    self.tce_layer = nn.Embedding.from_pretrained(self.tce_matrix, freeze=True)
                """

        # -----------------------------------------------------------------------------------------------
        # 
        # initialize Classification head: maps the (potentially) combined input (transformer output or transformer output + optional TCE embeddings) 
        # to the final logits, introducing additional learnable parameters and allowing for flexibility to adapt the model to the specific task.
        #

        print("combined_size:", combined_size)

        # simplified classification head that adjusts the size of the Linear layer according  
        # to the method we are using to combine TCEs with built in transformer embeddings
        self.classifier = nn.Linear(combined_size, self.num_classes)                                  
        
        print("self.classifier:", self.classifier)
        
        #
        # force all of the tensors to be stored contiguously in memory
        #
        for param in self.transformer.parameters():
            param.data = param.data.contiguous()
    

    def _normalize_tce(self, tce_matrix, embedding_mean, embedding_std):
        """
        Normalize the TCE matrix to align with the transformer's embedding distribution.

        Args:
            tce_matrix: TCE matrix (vocab_size x num_classes).
            embedding_mean: Mean of the transformer embeddings (1D tensor, size=model_dim).
            embedding_std: Standard deviation of the transformer embeddings (1D tensor, size=model_dim).

        Returns:
            Normalized TCE matrix (vocab_size x num_classes).
        """

        target_dim = embedding_mean.shape[0]  # Set target_dim to model embedding size

        if (self.debug):
            print(f"tce_matrix: {tce_matrix.shape}, {tce_matrix.dtype}")
            #print("first row:", tce_matrix[0])
            print(f"embedding_mean: {embedding_mean.shape}, {embedding_mean.dtype}")
            #print(f'embedding_mean: {embedding_mean}')
            print(f"embedding_std: {embedding_std.shape}, {embedding_std.dtype}")
            #print(f'embedding_std: {embedding_std}')
            print("target_dim:", target_dim)

        device = embedding_mean.device                      # Ensure all tensors are on the same device
        tce_matrix = tce_matrix.to(device)

        # 1 Normalize TCE matrix row-wise (i.e. ompute mean and std per row)
        tce_mean = tce_matrix.mean(dim=1, keepdim=True)
        tce_std = tce_matrix.std(dim=1, keepdim=True)
        tce_std[tce_std == 0] = 1                           # Prevent division by zero

        if (self.debug):
            print(f"tce_mean: {tce_mean.shape}, {tce_mean.dtype}")
            #print(f'tce_mean: {tce_mean}')
            print(f"tce_std: {tce_std.shape}, {tce_std.dtype}")
            #print(f'tce_std: {tce_std}')

        normalized_tce = (tce_matrix - tce_mean) / tce_std

        if (self.debug):
            print(f"normalized_tce (pre-scaling): {normalized_tce.shape}")

        # 2. Scale to match embedding statistics
        normalized_tce = normalized_tce * embedding_std.mean() + embedding_mean.mean()

        # 3. Project normalized TCE into the target dimension (e.g., 128)
        projection = torch.nn.Linear(tce_matrix.size(1), target_dim, bias=False).to(device)
        projected_tce = projection(normalized_tce)

        if self.debug:
            print(f"Projected TCE matrix: {projected_tce.shape}")

        # check for Nan or Inf values after normalization
        if torch.isnan(projected_tce).any() or torch.isinf(projected_tce).any():
            print("[WARNING]: projected_tce contains NaN or Inf values after normalization.")
            #raise ValueError("[ERROR] projected_tce contains NaN or Inf values after normalization.")

        return projected_tce


    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass for the LCSequenceClassifier, includes support for integrated
        TCE computation in the event that the Classifier has been set up with TCEs (ie supervised is True)

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size x seq_length).
            attention_mask (torch.Tensor): Attention mask for input tokens.
            labels (torch.Tensor, optional): Labels for computing loss.

        Returns:
            logits (torch.Tensor): Output logits from the classifier.
            loss (torch.Tensor, optional): Loss value if labels are provided.
        """
        if (self.debug):
            print("LCSequenceClassifier:forward()...")
            print(f"\tinput_ids: {type(input_ids)}, {input_ids.shape}")
            print("input_ids:", input_ids)
            print(f"\tattention_mask: {type(attention_mask)}, {attention_mask.shape}")
            print(f"\tlabels: {type(labels)}, {labels.shape}")

        # Validate input_ids
        invalid_tokens = [token.item() for token in input_ids.flatten() if token.item() not in self.vocab_set]
        if invalid_tokens:
            raise ValueError(
                f"Invalid token IDs found in input_ids: {invalid_tokens[:10]}... "
                f"(total {len(invalid_tokens)}) out of tokenizer vocabulary size {len(self.vocab_set)}."
            )

        # Pass inputs through the transformer model, ie Base model forward pass
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]  # Use hidden states if available
        else:
            raise AttributeError("Transformer model did not output hidden states. Ensure output_hidden_states=True is set.")
        if (self.debug):
            print(f"last_hidden_state: {type(last_hidden_state)}, {last_hidden_state.shape}")

        # default 
        combined_output = last_hidden_state     # Use only the transformer outputs


        # -----------------------------------------------------------------------------------------------
        #
        # Integrate TCEs if supervised is True
        #
        #if (self.supervised and (self.tce_matrix is not None)):
        if self.supervised:
            
            if self.tce_layer is None:
                raise ValueError("[ERROR]:supervised is True but tce_layer embedding layer is None.")
            
            if (self.debug):
                print("integrating TCEs into combined_output...")

            # Debug info: Check for out-of-range indices
            invalid_indices = input_ids[input_ids >= self.vocab_size]
            if invalid_indices.numel() > 0:
                print(f"[WARNING] Found invalid indices in input_ids: {invalid_indices.tolist()} (max valid index: {self.vocab_size - 1})")

            # Extract all relevant indices for pooling TCE embeddings
            tce_indices = input_ids                                     # Assuming all input tokens are relevant for TCEs
            tce_embeddings = self.tce_layer(tce_indices)                # (batch_size, seq_length, tce_dim)

            # Combine transformer outputs with TCE embeddings
            if self.comb_method == 'cat':
                combined_output = torch.cat((last_hidden_state, tce_embeddings), dim=-1)  # (batch_size, seq_length, hidden_size + tce_dim)
            elif self.comb_method == 'add':
                combined_output = last_hidden_state + tce_embeddings  # Element-wise addition
            elif self.comb_method == 'dot':
                combined_output = last_hidden_state * tce_embeddings  # Element-wise multiplication
            else:
                raise ValueError(f"Unsupported combination method: {self.comb_method}")
        #
        # -----------------------------------------------------------------------------------------------

        if (self.debug):
            print(f"combined_output: {type(combined_output)}, {combined_output.shape}")

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(combined_output.size())
            masked_output = combined_output * mask_expanded
            sum_hidden_state = torch.sum(masked_output, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)  # Avoid division by zero
            pooled_output = sum_hidden_state / sum_mask
        else:
            print("WARNING: No attention mask provided for pooling.")
            pooled_output = combined_output.mean(dim=1)  # Fallback to mean pooling

        if self.debug:
            print(f"pooled_output: {type(pooled_output)}, {pooled_output.shape}")

        # send pooled_output to classification head
        logits = self.classifier(pooled_output)
        if (self.debug):
            print(f'logits: {type(logits)}, {logits.shape}, {logits}')

        #
        # compute loss
        #
        loss = None
        if labels is not None:
            if self.class_type in ['multi-label', 'multilabel']:
                # BCEWithLogitsLoss requires float labels for multi-label classification
                loss = self.loss_fn(logits, labels.float())
            elif self.class_type in ['single-label', 'singlelabel']:
                # CrossEntropyLoss expects long/int labels for single-label classification
                loss = self.loss_fn(logits, labels.long())
            else:
                raise ValueError(f"Unsupported classification type: {self.class_type}")
        else:
            print("WARNINMG: No labels provided for loss calculation.")

        return {"loss": loss, "logits": logits}
    

    def get_embedding_dims(self):
        """
        Retrieve the dimensions of the embedding layer.

        Returns:
            Tuple[int, int]: A tuple containing the vocabulary size and embedding dimension.
        """
        # Identify the embedding layer dynamically
        if hasattr(self.transformer, 'bert'):
            embedding_layer = self.transformer.bert.embeddings.word_embeddings
        elif hasattr(self.transformer, 'roberta'):
            embedding_layer = self.transformer.roberta.embeddings.word_embeddings
        elif hasattr(self.transformer, 'distilbert'):
            embedding_layer = self.transformer.distilbert.embeddings.word_embeddings
        elif hasattr(self.transformer, 'transformer'):        
            if hasattr(self.transformer.transformer, 'word_embedding'):                                     # XLNet
                embedding_layer = self.transformer.transformer.word_embedding
            elif hasattr(self.transformer.transformer, 'wte'):                                              # GPT-2
                embedding_layer = self.transformer.transformer.wte
            else:
                raise ValueError("Unsupported model type or embedding layer not found.")
        elif hasattr(self.transformer, 'model') and hasattr(self.transformer.model, 'embed_tokens'):  # LLaMA
            embedding_layer = self.transformer.model.embed_tokens
        else:
            raise ValueError("Unsupported model type or embedding layer not found.")

        # Extract dimensions
        vocab_size, embedding_dim = embedding_layer.weight.size()

        if self.debug:
            print(f"Embedding layer dimensions: vocab_size={vocab_size}, embedding_dim={embedding_dim}")

        return vocab_size, embedding_dim





# --------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Simple Classifier modified form of original code from: https://github.com/ashfarhangi/Protoformer
#

"""

RANDOM_SEED = 47
torch.manual_seed(RANDOM_SEED) 


MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-05
WEIGHT_DECAY = 1e-05

num_classes = len(df_profile.labels.unique())
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
baseline_model = BERTClass(num_classes)
baseline_model.to(device)
"""

# --------------------------------------------------------------------------------------------------------------------------------------------------------------






class LCBERTBaseClassifier(nn.Module):
    """
    Base class for all BERT based LC classifiers (BERT, RoBERTa, DistilBERT).

    Parameters:
    - model_name: Name of the model architecture.
    - cache_dir: Directory for loading the pre-trained model.
    - num_classes: Number of output classes for classification.
    - class_type: Classification type ('single-label' or 'multi-label').
    - debug: Whether to enable debug logging.
    """
    def __init__(self, model_name, cache_dir, num_classes, class_type, debug=False):

        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.class_type = class_type
        self.hidden_size = None  # To be set dynamically
        self.debug = debug

        # Initialize loss function based on classification type
        if class_type in ['multi-label', 'multilabel']:
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif class_type in ['single-label', 'singlelabel']:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError("class_type must be 'single-label' or 'multi-label'")
        print("loss_fn:", self.loss_fn)

        self.dropout = nn.Dropout(0.6)
        self.classifier = None  # To be initialized after getting hidden size

    def forward(self, input_ids, attention_mask, labels=None):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def get_embedding_dims(self):
        """
        Returns the dimensions of the embedding layer as (hidden_size, num_classes).
        """
        return self.l1.embeddings.word_embeddings.weight.shape

    def compute_loss(self, logits, labels):
        """
        Compute the loss using the provided logits and labels.
        """
        loss = None
        if labels is not None:
            if self.class_type in ['multi-label', 'multilabel']:
                loss = self.loss_fn(logits, labels.float())
            elif self.class_type in ['single-label', 'singlelabel']:
                loss = self.loss_fn(logits, labels.long())
        return loss
    

    def _normalize_tce(self, tce_matrix, embedding_mean, embedding_std):
        """
        Normalize the TCE matrix to align with the transformer's embedding distribution.

        Args:
            tce_matrix: TCE matrix (vocab_size x num_classes).
            embedding_mean: Mean of the transformer embeddings (1D tensor, size=model_dim).
            embedding_std: Standard deviation of the transformer embeddings (1D tensor, size=model_dim).

        Returns:
            Normalized TCE matrix (vocab_size x num_classes).
        """

        target_dim = embedding_mean.shape[0]  # Set target_dim to model embedding size

        if (self.debug):
            print(f"tce_matrix: {tce_matrix.shape}, {tce_matrix.dtype}")
            #print("first row:", tce_matrix[0])
            print(f"embedding_mean: {embedding_mean.shape}, {embedding_mean.dtype}")
            #print(f'embedding_mean: {embedding_mean}')
            print(f"embedding_std: {embedding_std.shape}, {embedding_std.dtype}")
            #print(f'embedding_std: {embedding_std}')
            print("target_dim:", target_dim)

        device = embedding_mean.device                      # Ensure all tensors are on the same device
        tce_matrix = tce_matrix.to(device)

        # 1 Normalize TCE matrix row-wise (i.e. ompute mean and std per row)
        tce_mean = tce_matrix.mean(dim=1, keepdim=True)
        tce_std = tce_matrix.std(dim=1, keepdim=True)
        tce_std[tce_std == 0] = 1                           # Prevent division by zero

        if (self.debug):
            print(f"tce_mean: {tce_mean.shape}, {tce_mean.dtype}")
            #print(f'tce_mean: {tce_mean}')
            print(f"tce_std: {tce_std.shape}, {tce_std.dtype}")
            #print(f'tce_std: {tce_std}')

        normalized_tce = (tce_matrix - tce_mean) / tce_std

        if (self.debug):
            print(f"normalized_tce (pre-scaling): {normalized_tce.shape}")

        # 2. Scale to match embedding statistics
        normalized_tce = normalized_tce * embedding_std.mean() + embedding_mean.mean()

        # 3. Project normalized TCE into the target dimension (e.g., 128)
        projection = torch.nn.Linear(tce_matrix.size(1), target_dim, bias=False).to(device)
        projected_tce = projection(normalized_tce)

        if self.debug:
            print(f"Projected TCE matrix: {projected_tce.shape}")

        # check for Nan or Inf values after normalization
        if torch.isnan(projected_tce).any() or torch.isinf(projected_tce).any():
            print("[WARNING]: projected_tce contains NaN or Inf values after normalization.")
            #raise ValueError("[ERROR] projected_tce contains NaN or Inf values after normalization.")

        return projected_tce


    def _filter_transformer_embeddings(self, model, relevant_tokens):
        """
        Replace the transformer embedding layer with a reduced embedding matrix and update the tokenizer.
        """
        print("Filtering transformer embeddings to align with dataset...")
        print(f"relevant_tokens: {type(relevant_tokens)} {len(relevant_tokens)}")

        # Extract original embeddings
        embedding_layer = model.get_input_embeddings()
        original_embeddings = embedding_layer.weight.data
        print("original_embeddings:", type(original_embeddings), original_embeddings.shape)

        # Validate relevant_tokens
        assert all(0 <= idx < original_embeddings.size(0) for idx in relevant_tokens), \
            "Relevant tokens contain indices out of range of the original embeddings."

        # Get embeddings for relevant tokens
        relevant_embeddings = original_embeddings[relevant_tokens, :]
        print("relevant_embeddings:", type(relevant_embeddings), relevant_embeddings.shape)

        # Replace the embedding layer
        reduced_embedding_layer = nn.Embedding.from_pretrained(relevant_embeddings)
        print("reduced_embedding_layer:", type(reduced_embedding_layer), reduced_embedding_layer)

        model.set_input_embeddings(reduced_embedding_layer)
        print("updated model:", model)

        return model, relevant_embeddings
    


class LCLinearBaseClassifier(LCBERTBaseClassifier):
    """
    Base class for linear classifiers.
    """
    def __init__(self, model_class, model_name, cache_dir, num_classes, class_type, debug=False):
        super().__init__(model_name, cache_dir, num_classes, class_type, debug)
        self.l1 = model_class.from_pretrained(model_name, cache_dir=cache_dir, output_hidden_states=True)
        self.hidden_size = self.l1.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output[0][:, 0]  # CLS token representation
        logits = self.classifier(self.dropout(pooled_output))

        if self.debug:
            print(f"logits: {logits.shape}")

        # Compute loss if labels are provided
        loss = self.compute_loss(logits, labels)
        return {"loss": loss, "logits": logits}


# Model-specific subclasses
class LCLinearBERTClassifier(LCLinearBaseClassifier):
    def __init__(self, model_name, cache_dir, num_classes, class_type, debug=False):
        super().__init__(BertModel, model_name, cache_dir, num_classes, class_type, debug)

class LCLinearRoBERTaClassifier(LCLinearBaseClassifier):
    def __init__(self, model_name, cache_dir, num_classes, class_type, debug=False):
        super().__init__(RobertaModel, model_name, cache_dir, num_classes, class_type, debug)

class LCLinearDistilBERTClassifier(LCLinearBaseClassifier):
    def __init__(self, model_name, cache_dir, num_classes, class_type, debug=False):
        super().__init__(DistilBertModel, model_name, cache_dir, num_classes, class_type, debug)




class LCCNNBaseClassifier(LCBERTBaseClassifier):
    """
    Base class for CNN-based classifiers.
    """
    def __init__(self, model_class, model_name, cache_dir, num_classes, class_type, num_channels=256, relevant_tokens=None, debug=False):

        super().__init__(model_name, cache_dir, num_classes, class_type, debug)

        print(f"LCCNNBaseClassifier:__init__()... model_name: {model_name}, cache_dir: {cache_dir}, num_classes: {num_classes}, class_type: {class_type}, num_channels: {num_channels}, debug: {debug}")

        if (relevant_tokens is not None):
            print(f"relevant_tokens: {type(relevant_tokens)}, {len(relevant_tokens)}")
            self.l1, _ = self._filter_transformer_embeddings(self.l1, relevant_tokens)
        else:
            self.l1 = model_class.from_pretrained(model_name, cache_dir=cache_dir, output_hidden_states=True)
        
        self.hidden_size = self.l1.config.hidden_size

        # CNN-based layers
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=(3, self.hidden_size), stride=1)
        self.conv2 = nn.Conv2d(1, num_channels, kernel_size=(5, self.hidden_size), stride=1)
        self.conv3 = nn.Conv2d(1, num_channels, kernel_size=(7, self.hidden_size), stride=1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(num_channels * 3, num_classes)  # num_channels filters * 3 convolution layers

    def forward(self, input_ids, attention_mask, labels=None):
        # Get transformer embeddings
        output = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output[0]  # Shape: (batch_size, seq_len, hidden_size)

        # Reshape for CNN (add channel dimension)
        hidden_states = hidden_states.unsqueeze(1)  # Shape: (batch_size, 1, seq_len, hidden_size)

        # Apply convolutions
        conv1_out = torch.relu(self.conv1(hidden_states)).squeeze(3)
        conv2_out = torch.relu(self.conv2(hidden_states)).squeeze(3)
        conv3_out = torch.relu(self.conv3(hidden_states)).squeeze(3)

        # Apply max pooling
        pool1 = torch.max(conv1_out, dim=2)[0]
        pool2 = torch.max(conv2_out, dim=2)[0]
        pool3 = torch.max(conv3_out, dim=2)[0]

        # Concatenate pooled outputs
        features = torch.cat((pool1, pool2, pool3), dim=1)

        # Apply dropout and classification layer
        features = self.dropout(features)
        logits = self.classifier(features)

        if self.debug:
            print(f"logits: {logits.shape}")

        # Compute loss if labels are provided
        loss = self.compute_loss(logits, labels)
        return {"loss": loss, "logits": logits}



# Model-specific subclasses
class LCCNNBERTClassifier(LCCNNBaseClassifier):
    def __init__(self, model_name, cache_dir, num_classes, class_type, num_channels=256, relevant_tokens=None, debug=False):
        super().__init__(BertModel, model_name, cache_dir, num_classes, class_type, num_channels, relevant_tokens, debug)

class LCCNNRoBERTaClassifier(LCCNNBaseClassifier):
    def __init__(self, model_name, cache_dir, num_classes, class_type, num_channels=256, relevant_tokens=None, debug=False):
        super().__init__(RobertaModel, model_name, cache_dir, num_classes, class_type, num_channels, relevant_tokens, debug)

class LCCNNDistilBERTClassifier(LCCNNBaseClassifier):
    def __init__(self, model_name, cache_dir, num_classes, class_type, num_channels=256, relevant_tokens=None, debug=False):
        super().__init__(DistilBertModel, model_name, cache_dir, num_classes, class_type, num_channels, relevant_tokens, debug)




class LC_Linear_BERT_Classifier_orig(LCBERTBaseClassifier):

    def __init__(self, model_name, cache_dir, num_classes, class_type, debug=False):
        super().__init__(model_name=model_name, cache_dir=cache_dir, num_classes=num_classes, class_type=class_type, debug=debug)
        self.l1 = BertModel.from_pretrained(model_name, cache_dir=cache_dir, output_hidden_states=True)
        self.hidden_size = self.l1.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output_1[0][:, 0]  # CLS token representation
        logits = self.classifier(self.dropout(pooled_output))

        if self.debug:
            print(f"logits: {logits.shape}")

        # Compute loss if labels are provided
        loss = self.compute_loss(logits, labels)
        return {"loss": loss, "logits": logits}
    



class LC_CNN_BERT_Classifier_orig(LCBERTBaseClassifier):

    def __init__(self, model_name, cache_dir, num_classes, class_type, num_channels, debug=False):

        super().__init__(
            model_name=model_name, 
            cache_dir=cache_dir, 
            num_classes=num_classes, 
            class_type=class_type, 
            debug=debug)

        def_model = BertModel.from_pretrained(model_name, cache_dir=cache_dir, output_hidden_states=True)

        # ----------------------------------------------------------------------------------------------------------------------------
        #
        # Filter the transformer embeddings to only include relevant tokens
        # that are present from the Dataset vocabulary (as tokenized)
        #
        if relevant_tokens is not None:
            self.l1, self.relevant_embeddings = self._filter_transformer_embeddings(
                                                            model=def_model, 
                                                            relevant_tokens=relevant_tokens
                                                            )                    
        else:
            self.l1 = model
            self.relevant_embeddings = None
        # ----------------------------------------------------------------------------------------------------------------------------

        self.hidden_size = self.l1.config.hidden_size
        print("self.hidden_size:", self.hidden_size)

        # CNN-based classifier
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=(3, self.hidden_size), stride=1)
        self.conv2 = nn.Conv2d(1, num_channels, kernel_size=(5, self.hidden_size), stride=1)
        self.conv3 = nn.Conv2d(1, num_channels, kernel_size=(7, self.hidden_size), stride=1)
        self.dropout = nn.Dropout(0.5)

        # Classification layer
        self.classifier = nn.Linear(num_channels * 3, num_classes)  # num_channels filters * 3 convolution layers
        print("self.classifier:", self.classifier)

    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT embeddings
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output_1[0]  # Shape: (batch_size, seq_len, hidden_size)

        # Reshape for CNN (add channel dimension)
        hidden_states = hidden_states.unsqueeze(1)  # Shape: (batch_size, 1, seq_len, hidden_size)

        # Apply convolutions
        conv1_out = torch.relu(self.conv1(hidden_states)).squeeze(3)  # Shape: (batch_size, num_filters, seq_len - 2)
        conv2_out = torch.relu(self.conv2(hidden_states)).squeeze(3)  # Shape: (batch_size, num_filters, seq_len - 4)
        conv3_out = torch.relu(self.conv3(hidden_states)).squeeze(3)  # Shape: (batch_size, num_filters, seq_len - 6)

        # Apply max pooling
        pool1 = torch.max(conv1_out, dim=2)[0]  # Shape: (batch_size, num_filters)
        pool2 = torch.max(conv2_out, dim=2)[0]  # Shape: (batch_size, num_filters)
        pool3 = torch.max(conv3_out, dim=2)[0]  # Shape: (batch_size, num_filters)

        # Concatenate pooled outputs
        features = torch.cat((pool1, pool2, pool3), dim=1)  # Shape: (batch_size, num_filters * 3)

        # Apply dropout and classification layer
        features = self.dropout(features)
        logits = self.classifier(features)

        if self.debug:
            print(f"logits: {logits.shape}")

        # Compute loss if labels are provided
        loss = self.compute_loss(logits, labels)
        return {"loss": loss, "logits": logits}






class LC_Linear_RoBERTa_Classifier_orig(LCBERTBaseClassifier):

    def __init__(self, model_name, cache_dir, num_classes, class_type, debug=False):
        super().__init__(model_name=model_name, cache_dir=cache_dir, num_classes=num_classes, class_type=class_type, debug=debug)
        self.l1 = RobertaModel.from_pretrained(model_name, cache_dir=cache_dir, output_hidden_states=True)
        self.hidden_size = self.l1.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)


    def forward(self, input_ids, attention_mask, labels=None):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output_1[0][:, 0]  # CLS token representation
        logits = self.classifier(self.dropout(pooled_output))

        if self.debug:
            print(f"logits: {logits.shape}")

        # Compute loss if labels are provided
        loss = self.compute_loss(logits, labels)
        return {"loss": loss, "logits": logits}


class LC_Linear_DistillBERT_Classifier_orig(LCBERTBaseClassifier):

    def __init__(self, model_name, cache_dir, num_classes, class_type, debug=False):
        super().__init__(model_name=model_name, cache_dir=cache_dir, num_classes=num_classes, class_type=class_type, debug=debug)
        self.l1 = DistilBertModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.hidden_size = self.l1.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output_1[0][:, 0]  # CLS token representation
        logits = self.classifier(self.dropout(pooled_output))
        
        if self.debug:
            print(f"logits: {logits.shape}")

        # Compute loss if labels are provided
        loss = self.compute_loss(logits, labels)
        return {"loss": loss, "logits": logits}














class LC_LSTM_BERT_Classifier(LCBERTBaseClassifier):

    def __init__(self, model_name, cache_dir, num_classes, class_type, hidden_size=128, debug=False):
        super().__init__(model_name=model_name, cache_dir=cache_dir, num_classes=num_classes, class_type=class_type, debug=debug)
        self.l1 = BertModel.from_pretrained(model_name, cache_dir=cache_dir, output_hidden_states=True)
        self.hidden_size = self.l1.config.hidden_size

        # LSTM-based projection
        self.lstm = nn.LSTM(self.hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output_1[0]  # Raw embeddings: (batch_size, seq_len, hidden_size)

        # Pass through LSTM
        lstm_out, _ = self.lstm(hidden_states)  # LSTM output: (batch_size, seq_len, hidden_size)
        final_hidden_state = lstm_out[:, -1, :]  # Use the last hidden state

        logits = self.fc(final_hidden_state)

        if self.debug:
            print(f"logits: {logits.shape}")

        # Compute loss if labels are provided
        loss = self.compute_loss(logits, labels)
        return {"loss": loss, "logits": logits}


class LC_ATTN_BERT_Classifier(LCBERTBaseClassifier):

    def __init__(self, model_name, cache_dir, num_classes, class_type, hidden_size=128, debug=False):
        super().__init__(model_name=model_name, cache_dir=cache_dir, num_classes=num_classes, class_type=class_type, debug=debug)
        self.l1 = BertModel.from_pretrained(model_name, cache_dir=cache_dir, output_hidden_states=True)
        self.hidden_size = self.l1.config.hidden_size

        # Attention-based projection
        self.lstm = nn.LSTM(self.hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)  # (batch_size, hidden_size)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)
        soft_attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input_ids, attention_mask, labels=None):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = output_1[0]  # Raw embeddings: (batch_size, seq_len, hidden_size)

        # Pass through LSTM
        lstm_out, (final_hidden_state, _) = self.lstm(hidden_states)  # (batch_size, seq_len, hidden_size)

        # Apply attention
        attn_output = self.attention_net(lstm_out, final_hidden_state)

        logits = self.fc(attn_output)

        if self.debug:
            print(f"logits: {logits.shape}")

        # Compute loss if labels are provided
        loss = self.compute_loss(logits, labels)
        return {"loss": loss, "logits": logits}




class LCDataFormat(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        cur_doc = str(self.data.doc[index])
        cur_doc = " ".join(cur_doc.split())
        inputs = self.tokenizer.encode_plus(
            cur_doc,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.labels[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len



# -----------------------------------------------------------------------------------------------------------------------