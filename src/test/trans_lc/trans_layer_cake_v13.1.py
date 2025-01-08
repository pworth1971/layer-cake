import argparse
import os
import numpy as np
from time import time

from scipy.sparse import csr_matrix

# sklearn
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder


# PyTorch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# HuggingFace Transformers library
import transformers     
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


# Custom imports and classes and utilities for the Layer Cake
from data.lc_trans_dataset import RANDOM_SEED, MAX_TOKEN_LENGTH, SUPPORTED_DATASETS, VECTOR_CACHE, VAL_SIZE
from data.lc_trans_dataset import get_dataset_data, show_class_distribution, check_empty_docs, spot_check_documents
from data.lc_trans_dataset import LCTokenizer, get_vectorized_data, lc_class_weights

from util.metrics import evaluation_nn
from util.common import initialize_testing, get_embedding_type

from embedding.supervised import compute_tces
from embedding.pretrained import MODEL_MAP
from model.classification import SUPPORTED_OPS, LCSequenceClassifier, LCCNNBERTClassifier, LCCNNDistilBERTClassifier, LCCNNRoBERTaClassifier, LCCNNXLNetClassifier, LCCNNGPT2Classifier
from model.classification import LCLinearBERTClassifier, LCLinearDistilBERTClassifier, LCLinearRoBERTaClassifier, LCLinearXLNetClassifier, LCLinearGPT2Classifier


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
#
# hyper parameters
#
MC_THRESHOLD = 0.5                  # Multi-class threshold
PATIENCE = 5                        # Early stopping patience
LEARNING_RATE = 1e-6                # Learning rate
EPOCHS = 33


#
# batch sizes for training and testing
# set depending upon dataset size
#
DEFAULT_MIN_CPU_BATCH_SIZE = 8
DEFAULT_MIN_MPS_BATCH_SIZE = 8
DEFAULT_MIN_CUDA_BATCH_SIZE = 8

DEFAULT_MAX_CPU_BATCH_SIZE = 32
DEFAULT_MAX_MPS_BATCH_SIZE = 32
DEFAULT_MAX_CUDA_BATCH_SIZE = 32


    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


# Get the full model identifier and load from local directory
def get_model_identifier(pretrained, cache_dir=VECTOR_CACHE):

    model_name = MODEL_MAP.get(pretrained, pretrained)
    model_path = os.path.join(cache_dir, pretrained)

    return model_name, model_path
    


class LCDataset(Dataset):
    """
    Dataset class for handling both input text and labels for LCHFTCEClassifier

    Parameters:
    - texts: List of input text samples.
    - labels: Multi-label binary vectors or single-label indices.
    - tokenizer: Hugging Face tokenizer for text tokenization.
    - max_length: Maximum length of tokenized sequences.
    - class_type: 'multi-label' or 'single-label' classification type.
    """

    def __init__(self, texts, labels, tokenizer, max_length=MAX_TOKEN_LENGTH, class_type='multi-label'):
        
        print(f'LCDataset() class_type: {class_type}, len(texts): {len(texts)}, len(labels): {len(labels)}, max_length: {max_length}')

        assert len(texts) == len(labels), "Mismatch between texts and labels lengths."

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.class_type = class_type


    def __len__(self):
        return len(self.texts)

        
    def shape(self):
        """
        Returns the shape of the dataset as a tuple: (number of texts, number of labels).
        """
        return len(self.texts), len(self.labels) if self.labels is not None else 0
    

    def __getitem__(self, idx):
        """
        Get an individual sample from the dataset.

        Returns:
        - item: Dictionary containing tokenized inputs, labels, and optionally supervised embeddings.
        """
        text = self.texts[idx]

        if self.labels is not None:
            labels = self.labels[idx] 
        else:
            print("WARNING: No labels provided for LCDataset dataset item at index", idx) 
            0  # Default label is 0 if missing

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}  # Remove batch dim

        # Add labels
        if self.class_type == 'single-label':
            item["labels"] = torch.tensor(labels, dtype=torch.long)
        else:
            item["labels"] = torch.tensor(labels, dtype=torch.float)

        return item



def custom_data_collator(batch):
    """
    Custom data collator for handling variable-length sequences and TCEs.

    Parameters:
    - batch: List of individual samples from the dataset.

    Returns:
    - collated: Dictionary containing collated inputs, labels, and optionally TCEs.
    """
    collated = {
        "input_ids": torch.stack([f["input_ids"] for f in batch]),
        "attention_mask": torch.stack([f["attention_mask"] for f in batch]),
    }

    if "labels" in batch[0]:
        collated["labels"] = torch.stack([f["labels"] for f in batch])

    return collated



def compute_metrics(eval_pred, class_type='single-label', threshold=0.5):
    """
    Compute evaluation metrics for classification tasks.

    Args:
    - eval_pred: `EvalPrediction` object with `predictions` and `label_ids`.
    - class_type: 'single-label' or 'multi-label'.
    - threshold: Threshold for binary classification in multi-label tasks.

    Returns:
    - Dictionary of computed metrics.
    """
    """
    print(f'compute_metrics()... class_type: {class_type}')
    
    if (class_type in ['multi-label', 'multilabel']):
        print(f'threshold: {threshold}')
    """

    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    if class_type == 'single-label':
        # Convert predictions to class indices
        preds = np.argmax(predictions, axis=1)                  
    elif class_type == 'multi-label':
        # Threshold predictions for multi-label classification
        preds = (predictions > threshold).astype(int)
        labels = labels.astype(int)                                     # Ensure labels are binary
    else:
        raise ValueError(f"Unsupported class_type: {class_type}")

    """
    print("preds:", type(preds), preds.shape)
    print("labels:", type(labels), labels.shape)
    """
    
    # Compute metrics
    f1_micro = f1_score(labels, preds, average='micro', zero_division=1)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=1)
    precision = precision_score(labels, preds, average='micro', zero_division=1)
    recall = recall_score(labels, preds, average='micro', zero_division=1)

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
    }



def get_hf_models(model_name, model_path, num_classes, tokenizer):
    """
    Load Hugging Face transformer models with configurations to enable hidden states.
    Ensure pad_token_id is consistently fetched from the tokenizer.

    Parameters:
    - model_name: Name of the Hugging Face model.
    - model_path: Path to the Hugging Face model.
    - num_classes: Number of classes for classification.
    - tokenizer: Hugging Face tokenizer.

    Returns:
    - hf_trans_model: Hugging Face transformer model.
    - hf_trans_class_model: Hugging Face transformer model for classification.
    """

    print(f'get_hf_models(): model_name: {model_name}, model_path: {model_path}, num_classes: {num_classes}')

    # Ensure the tokenizer has a pad_token_id set
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer does not have a valid pad_token_id. Ensure the tokenizer is properly configured before calling this function.")

    pad_token_id = tokenizer.pad_token_id
    print(f"Using pad_token_id: {pad_token_id} (token: {tokenizer.pad_token})")

    # Initialize Hugging Face Transformer model
    hf_trans_model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
    
    # Ensure hidden states are enabled for the base model
    hf_trans_model.config.output_hidden_states = True

    # Ensure the model config uses the tokenizer's pad_token_id
    if hf_trans_model.config.pad_token_id is None:
        print("hf_trans_model padding token ID is None, setting to tokenizer.pad_token_id...")
        hf_trans_model.config.pad_token_id = pad_token_id

    # Initialize Hugging Face Transformer model for sequence classification
    hf_trans_class_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=model_path,
        num_labels=num_classes,      # Specify the number of classes for classification
        pad_token_id=pad_token_id    # Use tokenizer's pad_token_id
    )

    # Ensure hidden states are enabled for the classification model
    hf_trans_class_model.config.output_hidden_states = True

    # Ensure the model config uses the tokenizer's pad_token_id
    if hf_trans_class_model.config.pad_token_id is None:
        print("hf_trans_class_model padding token ID is None, setting to tokenizer.pad_token_id...")
        hf_trans_class_model.config.pad_token_id = pad_token_id

    return hf_trans_model, hf_trans_class_model



def compute_dataset_vocab(train_texts, val_texts, test_texts, tokenizer):
    """
    Compute the dataset vocabulary as token IDs that align with the tokenizer's vocabulary.
    
    Args:
        train_texts (list of str): Training set texts.
        val_texts (list of str): Validation set texts.
        test_texts (list of str): Test set texts.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for the transformer model.
        
    Returns:
        list: Relevant token IDs for the dataset vocabulary.
    """
    print("compute_dataset_vocab()...")

    # Combine all texts from training, validation, and test sets
    all_texts = train_texts + val_texts + test_texts

    # Use a set to store unique tokens from the dataset
    dataset_vocab_tokens = set()

    # Tokenize each document and add the tokens to the dataset vocabulary
    for text in all_texts:
        tokens = tokenizer.tokenize(text)  # Tokenize the text using the tokenizer
        dataset_vocab_tokens.update(tokens)  # Add tokens to the dataset vocabulary

    # Convert dataset tokens to their respective token IDs using the tokenizer's vocabulary
    relevant_tokens = [
        tokenizer.vocab[token]
        for token in dataset_vocab_tokens
        if token in tokenizer.vocab
    ]

    print(f"Computed dataset vocabulary: {len(relevant_tokens)} relevant tokens out of {len(tokenizer.vocab)} total tokens in tokenizer.")
    return relevant_tokens



def build_model(model_name, model_path, num_classes, class_type, lc_tokenizer, class_weights, args, tce_matrix=None, debug=False):
    """
    build classifier model based on the specified model_name and model_path
    
    Parameters:
        -- model_name: name of the model to use
        -- model_path: path to the model
        -- num_classes: number of classes for classification
        -- class_type: classification type, either 'single-label' or 'multi-label'
        -- lc_tokenizer: LCTokenizer object
        -- class_weights: class weights for loss function
        -- args: command line arguments
        -- tce_matrix: tensor of class embeddings
        -- debug: turn on debug mode

    Returns:
        -- lc_model: the model to use for classification
    """

    print(f"building classifier model... pretrained: {args.pretrained}, model_name: {model_name}, model_path: {model_path}, num_classes: {num_classes}, class_type: {class_type}, debug: {debug}")

    if args.net == 'cnn':
        
        if (debug):
            print("using CNN classifier...")
        
        if args.pretrained == 'bert':
            cnn_model = LCCNNBERTClassifier(model_name=model_name, 
                                            cache_dir=model_path, 
                                            num_classes=num_classes,
                                            class_type=class_type, 
                                            lc_tokenizer=lc_tokenizer, 
                                            num_channels=args.channels, 
                                            supervised=args.supervised, 
                                            tce_matrix=tce_matrix, 
                                            class_weights=class_weights,
                                            normalize_tces=args.normalize, 
                                            dropout_rate=args.dropprob, 
                                            comb_method=args.sup_mode, 
                                            debug=debug
                                            )
        elif args.pretrained == 'roberta':
            cnn_model = LCCNNRoBERTaClassifier(model_name=model_name, 
                                            cache_dir=model_path, 
                                            num_classes=num_classes,
                                            class_type=class_type, 
                                            lc_tokenizer=lc_tokenizer, 
                                            num_channels=args.channels, 
                                            supervised=args.supervised, 
                                            tce_matrix=tce_matrix, 
                                            class_weights=class_weights,
                                            normalize_tces=args.normalize, 
                                            dropout_rate=args.dropprob, 
                                            comb_method=args.sup_mode, 
                                            debug=debug
                                            )
        elif args.pretrained == 'distilbert':
            cnn_model = LCCNNDistilBERTClassifier(model_name=model_name, 
                                            cache_dir=model_path, 
                                            num_classes=num_classes,
                                            class_type=class_type, 
                                            lc_tokenizer=lc_tokenizer, 
                                            num_channels=args.channels, 
                                            supervised=args.supervised, 
                                            tce_matrix=tce_matrix, 
                                            class_weights=class_weights,
                                            normalize_tces=args.normalize, 
                                            dropout_rate=args.dropprob, 
                                            comb_method=args.sup_mode, 
                                            debug=debug
                                            )
        elif args.pretrained == 'xlnet':
            cnn_model = LCCNNXLNetClassifier(model_name=model_name, 
                                            cache_dir=model_path, 
                                            num_classes=num_classes,
                                            class_type=class_type, 
                                            lc_tokenizer=lc_tokenizer, 
                                            num_channels=args.channels, 
                                            supervised=args.supervised, 
                                            tce_matrix=tce_matrix, 
                                            class_weights=class_weights,
                                            normalize_tces=args.normalize, 
                                            dropout_rate=args.dropprob, 
                                            comb_method=args.sup_mode, 
                                            debug=debug
                                            )
        elif args.pretrained == 'gpt2':
            cnn_model = LCCNNGPT2Classifier(model_name=model_name, 
                                            cache_dir=model_path, 
                                            num_classes=num_classes,
                                            class_type=class_type, 
                                            lc_tokenizer=lc_tokenizer, 
                                            num_channels=args.channels, 
                                            supervised=args.supervised, 
                                            tce_matrix=tce_matrix, 
                                            class_weights=class_weights,
                                            normalize_tces=args.normalize, 
                                            dropout_rate=args.dropprob, 
                                            comb_method=args.sup_mode, 
                                            debug=debug
                                            )
        else:
            raise ValueError(f"Unsupported model class for net: {args.net} and pretrained model: {args.pretrained}")
        
        if (debug):
            print("\nCNN Classifier Model:\n", cnn_model)
        
        lc_model = cnn_model

    elif args.net == 'hf.sc':
        
        if (debug):
            print("using HuggingFace Sequence Classifier...")
        
        hf_trans_model, hf_trans_class_model = get_hf_models(
            model_name, 
            model_path, 
            num_classes=num_classes, 
            tokenizer=tokenizer
        )
        hf_trans_model.to(device)
        hf_trans_class_model.to(device)

        print("\nhf_trans_model:\n", hf_trans_model)
        print("\nhf_trans_class_model:\n", hf_trans_class_model)

        hf_trans_model = hf_trans_class_model
        print("\nHuggingFace Transformer Model:\n", hf_trans_model)

        #
        # note we instantiate only with relevant_tokens 
        # from our custom tokenizer (lc_tokenizer)
        #
        lc_model = LCSequenceClassifier(
            hf_model=hf_trans_model,                                    # HuggingFace transformer model being used
            num_classes=num_classes,                                    # number of classes for classification
            lc_tokenizer=lc_tokenizer,                                  # LCTokenizer (used for validation)
            class_type=class_type,                                      # classification type, options 'single-label' or 'multi-label'
            class_weights=class_weights,                                # class weights for loss function
            supervised=args.supervised,
            tce_matrix=tce_matrix,
            normalize_tces=args.normalize,                              # normalize TCE matrix using underlying embedding mean and std                     
            dropout_rate=args.dropprob,                                 # dropout rate for TCEs
            comb_method=args.sup_mode,                                  # combination method for TCEs with model embeddings, options 'cat', 'add', 'dot'
            #debug=True                                                 # turns on active forware debugging
        )

    else:
        raise ValueError(f"Unsupported neural model: {args.net}")
    
    return lc_model



# Parse arguments
def parse_args():

    parser = argparse.ArgumentParser(description="Transformer Layer Cake: Text Classification with Transformer Models.")
    
    # system params
    parser.add_argument('--dataset', required=True, type=str, choices=SUPPORTED_DATASETS, 
                        help='Dataset to use')
    parser.add_argument('--show-dist', action='store_true', default=True, 
                        help='Show dataset class distribution')
    #parser.add_argument('--dist', action='store_true', default=False, help='show class distribution plots')
    parser.add_argument('--vtype', type=str, default='tfidf', metavar='N', 
                        help=f'dataset base vectorization strategy, in [tfidf, count]')
    parser.add_argument('--pretrained', type=str, choices=['bert', 'roberta', 'distilbert', 'xlnet', 'gpt2'], 
                        help='supported language model types for dataset representation')
    parser.add_argument('--net', type=str, default='hf.sc', metavar='str', 
                        help=f'supported models, either hf.sc, linear, cnn, lstm or attn (defaults to hf.sc)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--log-file', type=str, default='../log/trans_lc_nn_test.test', 
                        help='Path to log file')
    parser.add_argument('--force', action='store_true', default=False, 
                        help='do not check if this experiment has already been run')
    parser.add_argument('--weighted', action='store_true', default=False, 
                        help='Whether or not to use class_weights in the loss funtion of the classifier (default False)')

    # model params
    parser.add_argument('--class-model', action='store_true', default=True,
                        help='use simple transformer classifier if True, else use the HF Sequence Classifier (customizede) model.')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, 
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=EPOCHS, 
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=PATIENCE, 
                        help='Patience for early stopping')
    parser.add_argument('--dropprob', type=float, default=0.5, metavar='[0.0, 1.0]', 
                        help='dropout probability for classifier head (default: 0.5)')
    parser.add_argument('--learnable', type=int, default=0, metavar='int', 
                        help='dimension of the learnable embeddings (default 0)')
    parser.add_argument('--droptype', type=str, default='sup', metavar='DROPTYPE',
                        help=f'chooses the type of dropout to apply after the embedding layer. Default is "sup" which '
                             f'only applies to word-class embeddings (if present). Other options include "none" which '
                             f'does not apply dropout (same as "sup" with no supervised embeddings), "full" which '
                             f'applies dropout to the entire embedding, or "learn" that applies dropout only to the '
                             f'learnable embedding.')
    """
    parser.add_argument('--tunable', action='store_true', default=False,
                        help='pretrained embeddings are tunable from the beginning (default False, i.e., static)')
    """
    parser.add_argument('--tunable', type=str, default=None, metavar='str',
                        help='whether or not to have model parameters (gradients) tunable. One of [classifier, embedding, none]. Default to None.')
    parser.add_argument('--channels', type=int, default=256, metavar='int',
                        help='number of cnn out-channels (default: 256)')
    """
    parser.add_argument('--emb-filter', action='store_true', default=True,
                        help='filter model embeddings to align with dataset vocabulary (default False)')
    """

    # TCE params
    parser.add_argument('--supervised', action='store_true', 
                        help='Use supervised embeddings (TCEs')
    parser.add_argument('--sup-mode', type=str, default='cat', 
                        help=f'How to combine TCEs with model embeddings (in {SUPPORTED_OPS})')
    parser.add_argument('--nozscore', action='store_true', default=True,
                        help='disables z-scoring form the computation of TCE')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='normalizes TCE matrix using underlying embedding mean and std. Default == True')
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
    parser.add_argument('--max-label-space', type=int, default=300, metavar='int',
                        help='larger dimension allowed for the feature-label embedding (if larger, then PCA with this '
                             'number of components is applied (default 300)')

    return parser.parse_args()



# Main
if __name__ == "__main__":

    program = 'trans_layer_cake'
    version = '13.1'

    print(f'\t--- TRANS_LAYER_CAKE Version: {version} ---')

    args = parse_args()
    print("args:", args)
    
    # Get the full model identifier and cache directory path for tokenizer/model
    model_name, model_path = get_model_identifier(args.pretrained)
    print("model_name:", model_name)
    print("model_path:", model_path)

    if (args.pretrained == 'bert'):
        args.bert_path = model_path
    elif (args.pretrained == 'roberta'):
        args.roberta_path = model_path
    elif (args.pretrained == 'distilbert'):
        args.distilbert_path = model_path
    elif (args.pretrained == 'xlnet'):
        args.xlnet_path = model_path
    elif (args.pretrained == 'gpt2'):
        args.gpt2_path = model_path
    else:
        raise ValueError("Unsupported pretrained model:", args.pretrained)
    
    print("args:", args)    

    # initialize logging and other system run variables
    already_modelled, logfile, method_name, pretrained, embeddings, embedding_type, emb_path, lm_type, mode, system = initialize_testing(args, program, version)

    # check to see if model params have been computed already
    if (already_modelled and not args.force):
        print(f'--- model {method_name} with embeddings {embeddings}, pretrained == {pretrained}, and wc_supervised == {args.supervised} for {args.dataset} already calculated, run with --force option to override. ---')
        exit(0)

    print("embedding_type:", embedding_type)
    print("embeddings:", embeddings)    
    print("embedding_path:", emb_path)
    print("lm_type:", lm_type)
    print("mode:", mode)
    print("system:", system)
    
    # Check for CUDA and MPS availability
    # Default to CPU if neither CUDA nor MPS is available
    if torch.cuda.is_available():
        device = torch.device("cuda")

        if (args.dataset == 'rcv1'):
            batch_size = DEFAULT_MIN_CUDA_BATCH_SIZE
        else:
            batch_size = DEFAULT_MAX_CUDA_BATCH_SIZE

        # for eval_step calculation
        num_devices = system.get_num_gpus()

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

        if (args.dataset == 'rcv1'):
            batch_size = DEFAULT_MIN_MPS_BATCH_SIZE
        else:
            batch_size = DEFAULT_MAX_MPS_BATCH_SIZE

        # for eval_step calculation
        num_devices = 1                                # MPS only supports a single GPU
    else:
        if (args.dataset == 'rcv1'):
            batch_size = DEFAULT_MIN_CPU_BATCH_SIZE
        else:
            batch_size = DEFAULT_MAX_CPU_BATCH_SIZE     
        
        # for eval_step calculation
        num_devices = 1                                # No GPUs available (CPU), so we default to 1

    print(f"device: {device}")
    print("num_devices:", num_devices)
    print("batch_size:", batch_size)

    torch.manual_seed(args.seed)

    print(f'\n\t--- Loading dataset {args.dataset} data ---')

    #
    # Load dataset training and test data, associated labels, 
    # as well as number of classes (labels) and the target names of 
    # the dataset 
    #
    train_data, train_target, test_data, labels_test, num_classes, target_names, class_type = get_dataset_data(args.dataset, args.seed)

    print("class_type:", class_type)
    print("num_classes:", num_classes)
    print("target_names:", target_names)

    print("train_data:", type(train_data), len(train_data))
    print("train_data[0]:", type(train_data[0]), train_data[0])
    print("train_target:", type(train_target), len(train_target))
    print("train_target[0]:", type(train_target[0]), train_target[0].shape, train_target[0])

    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), len(labels_test))
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])

    # Check for empty strings in train_data and test_data
    check_empty_docs(train_data, "Train")
    check_empty_docs(test_data, "Test")

    lc_tokenizer = LCTokenizer(
        model_name=model_name,
        model_path=model_path,
        lowercase=True,                                         # should align with the way dataset is (pre) processed
        remove_special_tokens=False,
        padding='max_length',
        truncation=True
    )

    tokenizer = lc_tokenizer.tokenizer
    print("LCTokenizer tokenizer configuration:")
    print(f'  tokenizer: {type(tokenizer)}, {tokenizer}')
    print(f"  Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  Max length: {lc_tokenizer.max_length}")

    # Split off validation data from the training set
    texts_train, texts_val, labels_train, labels_val = train_test_split(
                                                            train_data, 
                                                            train_target, 
                                                            test_size=VAL_SIZE, 
                                                            shuffle=True,
                                                            random_state=RANDOM_SEED
                                                            )

    print("texts_train:", type(texts_train), len(texts_train))
    print("texts_train[0]:", type(texts_train[0]), texts_train[0])
    print("labels_train:", type(labels_train), len(labels_train))
    print("labels_train[0]:", type(labels_train[0]), labels_train[0].shape, labels_train[0])

    print("texts_val:", type(texts_val), len(texts_val))
    print("texts_val[0]:", type(texts_val[0]), texts_val[0])
    print("labels_val:", type(labels_val), len(labels_val))
    print("labels_val[0]:", type(labels_val[0]), labels_val[0].shape, labels_val[0])

    print("test_data:", type(test_data), len(test_data))
    print("test_data[0]:", type(test_data[0]), test_data[0])
    print("labels_test:", type(labels_test), len(labels_test))
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])

    #
    # Vectorize the text data
    #
    print("\n\t vectorizing dataset...")

    vectorizer, Xtr, Xval, Xte = get_vectorized_data(
        texts_train,
        texts_val,
        test_data,
        lc_tokenizer,
        args.dataset,
        args.pretrained,
        args.vtype,
        debug=True
    )

    print("vectorizer:\n", vectorizer)

    print("Xtr:", type(Xtr), Xtr.shape)
    #print("Xtr[0]:", type(Xtr[0]), Xtr[0].shape, Xtr[0].toarray().flatten())
    #print("Xtr[1]:", type(Xtr[1]), Xtr[1].shape, Xtr[1])
    #print("Xtr[1]:", type(Xtr[2]), Xtr[2].shape, Xtr[2])

    print("Xval:", type(Xval), Xval.shape)
    #print("Xval[0]:", type(Xval[0]), Xval[0].shape, Xval[0])
    
    print("Xte:", type(Xte), Xte.shape)
    #print("Xte[0]:", type(Xte[0]), Xte[0].shape, Xte[0])

    # convert single label y values from array of scalaers to one hot encoded
    print("vectorizer.vocabulary_:", type(vectorizer.vocabulary_), len(vectorizer.vocabulary_))
    vec_vocab_size = len(vectorizer.vocabulary_)
    print("vec_vocab_size:", vec_vocab_size)
    tok_vocab_size = len(tokenizer.get_vocab())
    print("tok_vocab_size:", tok_vocab_size)

    #
    # validate that the vectorize and tokenizer vocabularies are mirrors of each other
    # this is imperative for the TCE matrix computation alignment with the (pretrained) 
    # model embeddings
    #
    assert set(vectorizer.vocabulary_.keys()).issubset(tokenizer.get_vocab().keys()), "Vectorizer vocabulary must be a subset of tokenizer vocabulary"

    # Assertion: Ensure vectorizer vocabulary size matches tokenizer vocabulary size
    assert vec_vocab_size == tok_vocab_size, \
        f"Vectorizer vocab size ({vec_vocab_size}) must equal tokenizer vocab size ({tok_vocab_size})"

    tok_vocab = set(tokenizer.get_vocab().keys())
    vec_vocab = set(vectorizer.vocabulary_.keys())
    print(f"Tokenizer vocab size: {len(tok_vocab)}, Vectorizer vocab size: {len(vec_vocab)}")
    print(f"Vocabulary intersection size: {len(tok_vocab & vec_vocab)}")

    #
    # spot check the vectorized data to make sure the vectorizer 
    # and tokenizer are in sync, this is crucial for TCE computation
    #
    spot_check_documents(
        documents=texts_train,
        vectorizer=vectorizer,
        lc_tokenizer=lc_tokenizer,                # we pass in the LCTokenizer 
        vectorized_data=Xtr,       
        num_docs=3,
        debug=True
    )    

    # -----------------------------------------------------------------------------------------------
    #
    # embedding filtering code, not working as expected but leaving here for future reference
    #
    # set relevant_tokens to None if we are not filtering embedding layer, otherwise filter the
    # tokens in the LCTokenizer class to be fed into LCSequenceClassifier to update (filter) the
    # nn.Embedding layer
    #
    # NB this is not working, the Classifier does not like this filtered Embedding layer for some reason
    #
    """
    if not args.emb_filter:
        print('not filtering embeddings...')
        relevant_token_ids = None
    else:
        print('\n\tfiltering embeddings...')

        all_docs = texts_train + texts_val + test_data                                             # accumulate all of the docs together
        relevant_tokens, relevant_token_ids, mismatches = lc_tokenizer.filter_tokens(
                                                                    texts=all_docs, 
                                                                    dataset_name=args.dataset
                                                                    )
        
        print("relevant_tokens:", type(relevant_tokens), len(relevant_tokens))
        print("relevant_token_ids:", type(relevant_token_ids), len(relevant_token_ids))
        print("mismatches:", type(mismatches), len(mismatches))

        # Get the size of the dataset-specific vocabulary
        filtered_vocab_size = len(relevant_tokens)
        print(f"Relevant tokens size: {filtered_vocab_size}")

        #
        # TODO: may need to update the vocab_size variable here for downstream processing (eg. TCE computation)
        #
    """
    # -----------------------------------------------------------------------------------------------


    # -----------------------------------------------------------------------------------------------
    # 
    # compute supervised embeddings if need be by calling compute_supervised_embeddings
    # if args.supervised is True
    #
    if (args.supervised):

        print("\n\tcomputing TCEs...")

        if (class_type in ['single-label', 'singlelabel']):

            print("single label, converting target labels to to one-hot for tce computation...")
            
            # Assuming labels are a numpy array of shape (num_samples,)
            def to_one_hot(labels, num_classes):
                encoder = OneHotEncoder(categories='auto', sparse_output=True, dtype=np.float32)
                labels = labels.reshape(-1, 1)
                one_hot = encoder.fit_transform(labels)  # Result is a sparse matrix
                return one_hot

            one_hot_labels_train = to_one_hot(labels_train, num_classes=num_classes)

        else:
            one_hot_labels_train = labels_train

        # Ensure one_hot_labels_train is a csr_matrix
        if not isinstance(one_hot_labels_train, csr_matrix):
            print("Converting one_hot_labels_train to csr_matrix...")
            one_hot_labels_train = csr_matrix(one_hot_labels_train)
            
        tce_matrix = compute_tces(
            vocabsize=vec_vocab_size,
            vectorized_training_data=Xtr,
            training_label_matrix=one_hot_labels_train,
            opt=args,
            debug=True
        )
        tce_matrix.to(device)           # must move the TCE embeddings to same device as model

        print("tce_matrix:", type(tce_matrix), tce_matrix.shape)
        print("tce_matrix[0]:", type(tce_matrix[0]), tce_matrix[0].shape, tce_matrix[0])

        if torch.isnan(tce_matrix).any() or torch.isinf(tce_matrix).any():
            print("[WARNING}: tce_matrix contains NaN or Inf values during initialization.")
            #raise ValueError("[ERROR] tce_matrix contains NaN or Inf values during initialization.")
    else:
        tce_matrix = None
    # -----------------------------------------------------------------------------------------------


    #
    # if specified, show the cl;ass distribution
    # especially helpful for multi-label datasets
    # where the class is unevenly distributed and hence
    # affects micro-f1 scores out of testing (with smaller 
    # samples where under represented classes in training 
    # are further underrepresented in the test dataset)
    #
    if (args.show_dist):

        train_cls_wghts = show_class_distribution(
            labels=labels_train, 
            target_names=target_names, 
            class_type=class_type, 
            dataset_name=args.dataset+':train'
            )
        
        """
        val_cls_wghts = show_class_distribution(
            labels=labels_val, 
            target_names=target_names, 
            class_type=class_type, 
            dataset_name=args.dataset+':val'
            )
        
        test_cls_wghts = show_class_distribution(
            labels=labels_test, 
            target_names=target_names, 
            class_type=class_type, 
            dataset_name=args.dataset+':test'
            )
        """

    if (args.weighted):
        print("computing class weights for classifier loss function...")
        class_weights = lc_class_weights(labels_train, task_type=class_type)
    else:
        print("--- not using class weights... ---")
        class_weights = None

    
    print("\n\t building model...")

    lc_model = build_model(
        model_name=model_name,
        model_path=model_path,
        num_classes=num_classes,
        class_type=class_type,
        lc_tokenizer=lc_tokenizer,
        class_weights=class_weights,
        args=args,
        tce_matrix=tce_matrix,
        #debug=True
        )

    #lc_model.xavier_uniform()
    lc_model = lc_model.to(device)
    if args.tunable == 'pretrained':
        lc_model.finetune_pretrained()
    elif args.tunable == 'classifier':
        lc_model.finetune_classifier()
    else:
        print(f"both classifier and embedding layers are not tunable...")

    if (args.supervised):
        print("validating tce alignment...")
        lc_model.validate_tce_alignment()

    print("\n\t-- Final Model --:\n", lc_model)

    # Get embedding size from the model
    dimensions, vec_size = lc_model.get_embedding_dims()
    #print(f'dimensions: {dimensions}, vec_size: {vec_size}')
    dimensions = f'({dimensions},{vec_size})'
    # Concatenate supervised-specific dimensions if args.supervised is True
    if args.supervised:
        # Convert tce_matrix.shape to string before concatenating
        dimensions = f"{dimensions}:{str(tce_matrix.shape)}"
    # Log the dimensions
    print("dimensions:", dimensions)

    # Prepare datasets
    train_dataset = LCDataset(
        texts_train, 
        labels_train, 
        tokenizer, 
        max_length=lc_tokenizer.max_length,
        class_type=class_type 
    )

    val_dataset = LCDataset(
        texts_val, 
        labels_val, 
        tokenizer, 
        max_length=lc_tokenizer.max_length,
        class_type=class_type
    )

    test_dataset = LCDataset(
        test_data, 
        labels_test, 
        tokenizer, 
        max_length=lc_tokenizer.max_length,
        class_type=class_type
    )

    tinit = time()

    training_args_opt = TrainingArguments(
        output_dir='../out',
        evaluation_strategy="epoch",                        
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir='../log',
        run_name='trans_layer_cake',
        seed=args.seed,
        report_to="none"
    )

    # Trainer with custom data collator
    trainer = Trainer(
        model=lc_model,                                                                                                 # use custom model 
        args=training_args_opt,                                                                                         # TrainingArguments
        train_dataset=train_dataset,                                                                                    # Training dataset
        eval_dataset=val_dataset,                                                                                       # Evaluation dataset
        data_collator=custom_data_collator,                                                                             # Custom data collator
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, class_type, threshold=MC_THRESHOLD),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    print("\n\tbuilding model...")

    # Enable anomaly detection during training
    with torch.autograd.set_detect_anomaly(True):
        try:
            model_deets = trainer.train()
            print("model_deets\n:", model_deets)
        except RuntimeError as e:
            print(f"An error occurred during training: {e}")
            
    final_loss = model_deets.training_loss
    print("final_loss:", final_loss)

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:", test_results)

    # Predictions
    preds = trainer.predict(test_dataset)
    if class_type == 'single-label':
        y_pred = np.argmax(preds.predictions, axis=1)
    else:
        y_pred = (preds.predictions > 0.5).astype(int)
    
    """
    print("labels_test:", type(labels_test), labels_test.shape)
    print("labels_test[0]:", type(labels_test[0]), labels_test[0].shape, labels_test[0])
    print("y_pred:", type(y_pred), y_pred.shape)
    print("y_pred[0]:", type(y_pred[0]), y_pred[0].shape, y_pred[0])
    """

    print(classification_report(labels_test, y_pred, target_names=target_names, digits=4))

    macrof1, microf1, acc, h_loss, precision, recall, j_index = evaluation_nn(labels_test, y_pred, classification_type=class_type)
    print("\n--Layer Cake Metrics--")
    print(f"Macro-F1 = {macrof1:.4f}, Micro-F1 = {microf1:.4f}, Accuracy = {acc:.4f}, H-loss = {h_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, Jaccard = {j_index:.4f}")
    
    tend = time() - tinit

    measure_prefix = 'final-te'
    #epoch = trainer.state.epoch
    epoch = int(round(trainer.state.epoch))
    print("epoch:", epoch)

    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-macro-f1', value=macrof1, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-micro-f1', value=microf1, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-loss', value=final_loss, timelapse=tend)

    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-hamming-loss', value=h_loss, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-precision', value=precision, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-recall', value=recall, timelapse=tend)
    logfile.insert(dimensions=dimensions, epoch=epoch, measure=f'{measure_prefix}-jacard-index', value=j_index, timelapse=tend)

    print("\n\t--- model training and evaluation complete---\n")


