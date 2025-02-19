from time import time
import numpy as np
from src.model.deprecated.layers import *
from transformers import BertModel, BertTokenizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import make_scorer, recall_score, hamming_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

from util.metrics import evaluation_ml
from util.common import OUT_DIR

import logging
logging.basicConfig(level=logging.INFO)


NUM_JOBS = -1                   # important to manage CUDA memory allocation
#NUM_JOBS = 40                  # for rcv1 dataset which has 101 classes, too many to support in parallel

NUM_SAMPLED_PARAMS = 30         # Number of parameter settings that are sampled by RandomizedSearchCV

#
# Neural Models
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

        # Initialize the custom embedding layer with pre-trained or learnable embeddings.
        # This will combine pretrained and learnable embeddings.
        self.embed = EmbeddingCustom(vocab_size, learnable_length, pretrained, drop_embedding_range, drop_embedding_prop)

        print("self.embed:\n", self.embed)
        print("self.embed.dim():", self.embed.dim())
        
        print("pt dimensions:", self.embed.get_pt_dimensions())
        print("lrn dimensions:", self.embed.get_lrn_dimensions())

        # Initialize the projection layer (CNN, LSTM, or Attention) based on the net_type.
        self.projection = init__projection(net_type)(self.embed.dim(), hidden_size)

        # Linear layer to map the document embedding to output size (number of classes).
        self.label = nn.Linear(self.projection.dim(), output_size)

        self.vocab_size = vocab_size

    def get_embedding_size(self):
        return self.embed.get_pt_dimensions()

    def get_learnable_embedding_size(self):
        return self.embed.get_lrn_dimensions()
    
    def forward(self, input):
        """
        Expected input:
        - input: A list or tensor of tokenized words/subwords.
                 - If using pre-trained embeddings, the input should be a tensor of token indices.
                 - If using a tokenizer (e.g., BERT), input should be a list of tokens to be passed through the tokenizer.
                 
        Example:
        - For BERT: [["hello", "world"], ["example", "sentence"]]
        - For word embeddings: [[1, 2, 3], [4, 5, 6]]  (numerical indices)
        
        Returns:
        - logits: A tensor of shape [batch_size, output_size] representing class scores.
        """
        # Get word embeddings from the input tokens.
        word_emb = self.embed(input)                

        # Project word embeddings into document embeddings using CNN, LSTM, or Attention.
        doc_emb = self.projection(word_emb)

        # Get the logits (class scores) by passing document embeddings through the linear layer.
        logits = self.label(doc_emb)
        
        return logits

    def finetune_pretrained(self):
        self.embed.finetune_pretrained()

    def xavier_uniform(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)


class Token2BertEmbeddings:
    def __init__(self, pretrained_model_name='bert-base-uncased', max_length=500, device='cuda'):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.model = BertModel.from_pretrained(pretrained_model_name).eval().to(device)
        self.max_length = max_length
        self.device = device

    def embeddings(self, tokens):
        max_length = min(self.max_length, max(map(len,tokens)))  # for dynamic padding
        cls_t = self.tokenizer.cls_token
        sep_t = self.tokenizer.sep_token
        pad_idx = self.tokenizer.pad_token_id
        tokens = [[cls_t] + d[:max_length] + [sep_t] for d in tokens]
        index = [
            self.tokenizer.convert_tokens_to_ids(doc) + [pad_idx] * (max_length - (len(doc)-2)) for doc in
            tokens
        ]
        #index = [
        #    self.tokenizer.encode(d, add_special_tokens=True, max_length=max_length+2, pad_to_max_length=True)
        #    for d in docs
        #]
        index = torch.tensor(index).to(self.device)

        with torch.no_grad():
            outputs = self.model(index)
            contextualized_embeddings = outputs[0]
            # ignore embeddings for [CLS] and las one (either [SEP] or last [PAD])
            contextualized_embeddings = contextualized_embeddings[:,1:-1,:]
            return contextualized_embeddings

    def dim(self):
        return 768


class Token2WCEmbeddings(nn.Module):
    def __init__(self, WCE, WCE_range, WCE_vocab, drop_embedding_prop=0.5, max_length=500, device='cuda'):
        super(Token2WCEmbeddings, self).__init__()
        assert '[PAD]' in WCE_vocab, 'unknown index for special token [PAD] in WCE vocabulary'
        self.embed = EmbeddingCustom(len(WCE_vocab), 0, WCE, WCE_range, drop_embedding_prop).to(device)
        self.max_length = max_length
        self.device = device
        self.vocab = WCE_vocab
        self.pad_idx = self.vocab['[PAD]']
        self.unk_idx = self.vocab['[UNK]']

    def forward(self, tokens):
        max_length = min(self.max_length, max(map(len,tokens)))  # for dynamic padding
        tokens = [d[:max_length] for d in tokens]
        index = [
            [self.vocab.get(ti, self.unk_idx) for ti in doc] + [self.pad_idx]*(max_length - len(doc)) for doc in tokens
        ]
        index = torch.tensor(index).to(self.device)
        return self.embed(index)

    def dim(self):
        return self.embed.dim()

    def finetune_pretrained(self):
        self.embed.finetune_pretrained()


class BertWCEClassifier(nn.Module):
    ALLOWED_NETS = {'cnn', 'lstm', 'attn'}

    def __init__(self,
                 net_type,
                 output_size,
                 hidden_size,
                 token2bert_embeddings,
                 token2wce_embeddings):
        super(BertWCEClassifier, self).__init__()

        # Compute the total embedding dimension by combining BERT and WCE embeddings
        emb_dim = token2bert_embeddings.dim() + (0 if token2wce_embeddings is None else token2wce_embeddings.dim())
        print(f'Embedding dimensions {emb_dim}')

        # BERT embeddings object
        self.token2bert_embeddings = token2bert_embeddings

        # Word Context Embeddings (WCE) object (optional)
        self.token2wce_embeddings = token2wce_embeddings

        # Projection layer (CNN, LSTM, or Attention) initialized based on the net type
        self.projection = init__projection(net_type)(emb_dim, hidden_size)

        # Linear layer to map document embeddings to output class scores
        self.label = nn.Linear(self.projection.dim(), output_size)


    def forward(self, input): # list of lists of tokens
        """
        Expected input:
        - input: A list of lists of tokenized words/subwords.
                 - Each list represents a document or sentence.
                 - Tokens should be pre-processed and tokenized before passing to this method.
                 - Example: [["hello", "world"], ["example", "sentence"]] (for raw tokens)

        Returns:
        - logits: A tensor of shape [batch_size, output_size] representing class scores.
        """

        # BERT Embeddings: convert tokens to BERT token ids, apply padding, and get BERT embeddings
        contextualized_embeddings = self.token2bert_embeddings.embeddings(input)

        # WCE Embeddings: If WCE is used, get the embeddings for each token
        # convert tokens to ids for WCE, pad, and get WCEs
        if self.token2wce_embeddings is not None:
            wce_embeddings = self.token2wce_embeddings(input)
            # concatenate Bert embeddings with WCEs
            assert contextualized_embeddings.shape[1] == wce_embeddings.shape[1], 'shape mismatch between Bert and WCE'
            word_emb = torch.cat([contextualized_embeddings, wce_embeddings], dim=-1)
        else:
            word_emb = contextualized_embeddings

        # Project the concatenated embeddings into a document-level embedding
        doc_emb = self.projection(word_emb)

        # Get class logits from the document embeddings
        logits = self.label(doc_emb)

        return logits



    def finetune_pretrained(self):
        self.token2wce_embeddings.finetune_pretrained()

    def xavier_uniform(self):
        for model in [self.token2wce_embeddings, self.projection, self.label]:
            if model is None: continue
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





#
# ML Models
#

from scipy.sparse import issparse

# -------------------------------------------------------------------------------------------------------------------------------------------------
# ml_classification()
# -------------------------------------------------------------------------------------------------------------------------------------------------
def ml_classification(X_train, X_test, y_train, y_test, args, target_names, class_type='singlelabel'):
    
    print("\n\tML classification...")

    print('X_train:', type(X_train), X_train.shape)
    print('X_test:', type(X_test), X_test.shape)

    print('y_train:', type(y_train), y_train.shape)
    print('y_test:', type(y_test), y_test.shape)

    if issparse(X_train):

        print("Converting sparse X matrices to dense arrays...")
        X_train = X_train.toarray()
        X_test = X_test.toarray()

        print('X_train:', type(X_train), X_train.shape)
        print('X_test:', type(X_test), X_test.shape)

    if (issparse(y_train)):
        print("Converting sparse y matrices to dense arrays...")
        y_train = y_train.toarray()
        y_test = y_test.toarray()

        print('y_train:', type(y_train), y_train.shape)
        print('y_test:', type(y_test), y_test.shape)
        
    #print("y_train:", y_train)
    #print("y_test:", y_test)
        
    print("target_names:", target_names)
    print("class_type:", class_type)

    #
    # if y matrices are one-hot encoded, convert them to class labels (one dimension with class value)
    #
    if (class_type in ['singlelabel', 'single-label']):
        
        # If y_train is one-hot encoded, convert it to class labels first
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train = convert_labels(y_train)

        #print("y_train:", y_train)

        # If y_train is one-hot encoded, convert it to class labels first
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = convert_labels(y_test)

        #print("y_test:", y_test)

    tinit = time()

    # Support Vector Machine Classifier
    if (args.learner == 'svm'):                                     
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = run_svm_model(
            args.dataset,
            X_train,
            X_test,
            y_train,
            y_test,
            args,
            target_names,
            class_type=class_type
            )
    
    # Logistic Regression Classifier
    elif (args.learner == 'lr'):                                  
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = run_lr_model(
            args.dataset,
            X_train,
            X_test,
            y_train,
            y_test,
            args,
            target_names,
            class_type=class_type
            )

    # Naive Bayes (MultinomialNB) Classifier
    elif (args.learner == 'nb'):                                  
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = run_nb_model(
            args.dataset,
            X_train,
            X_test,
            y_train,
            y_test,
            args,
            target_names,
            class_type=class_type
            )
    
    else:
        print(f"Invalid learner '{args.learner}'")
        return None

    formatted_string = f'Macro F1: {Mf1:.4f} Micro F1: {mf1:.4f} Acc: {accuracy:.4f} Hamming Loss: {h_loss:.4f} Precision: {precision:.4f} Recall: {recall:.4f} Jaccard Index: {j_index:.4f}'
    print(formatted_string)

    tend = time() - tinit

    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index, tend



from sklearn.preprocessing import LabelEncoder

def convert_labels(y):

    # convert one hot encoded labels to single label
    # Initialize the label encoder
    label_encoder = LabelEncoder()

    y = np.argmax(y, axis=1)
    
    # Fit the encoder on y and transform it
    y_encoded = label_encoder.fit_transform(y)

    return y_encoded


# ---------------------------------------------------------------------------------------------------------------------
# run_svm_model()
# ---------------------------------------------------------------------------------------------------------------------
def run_svm_model(dataset, X_train, X_test, y_train, y_test, args, target_names, class_type='singlelabel'):
    
    print("\n\trunning SVM model...")

    # Check if it's a multilabel problem, and use OneVsRestClassifier if true
    if class_type in ['multilabel', 'multi-label']:
        print("Multilabel classification detected. Using OneVsRestClassifier...")
        
        classifier = OneVsRestClassifier(
            estimator=LinearSVC(class_weight='balanced', dual='auto', max_iter=1000),
            n_jobs=NUM_JOBS                         # parallelize the training
        )
    else:
        print("Single-label classification detected. Using regular SVM...")

        classifier = LinearSVC(class_weight='balanced', dual='auto', max_iter=1000)

    if not args.optimc:

        print("Running default SVM model params using LinearSVC and OneVsRestClassifier...")

        svc = LinearSVC(class_weight='balanced', dual='auto', max_iter=1000)
        
        ovr_svc = OneVsRestClassifier(
            estimator=svc, 
            n_jobs=NUM_JOBS
        )
        ovr_svc.fit(X_train, y_train)

        y_pred_default = ovr_svc.predict(X_test)

        print(classification_report(y_true=y_test, y_pred=y_pred_default, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_default

    # Case with optimization using RandomizedSearchCV
    else:

        print("Optimizing SVM model with RandomizedSearchCV...")
        
        param_distributions = {
            'estimator__penalty': ['l1', 'l2'],
            'estimator__loss': ['hinge', 'squared_hinge'],
            'estimator__C': np.logspace(-3, 3, 7)
        } if class_type == 'multilabel' else {
            'penalty': ['l1', 'l2'],
            'loss': ['hinge', 'squared_hinge'],
            'C': np.logspace(-3, 3, 7)
        }

        # Add zero_division=0 to precision and recall to suppress the warnings
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro', zero_division=0),
            'precision_score': make_scorer(precision_score, average='micro', zero_division=0),
            'hamming_loss': make_scorer(hamming_loss),
        }

        print("estimator:", classifier)
        print("param_distributions:", param_distributions)

        # Wrap RandomizedSearchCV around OneVsRestClassifier if multilabel
        randomized_search = RandomizedSearchCV(
            estimator=classifier,
            param_distributions=param_distributions,
            scoring=scorers,
            refit='f1_score',
            n_jobs=NUM_JOBS,
            cv=5,
            return_train_score=True,
            n_iter=NUM_SAMPLED_PARAMS                       # Number of parameter settings sampled
        )

        randomized_search.fit(X_train, y_train)
        
        print('Best parameters:', randomized_search.best_params_)
        best_model = randomized_search.best_estimator_
        
        # Predict on test set
        y_pred_best = best_model.predict(X_test)

        print(classification_report(y_true=y_test, y_pred=y_pred_best, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_best

    # Evaluate the model
    Mf1, mf1, accuracy, h_loss, precision, recall, j_index =    \
        evaluation_ml(y_test, y_preds, classification_type=class_type, debug=False)
    
    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index



# ---------------------------------------------------------------------------------------------------------------------
# run_lr_model()
# ---------------------------------------------------------------------------------------------------------------------
def run_lr_model(dataset, X_train, X_test, y_train, y_test, args, target_names, class_type='singlelabel'):

    print("\n\tRunning Logistic Regression model...")

    print("X_train:", type(X_train), X_train.shape)
    print("X_test:", type(X_test), X_test.shape)
    print("y_train:", type(y_train), y_train.shape)
    print("y_test:", type(y_test), y_test.shape)
    print("Target Names:", target_names)

    # Check if it's a multilabel problem, and use OneVsRestClassifier if true
    if class_type in ['multilabel', 'multi-label']:
        print("Multilabel classification detected. Using OneVsRestClassifier...")
        
        classifier = OneVsRestClassifier(
            estimator=LogisticRegression(max_iter=1000, class_weight='balanced', dual=False),
            n_jobs=NUM_JOBS
        )
    else:
        print("Single-label classification detected. Using regular Logistic Regression...")
        classifier = LogisticRegression(max_iter=1000, class_weight='balanced', dual=False)

    if not args.optimc:
        print("Running default Logistic Regression model using OneVsRestClassifier...")
        
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', dual=False)
        
        ovr_lr = OneVsRestClassifier(
            estimator=lr,
            n_jobs=NUM_JOBS
        )
        
        ovr_lr.fit(X_train, y_train)
        y_pred_default = ovr_lr.predict(X_test)

        print(classification_report(y_true=y_test, y_pred=y_pred_default, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_default

    # Case with optimization using GridSearchCV
    else:
        print("Optimizing Logistic Regression model with RandomizedSearchCV...")

        param_distributions = {
            'estimator__C': np.logspace(-3, 3, 7),                      # Regularization strength
            'estimator__penalty': ['l1', 'l2'],                         # Regularization method
            'estimator__solver': ['liblinear', 'saga']                  # Solvers compatible with L1 and L2 regularization
        } if class_type == 'multilabel' else {
            'C': np.logspace(-3, 3, 7),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        # Add zero_division=0 to precision and recall to suppress the warnings
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro', zero_division=0),
            'precision_score': make_scorer(precision_score, average='micro', zero_division=0),
            'hamming_loss': make_scorer(hamming_loss),
        }

        print("estimator:", classifier)
        print("param_distributions:", param_distributions)

        # Wrap RandomizedSearchCV around OneVsRestClassifier if multilabel
        randomized_search = RandomizedSearchCV(
            estimator=classifier,
            param_distributions=param_distributions,
            scoring=scorers,
            refit='f1_score',
            n_jobs=NUM_JOBS,
            cv=5,
            return_train_score=True,
            n_iter=NUM_SAMPLED_PARAMS                       # Number of parameter settings sampled
        )

        randomized_search.fit(X_train, y_train)
        
        print('Best parameters:', randomized_search.best_params_)
        best_model = randomized_search.best_estimator_
        
        # Predict on test set
        y_pred_best = best_model.predict(X_test)

        print(classification_report(y_true=y_test, y_pred=y_pred_best, target_names=target_names, digits=4, zero_division=0))
        y_preds = y_pred_best

    # Evaluate the model
    Mf1, mf1, accuracy, h_loss, precision, recall, j_index =    \
        evaluation_ml(y_test, y_preds, classification_type=class_type, debug=False)
    
    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index



# ---------------------------------------------------------------------------------------------------------------------
# run_nb_model()
# ---------------------------------------------------------------------------------------------------------------------
def run_nb_model(dataset, X_train, X_test, y_train, y_test, args, target_names, class_type='singlelabel'):
    
    print("\n\trunning Naive Bayes model...")

    # Check if it's a multilabel problem, and use OneVsRestClassifier if true
    if class_type in ['multilabel', 'multi-label']:
        print("Multilabel classification detected. Using OneVsRestClassifier...")
        
        classifier = OneVsRestClassifier(
            estimator=MultinomialNB(),
            n_jobs=NUM_JOBS
        )
    else:
        print("Single-label classification detected. Using regular Naive Bayes...")
        
        classifier = MultinomialNB()

    if not args.optimc:

        print("Running default Naive Bayes model using MultinomialNB and OneVsRestClassifier...")

        nb = MultinomialNB()
        ovr_nb = OneVsRestClassifier(estimator=nb, n_jobs=NUM_JOBS) if class_type in ['multilabel', 'multi-label'] else nb
        ovr_nb.fit(X_train, y_train)
        
        y_pred_default = ovr_nb.predict(X_test)

        print(classification_report(y_true=y_test, y_pred=y_pred_default, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_default

    # Case with optimization using GridSearchCV
    else:
        print("Optimizing Naive Bayes model with RandomizedSearchCV...")

        param_distributions = {
            'estimator__alpha': [0.1, 0.5, 1.0, 2.0]  # Smoothing parameter for MultinomialNB
        } if class_type == 'multilabel' else {
            'alpha': [0.1, 0.5, 1.0, 2.0]
        }

        # Add zero_division=0 to precision and recall to suppress the warnings
        scorers = {
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average='micro'),
            'recall_score': make_scorer(recall_score, average='micro', zero_division=0),
            'precision_score': make_scorer(precision_score, average='micro', zero_division=0),
            'hamming_loss': make_scorer(hamming_loss),
        }

        print("estimator:", classifier)
        print("param_distributions:", param_distributions)
        
        # Wrap RandomizedSearchCV around OneVsRestClassifier if multilabel
        randomized_search = RandomizedSearchCV(
            estimator=classifier,
            param_distributions=param_distributions,
            scoring=scorers,
            refit='f1_score',
            n_jobs=NUM_JOBS,
            cv=5,
            return_train_score=True,
            n_iter=NUM_SAMPLED_PARAMS                   # Number of parameter settings sampled
        )

        randomized_search.fit(X_train, y_train)
        
        print('Best parameters:', randomized_search.best_params_)
        best_model = randomized_search.best_estimator_
        
        # Predict on test set
        y_pred_best = best_model.predict(X_test)

        print(classification_report(y_true=y_test, y_pred=y_pred_best, target_names=target_names, digits=4, zero_division=0))

        y_preds = y_pred_best

    # Evaluate the model
    Mf1, mf1, accuracy, h_loss, precision, recall, j_index =    \
        evaluation_ml(y_test, y_preds, classification_type=class_type, debug=False)
    
    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index





def create_confusion_matrix(y_test, y_pred, category_names, title, file_name=OUT_DIR+'confusion_matrix.png', debug=True):
    """
    Create and display a confusion matrix with actual category names. NB only works with single label datasets
    
    Args:
    y_test (array-like): Ground truth (actual labels).
    y_pred (array-like): Predicted labels by the model.
    category_names (list): List of actual category names.
    title (str): Title of the plot.
    file_name (str): File name to save the confusion matrix.
    debug (bool): If True, will print additional information for debugging purposes.
    """

    print("Creating confusion matrix...")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix with category names on the axes
    fig, ax = plt.subplots(figsize=(12, 8))  # Set figure size

    # Display confusion matrix as a heatmap
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=category_names, yticklabels=category_names, ax=ax)

    # Set axis labels and title
    ax.set_xlabel('Predicted Categories', fontsize=14)
    ax.set_ylabel('Actual Categories', fontsize=14)
    plt.title(title, fontsize=16, pad=20)

    # Adjust layout and save the plot to a file
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight')  # Save to file
    plt.show()  # Display plot

    print(f"Confusion matrix saved as {file_name}")

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%")

    # Optionally print more detailed information
    if debug:
        print("\nConfusion Matrix Debug Information:")
        print("------------------------------------------------------")
        print("Confusion matrix shows actual classes as rows and predicted classes as columns.")
        print("\nConfusion Matrix Values:")
        for i in range(len(conf_matrix)):
            print(f"Actual category '{category_names[i]}':")
            for j in range(len(conf_matrix[i])):
                print(f"  Predicted as '{category_names[j]}': {conf_matrix[i][j]}")


