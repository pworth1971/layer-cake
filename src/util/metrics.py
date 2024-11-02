
from scipy.sparse import lil_matrix, issparse
from scipy.sparse import csr_matrix, find, issparse

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import hamming_loss, precision_score, recall_score, jaccard_score

import numpy as np

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse


def evaluation_nn(y_true, y_pred, classification_type='single-label', debug=False):
    """
    Evaluates the classification performance based on the true and predicted labels.
    It distinguishes between multilabel and singlelabel classification tasks and computes
    appropriate metrics. While scikit-learn provides a full set of evaluation metrics, they 
    treat special cases differently - i.e., when the number of true positives, false positives, 
    and false negatives ammount to 0, all affected metrics (precision, recall, and thus f1) output 
    0 in scikit-learn. Here we adhere to the common practice of outputting 1 in this case since 
    the classifier has correctly classified all examples as negatives.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - classification_type (str): Specifies 'multilabel' or 'singlelabel'.

    Returns:
    - Mf1 (float): Macro F1 score.
    - mf1 (float): Micro F1 score.
    - accuracy (float): Accuracy score
    - h_loss (float): Hamming loss
    - precision (float): Precision
    - recall (float): Recall
    - j_index (float): Jacard Index
    """

    """
    print("-- util.metrics.evaluation() --")
    print("y_true:", type(y_true), y_true.shape)
    print("y_pred:", type(y_pred), y_pred.shape)
    """

    #print("\n\tevaluating...")
    #print("classification_type:", classification_type)

    if classification_type in ['multilabel', 'multi-label']:
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = multilabel_eval(y_true, y_pred, debug=debug)    
    elif classification_type in ['singlelabel', 'single-label']:
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = singlelabel_eval_nn(y_true, y_pred, debug=debug)         
    else:
        print(f'Warning: unknown classification type {classification_type}')

    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index



def evaluation_ml(y_true, y_pred, classification_type='single-label', debug=True):
    """
    Evaluates the classification performance based on the true and predicted labels.
    It distinguishes between multilabel and singlelabel classification tasks and computes
    appropriate metrics. While scikit-learn provides a full set of evaluation metrics, they 
    treat special cases differently - i.e., when the number of true positives, false positives, 
    and false negatives ammount to 0, all affected metrics (precision, recall, and thus f1) output 
    0 in scikit-learn. Here we adhere to the common practice of outputting 1 in this case since 
    the classifier has correctly classified all examples as negatives.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - classification_type (str): Specifies 'multilabel' or 'singlelabel'.

    Returns:
    - Mf1 (float): Macro F1 score.
    - mf1 (float): Micro F1 score.
    - accuracy (float): Accuracy score
    - h_loss (float): Hamming loss
    - precision (float): Precision
    - recall (float): Recall
    - j_index (float): Jacard Index
    """

    """
    print("-- util.metrics.evaluation() --")
    print("y_true:", type(y_true), y_true.shape)
    print("y_pred:", type(y_pred), y_pred.shape)
    """

    print("\n\tevaluating...")
    print("classification_type:", classification_type)

    if classification_type in ['multilabel', 'multi-label']:
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = multilabel_eval(y_true, y_pred, debug=debug)    
    elif classification_type in ['singlelabel', 'single-label']:
        Mf1, mf1, accuracy, h_loss, precision, recall, j_index = singlelabel_eval_ml(y_true, y_pred, debug=debug)         
    else:
        print(f'Warning: unknown classification type {classification_type}')

    return Mf1, mf1, accuracy, h_loss, precision, recall, j_index


def multilabel_eval(y, y_, debug=False):
    """
    Evaluates multilabel classification performance by computing the F1 score,
    accuracy, and providing detailed metrics per label.

    Parameters:
    - y (CSR matrix): True multilabel matrix.
    - y_ (CSR matrix): Predicted multilabel matrix.

    Outputs detailed metrics including F1 scores per class and aggregated
    scores for true positives, false positives, and false negatives.

    Returns:
    - macro_f1 (float): Macro F1 score averaged over all classes.
    - micro_f1 (float): Micro F1 score calculated globally.
    - accuracy (float): Overall accuracy of the model.
    """
    
    # Ensure the matrices are in CSR format for indexing
    if not isinstance(y, csr_matrix):
        y = csr_matrix(y)
    if not isinstance(y_, csr_matrix):
        y_ = csr_matrix(y_)

    if (debug):
        print("-- to csr_matrices --")
        print("y:", y.shape, "\n", y)
        print("y_:", y_.shape, "\n", y_)

    # Calculate true positives (tp): 
    # where both y and y_ have a positive label
    tp = y.multiply(y_)    
    
    if (debug):
        print("tp:", tp.shape, "\n", tp)

    #true_ones = y==1
    #print("true_ones:", true_ones)

    # Create a new CSR matrix for false negatives (fn)
    # y and y_ are assumed to be in csr format
    # tp is true positives, already computed

    # False negatives: Conditions where y is 1 and y_ is 0
    # You can compute this by subtracting tp from y
    fn = y - tp
    if (debug):
        print("fn: ", fn.shape, "\n", fn)

    # Create a new CSR matrix for false positives (fp)
    # False positives: Conditions where y_ is 1 and y is 0
    # You can compute this by subtracting tp from y_
    fp = y_ - tp
    if (debug):
        print("fp: ", fp.shape, "\n", fp)

    # Note: The above computations utilize the fact that tp, y, and y_ must 
    # be in compatible formats and that subtraction of two csr_matrices is directly supported.


    #macro-f1
    tp_macro = np.asarray(tp.sum(axis=0), dtype=int).flatten()
    fn_macro = np.asarray(fn.sum(axis=0), dtype=int).flatten()
    fp_macro = np.asarray(fp.sum(axis=0), dtype=int).flatten()

    pos_pred = tp_macro+fp_macro
    pos_true = tp_macro+fn_macro
    prec=np.zeros(shape=tp_macro.shape,dtype=float)
    rec=np.zeros(shape=tp_macro.shape,dtype=float)
    np.divide(tp_macro, pos_pred, out=prec, where=pos_pred>0)
    np.divide(tp_macro, pos_true, out=rec, where=pos_true>0)
    den=prec+rec

    macrof1=np.zeros(shape=tp_macro.shape,dtype=float)
    np.divide(np.multiply(prec,rec),den,out=macrof1,where=den>0)
    macrof1 *=2

    macrof1[(pos_pred==0)*(pos_true==0)]=1
    macrof1 = np.mean(macrof1)

    #micro-f1
    tp_micro = tp_macro.sum()
    fn_micro = fn_macro.sum()
    fp_micro = fp_macro.sum()
    pos_pred = tp_micro + fp_micro
    pos_true = tp_micro + fn_micro
    prec = (tp_micro / pos_pred) if pos_pred>0 else 0
    rec  = (tp_micro / pos_true) if pos_true>0 else 0
    den = prec+rec
    microf1 = 2*prec*rec/den if den>0 else 0
    if pos_pred==pos_true==0:
        microf1=1

    #accuracy
    ndecisions = np.multiply(*y.shape)
    tn = ndecisions - (tp_micro+fn_micro+fp_micro)
    acc = (tp_micro+tn)/ndecisions

    # Hamming Loss
    h_loss = hamming_loss(y.toarray(), y_.toarray())

    # Precision
    precision = precision_score(y.toarray(), y_.toarray(), average='micro', zero_division=1)

    # Recall
    recall = recall_score(y.toarray(), y_.toarray(), average='micro', zero_division=1)

    # Jaccard Index
    j_index = jaccard_score(y.toarray(), y_.toarray(), average='micro', zero_division=1)

    return macrof1, microf1, acc, h_loss, precision, recall, j_index




def singlelabel_eval_nn(y, y_, debug=True):
    """
    Evaluates single label classification performance by computing the macro and micro
    F1 scores and accuracy.

    Parameters:
    - y (array-like): True single label array.
    - y_ (array-like): Predicted single label array.

    Returns:
    - macro_f1 (float): Macro F1 score.
    - micro_f1 (float): Micro F1 score.
    - acc (float): Overall accuracy of the model.
    - h_loss (float): Hamming loss
    - precision (float): Precision
    - recall (float): Recall
    - j_index (float): Jacard Index
    """
    
    #print("-- util.metrics.singlelabel_eval_nn() --")

    if (debug):
        print("y:", type(y), y.shape)
        print("y:\n", y)
        print("y_:", type(y_), y_.shape)
        print("y_:\n", y_)

    # Convert y and y_ to dense format if they are sparse
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    if issparse(y_):
        y_ = y_.toarray().flatten()

    # Initialize the LabelEncoder and fit it on the ground truth labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    if (debug):
        print("y:", type(y), y.shape)
        print("y:\n", y)
        print("y_:", type(y_), y_.shape)
        print("y_:\n", y_)

    macrof1 = f1_score(y, y_, average='macro')
    microf1 = f1_score(y, y_, average='micro')
    
    acc = accuracy_score(y, y_)

    h_loss = hamming_loss(y, y_)
    precision = precision_score(y, y_, average='macro')
    recall = recall_score(y, y_, average='macro')
    j_index = jaccard_score(y, y_, average='macro')
    
    return macrof1, microf1, acc, h_loss, precision, recall, j_index



def singlelabel_eval_ml(y, y_, debug=False):
    """
    Evaluates single label classification performance by computing the macro and micro
    F1 scores and accuracy.

    Parameters:
    - y (array-like): True single label array.
    - y_ (array-like): Predicted single label array.

    Returns:
    - macro_f1 (float): Macro F1 score.
    - micro_f1 (float): Micro F1 score.
    - acc (float): Overall accuracy of the model.
    - h_loss (float): Hamming loss
    - precision (float): Precision
    - recall (float): Recall
    - j_index (float): Jacard Index
    """
    
    print("-- util.metrics.singlelabel_eval() --")

    print("y:", type(y), y.shape)
    print("y_:", type(y_), y_.shape)

    #if issparse(y_): y_ = y_.toarray().flatten()

    macrof1 = f1_score(y, y_, average='macro')
    microf1 = f1_score(y, y_, average='micro')
    
    acc = accuracy_score(y, y_)

    h_loss = hamming_loss(y, y_)
    precision = precision_score(y, y_, average='macro')
    recall = recall_score(y, y_, average='macro')
    j_index = jaccard_score(y, y_, average='macro')
    
    return macrof1, microf1, acc, h_loss, precision, recall, j_index

