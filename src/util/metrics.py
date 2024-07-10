import numpy as np
from scipy.sparse import lil_matrix, issparse
from sklearn.metrics import f1_score, accuracy_score

from scipy.sparse import csr_matrix, find


import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress specific sklearn warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


"""
Scikit learn provides a full set of evaluation metrics, but they treat special cases differently.
I.e., when the number of true positives, false positives, and false negatives ammount to 0, all
affected metrices (precision, recall, and thus f1) output 0 in Scikit learn.
We adhere to the common practice of outputting 1 in this case since the classifier has correctly
classified all examples as negatives.
"""

def evaluation(y_true, y_pred, classification_type):

    print("-- util.metrics.evaluation() --")

    print("y_true:", y_true)
    print("y_pred:", y_pred)
    print("classification_type:", classification_type)

    if classification_type == 'multilabel':
        eval_function = multilabel_eval
    elif classification_type == 'singlelabel':
        eval_function = singlelabel_eval

    Mf1, mf1, accuracy = eval_function(y_true, y_pred)

    return Mf1, mf1, accuracy


def multilabel_eval_new(y, y_):

    print("-- util.metrics.multilabel_eval_new() --")

    if not isinstance(y, csr_matrix):
        y = csr_matrix(y)
    
    if not isinstance(y_, csr_matrix):
        y_ = csr_matrix(y_)

    # Calculate true positives
    tp = y.multiply(y_)

    # Calculate false negatives
    # `y` is true but `y_` is false (i.e., y - tp)
    fn = y - tp

    # Calculate false positives
    # `y_` is true but `y` is false (i.e., y_ - tp)
    fp = y_ - tp

    # Compute macro F1, micro F1, and accuracy
    # First, sum over rows to get per-label counts
    tp_sum = np.array(tp.sum(axis=0)).flatten()
    fn_sum = np.array(fn.sum(axis=0)).flatten()
    fp_sum = np.array(fp.sum(axis=0)).flatten()

    # Calculate precision and recall per label
    precision = np.divide(tp_sum, tp_sum + fp_sum, out=np.zeros_like(tp_sum, dtype=float), where=(tp_sum + fp_sum) > 0)
    recall = np.divide(tp_sum, tp_sum + fn_sum, out=np.zeros_like(tp_sum, dtype=float), where=(tp_sum + fn_sum) > 0)

    # Calculate F1 scores
    f1_scores = 2 * precision * recall / (precision + recall)
    f1_scores[np.isnan(f1_scores)] = 0  # Handle division by zero if both precision and recall are zero

    macro_f1 = np.mean(f1_scores)
    micro_precision = tp_sum.sum() / (tp_sum.sum() + fp_sum.sum()) if (tp_sum.sum() + fp_sum.sum()) > 0 else 0
    micro_recall = tp_sum.sum() / (tp_sum.sum() + fn_sum.sum()) if (tp_sum.sum() + fn_sum.sum()) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Calculate accuracy
    total_elements = y.shape[0] * y.shape[1]
    correct_predictions = tp_sum.sum() + (total_elements - (tp_sum + fp_sum + fn_sum).sum())
    accuracy = correct_predictions / total_elements

    return macro_f1, micro_f1, accuracy


def multilabel_eval_orig(y, y_):

    print()
    
    print("-- multilabel_eval() -- ")

    print("y, y_ shapes: ", y.shape, y_.shape)

    tp = y.multiply(y_)

    print("tp: ", tp.shape)

    fn = lil_matrix(y.shape)
    
    print("fn: ", fn.shape)
    
    true_ones = y==1

    print("true_ones: ", true_ones)

    fn[true_ones]=1-tp[true_ones]

    fp = lil_matrix(y.shape)
    pred_ones = y_==1
    if pred_ones.nnz>0:
        fp[pred_ones]=1-tp[pred_ones]

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

    return macrof1,microf1,acc


def multilabel_eval(y, y_):

    print()
    print("-- multilabel_eval() -- ")
    print("y, y_ shapes: ", y.shape, y_.shape)


    # Ensure the matrices are in a format that supports element-wise multiplication and assignment.
    if not isinstance(y, csr_matrix):
        y = csr_matrix(y)
    if not isinstance(y_, csr_matrix):
        y_ = csr_matrix(y_)

    """
    BEGIN OLD_CODE
    
    tp = y.multiply(y_)

    fn = lil_matrix(y.shape)
    true_ones = y==1
    fn[true_ones]=1-tp[true_ones]

    fp = lil_matrix(y.shape)
    pred_ones = y_==1
    if pred_ones.nnz>0:
        fp[pred_ones]=1-tp[pred_ones]
    
    END OLD_CODE
    """

    # Calculate true positives (tp), where both y and y_ have a positive label
    tp = y.multiply(y_)
    print("tp (true positives): ", tp.shape)

    fn = csr_matrix(y.shape)
    print("fn (false negatives): ", fn.shape)
    
    true_ones = y==1
    print("true_ones: ", true_ones)
    fn[true_ones]=1-tp[true_ones]

    fp = csr_matrix(y.shape)
    print("fn (false positives): ", fp.shape)

    pred_ones = y_==1
    print("pred_ones:", pred_ones)
    if pred_ones.nnz>0:
        fp[pred_ones]=1-tp[pred_ones]

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

    return macrof1,microf1,acc

def singlelabel_eval(y, y_):

    print("-- util.metrics.singlelabel_eval() --")

    if issparse(y_): y_ = y_.toarray().flatten()

    macrof1 = f1_score(y, y_, average='macro')
    microf1 = f1_score(y, y_, average='micro')
    
    acc = accuracy_score(y, y_)
    
    return macrof1, microf1, acc

