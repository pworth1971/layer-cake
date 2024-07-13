import numpy as np

from scipy.sparse import lil_matrix, issparse
from scipy.sparse import csr_matrix, find

from sklearn.metrics import f1_score, accuracy_score


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
    #print("y_true:", y_true.shape, "\n", y_true)
    #print("y_pred:", y_pred.shape, "\n", y_pred)
    
    if classification_type == 'multilabel':
        eval_function = multilabel_eval
    elif classification_type == 'singlelabel':
        eval_function = singlelabel_eval

    Mf1, mf1, accuracy = eval_function(y_true, y_pred)

    return Mf1, mf1, accuracy


def multilabel_eval(y, y_):
    
    print("-- multilabel_eval() --")
    #print("y:", y.shape, "\n", y)
    #print("y_:", y_.shape, "\n", y_)

    # Ensure the matrices are in CSR format for indexing
    if not isinstance(y, csr_matrix):
        y = csr_matrix(y)
    if not isinstance(y_, csr_matrix):
        y_ = csr_matrix(y_)

    #print("-- to csr_matrices --")
    #print("y:", y.shape, "\n", y)
    #print("y_:", y_.shape, "\n", y_)

    #log_arrs(y.toarray(), y_.toarray())

    # Calculate true positives (tp): 
    # where both y and y_ have a positive label
    tp = y.multiply(y_)    
    #print("tp:", tp.shape, "\n", tp)

    #true_ones = y==1
    #print("true_ones:", true_ones)

    # Create a new CSR matrix for false negatives (fn)
    # y and y_ are assumed to be in csr format
    # tp is true positives, already computed

    # False negatives: Conditions where y is 1 and y_ is 0
    # You can compute this by subtracting tp from y
    fn = y - tp
    #print("fn: ", fn.shape, "\n", fn)

    # Create a new CSR matrix for false positives (fp)
    # False positives: Conditions where y_ is 1 and y is 0
    # You can compute this by subtracting tp from y_
    fp = y_ - tp
    #print("fp: ", fp.shape, "\n", fp)

    # Note: The above computations utilize the fact that tp, y, and y_ must be in compatible formats
    # and that subtraction of two csr_matrices is directly supported.

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

