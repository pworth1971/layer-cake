import numpy as np

from data.tsr_function__ import get_supervised_matrix, get_tsr_matrix, information_gain, chi_square, conf_weight, word_prob
from model.embedding_predictor import EmbeddingPredictor

#from util.common import *

from sklearn.decomposition import PCA
from data.tsr_function__ import  STWFUNCTIONS


from scipy.sparse import csr_matrix


# ----------------------------------------------------------------------------------------------------------------------
def get_supervised_embeddings(X, y, max_label_space=300, binary_structural_problems=-1, method='dotn', dozscore=True, debug=False):
    """
    General function to compute supervised embeddings using different methods (e.g., TF-IDF, 
    PPMI, various TSR functions).

    Implementation: Depending on the method chosen and the structure of the label space, 
    applies z-score normalization and possibly PCA for dimensionality reduction if the 
    label space exceeds a defined limit.
    
    Args:
        X: Feature matrix (sparse or dense).
        y: Label matrix (sparse or dense).
        max_label_space: Maximum dimensions for label space (applies PCA if exceeded).
        binary_structural_problems: Threshold for binary problems (not used here).
        method: Supervised embedding method ('dotn', 'ppmi', etc.).
        dozscore: Whether to apply z-score normalization.
        debug: Debug flag for verbose output.

    Returns:
        F: Supervised embedding matrix.
    """

    print(f'get_supervised_embeddings(), method: {method}, dozscore: {dozscore}, max_label_space: {max_label_space}')

    # Validate sparse or dense matrix types
    is_sparse_X = isinstance(X, csr_matrix)
    is_sparse_y = isinstance(y, csr_matrix)

    # Check for empty rows or invalid values
    if is_sparse_X:
        if np.isnan(X.data).any() or np.isinf(X.data).any():
            raise ValueError("[ERROR] X contains NaN or Inf values.")
        """
        if X.getnnz(axis=1).min() == 0:
            print("[WARNING] X contains rows with no nonzero entries. Filtering them...")
            if (debug):
                nonzero_indices = X.getnnz(axis=1) > 0
                X = X[nonzero_indices]
                y = y[nonzero_indices]
                print(f"X.shape: {X.shape}, y.shape: {y.shape}")
                print("X:", X)
                print("y:", y)
            raise ValueError("[ERROR] X contains rows with no nonzero entries.")
        """
    else:
        raise ValueError("[ERROR] X is not a csr_matrix.")
        return None

    if is_sparse_y:
        if np.isnan(y.data).any() or np.isinf(y.data).any():
            raise ValueError("[ERROR] y contains NaN or Inf values.")
        """
        if y.getnnz(axis=1).min() == 0:
            raise ValueError("[ERROR] y contains rows with no nonzero entries.")
        """
    else:
        raise ValueError("[ERROR] y is not a csr_matrix.")
        return None


    nC = y.shape[1]
 
    if (debug):
        print("X:", type(X), X.shape)
        print("X[0]:", type(X[0]), X[0])

        print("Y:", type(y), y.shape)
        print("Y[0]:", type(y[0]), y[0])
        
        print("nC:", {nC})

    if nC==2 and binary_structural_problems > nC:
        raise ValueError('not implemented in this branch')

    if method=='ppmi':
        F = supervised_embeddings_ppmi(X, y)
    elif method == 'dotn':
        F = supervised_embeddings_tfidf(X, y, debug=debug)
    elif method == 'ig':
        F = supervised_embeddings_tsr(X, y, information_gain)
    elif method == 'chi2':
        F = supervised_embeddings_tsr(X, y, chi_square)
    elif method == 'cw':
        F = supervised_embeddings_tsr(X, y, conf_weight)
    elif method == 'wp':
        F = supervised_embeddings_tsr(X, y, word_prob)
        
    if dozscore:
        #F = zscores(F, axis=0)
        F = normalize_zscores(F)

    if max_label_space!=-1 and nC > max_label_space:
        print(f'supervised matrix has more dimensions ({nC}) than the allowed limit {max_label_space}. '
              f'Applying PCA(n_components={max_label_space})')
        pca = PCA(n_components=max_label_space)
        F = pca.fit(F).transform(F)

    return F
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
def supervised_embeddings_tfidf(X, Y, debug=False):
    """
    Computes term frequency-inverse document frequency (TF-IDF) like features but supervised with 
    label information Y.

    Implementation: Uses the term frequencies X and a supervised label matrix Y to weight 
    the term frequencies by how
    """

    print("supervised_embeddings_tfidf()")

    if (debug):
        print("X:", type(X), {X.shape})
        print("X[0]:", type(X[0]), X[0])
        print("Y:", type(Y), {Y.shape})
        print("Y[0]:", type(Y[0]), Y[0])
    
    # Compute tf-idf normalization
    tfidf_norm = X.sum(axis=0)                  # Sum of term frequencies

    epsilon = 1e-6                              # Small constant to prevent division by zero
    tfidf_norm = tfidf_norm + epsilon

    #tfidf_norm = np.asarray(tfidf_norm).flatten()  # Ensure it's a 1D array
    #tfidf_norm = tfidf_norm[:, None]  # Reshape to (30000, 1) for broadcasting

    if (debug):
        print("tfidf_norm:", type(tfidf_norm), tfidf_norm.shape)
        print("tfidf_norm[0]:", type(tfidf_norm[0]), tfidf_norm[0])

        numerator = (X.T).dot(Y)
        print("numerator:", type(numerator), {numerator.shape})
        print("numerator[0]:", type(numerator[0]), numerator[0])

        #denominator = tfidf_norm.T
        denominator = tfidf_norm
        print("denominator:", type(denominator), {denominator.shape})
        print("denominator[0]:", type(denominator[0]), denominator[0])

    F = (X.T).dot(Y) / tfidf_norm.T
    #F = (X.T).dot(Y) / tfidf_norm
    
    return F
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def supervised_embeddings_ppmi(X,Y):
    """
    Calculates Positive Pointwise Mutual Information (PPMI) for embeddings, supervised by labels. 

    Implementation: Converts the term matrix X into a binary presence matrix, calculates joint 
    probabilities of terms and labels, and computes the PPMI.
    """
    Xbin = X>0
    D = X.shape[0]
    Pxy = (Xbin.T).dot(Y)/D
    Px = Xbin.sum(axis=0)/D
    Py = Y.sum(axis=0)/D
    F = np.asarray(Pxy/(Px.T*Py))
    F = np.maximum(F, 1.0)
    F = np.log(F)
    return F
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def supervised_embeddings_tsr(X,Y, tsr_function=information_gain, max_documents=25000):
    """
    Computes embeddings using a term specificity ranking (TSR) function, supervised by labels. 

    Implementation: Potentially reduces the data size for efficiency, uses get_supervised_matrix 
    to create a contingency matrix between terms and labels, then computes the TSR using various 
    statistical measures (e.g., information gain).
    """    
    D = X.shape[0]
    
    if D>max_documents:
        print(f'sampling {max_documents}')
        random_sample = np.random.permutation(D)[:max_documents]
        X = X[random_sample]
        Y = Y[random_sample]
    cell_matrix = get_supervised_matrix(X, Y)
    
    F = get_tsr_matrix(cell_matrix, tsr_score_funtion=tsr_function).T
    
    return F
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------

def normalize_zscores(data, debug=False):
    """
    Function to compute z-scores for each feature across all samples. Replacement for 
    zscores() below, which is not working with the new numpy libraries.

    Parameters:
        data (numpy.ndarray): The input data array with shape (n_samples, n_features).
 
    Returns:
        numpy.ndarray: Z-score normalized data array.
    """    
    
    if (debug):
        print("--- normalize_zscores() ---")
        print("data:", type(data), {data.shape})

    if isinstance(data, np.matrix):
        arrData = np.asarray(data)
    elif isinstance(data, np.ndarray):
        arrData = data
    else:
        arrData = data.toarray()
    
    if (debug):
        print("arrData:", type(arrData), {arrData.shape})

    """
    means = np.mean(data, axis=0)       # Mean of the data (computing along the rows: axis=0)
    stds = np.std(data, axis=0)         # Standard deviation of the data (computing along the rows: axis=0) 
    z_scores = (data - means) / stds    # Compute the z-scores: (x - mean) / std
    """

    means = np.mean(arrData, axis=0)       # Mean of the data (computing along the rows: axis=0)
    stds = np.std(arrData, axis=0)         # Standard deviation of the data (computing along the rows: axis=0) 
    z_scores = (arrData - means) / stds    # Compute the z-scores: (x - mean) / std
    
    return z_scores
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def zscores_deprecated(x, axis=0):                             #scipy.stats.zscores does not avoid division by 0
    """
    Normalizes an array X using z-score normalization.

    Implementation: The standard deviation is calculated with a minimum clip value to 
    prevent division by zero. This normalized form ensures each feature (column if axis=0) 
    has zero mean and unit variance.
    """    
    print("\t--- zscores() ---")
    print("x:", type(x), {x.shape})
    #print("x[0]:", x[0])
    print("x dtype:", x.dtype)                  # Check data type
    print("axis: ", {axis})

    #arrX = x.todense(x)
    arrX = x.todense()                          # coo_matrix -> dense matrix
    print("arrX shape:", {arrX.shape})
    
    np_std = np.std(arrX, ddof=1, axis=axis)
    std = np.clip(np_std, 1e-5, None)

    #std = np.clip(np.std(x, ddof=1, axis=axis), 1e-5, None)
    #mean = np.mean(x, axis=axis)
    mean = np.mean(arrX, axis=axis)
    
    #return (x - mean) / std
    return (arrX - mean) / std
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

def fit_predict(W, F, mode='all'):
    """
    Trains a regression model to predict embeddings for a vocabulary based on an
    existing embedding matrix and then uses this model to predict missing or all embeddings.

    Learns a regression function from W->F using the embeddings which appear both in W and F and 
    produces a prediction for the missing embeddings in F (mode='missing') or for all (mode='all')

    Implementation: Uses an EmbeddingPredictor class (not defined in the provided code) that presumably 
    handles the fitting of a predictive model. The function adjusts predictions based on the mode 
    specified (all or missing), handling the embeddings differently based on the data available and 
    the desired output.

        :param W: an embedding matrix of shape (V1,E1), i.e., vocabulary-size-1 x embedding-size-1
        :param F: an embedding matris of shape (V2,E2), i.e., vocabulary-size-2 x embedding-size-2
        :param mode: indicates which embeddings are to be predicted. mode='all' will return a matrix of shape (V1,E2) where
        all V1 embeddings are predicted; when mode='missing' only the last V1-V2 embeddings will be predicted, and the previous
        V2 embeddings are copied
    
        :return: an embedding matrix of shape (V1,E2)
    """
    V1,E1=W.shape
    V2,E2=F.shape
    assert mode in {'all','missing'}, 'wrong mode; availables are "all" or "missing"'

    e = EmbeddingPredictor(input_size=E1, output_size=E2).cuda()
    e.fit(W[:V2], F)
    if mode == 'all':
        print('learning to predict all embeddings')
        F = e.predict(W)
    elif mode=='missing':
        print('learning to predict only the missing embeddings')
        Fmissing = e.predict(W[V2:])
        F = np.vstack((F, Fmissing))
    return F
# ----------------------------------------------------------------------------------------------------------------------





