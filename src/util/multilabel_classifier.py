from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

import numpy as np
from joblib import Parallel, delayed
from time import time

import warnings
from sklearn.exceptions import ConvergenceWarning

class MLClassifier:
    """
    Multi-label, multiclass classifier. Support for LinearSVC, LogisticRegression, and Naive Bayes models, 
    with individual optimizations per binary problem (binary classifier per class with results averaged 
    across classes) and functionalities for optimizing hyperparameters on a per-label basis.
    """

    def __init__(self, n_jobs=1, estimator=LinearSVC, scoring='accuracy', *args, **kwargs):
        """
        Parameters:
        - n_jobs (int): Number of parallel jobs to run for fitting models.
        - estimator (object): A scikit-learn classifier (e.g., LinearSVC, LogisticRegression, or MultinomialNB).
        - scoring (str): Scoring method to use for evaluating the performance of the models.
        """
        print("\n\tInitializing MLClassifier...")

        self.n_jobs = n_jobs
        self.estimator = estimator
        self.scoring = scoring
        self.args = args
        self.kwargs = kwargs
        self.verbose = self.kwargs.get('verbose', False)

        print("n_jobs:", self.n_jobs)
        print(f"Using {self.estimator.__name__} as the estimator.")
        print("estimator args:", self.kwargs)

    def fit(self, X, y, **grid_search_params):
        """
        Fits the classifier to the data. Can perform grid search to find the best parameters for each label's classifier.

        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target label matrix.
        - **grid_search_params (dict): Parameters for GridSearchCV like 'param_grid' and 'cv'.
        """
        print("Fitting MLClassifier...")
        print("X, y:", X.shape, y.shape)

        # Suppress specific sklearn warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)

        tini = time()

        assert len(y.shape) == 2 and set(np.unique(y).tolist()) == {0, 1}, 'data format is not multi-label'
        
        nD, nC = y.shape
        prevalence = np.sum(y, axis=0)

        print(f"Instantiating {nC} instances of {self.estimator.__name__} with kwargs: {self.kwargs}")

        # Initialize a classifier for each label
        self.models = np.array([self.estimator(*self.args, **self.kwargs) for _ in range(nC)])

        # Handle grid search if specified
        if grid_search_params and grid_search_params.get('param_grid'):
            print('Grid search activated with parameters:', grid_search_params)
            cv = grid_search_params.get('cv', 5)
            assert isinstance(cv, int), 'cv must be an int.'

            self.models = [
                GridSearchCV(model, refit=True, **grid_search_params, scoring=self.scoring) 
                if prevalence[i] >= cv else model
                for i, model in enumerate(self.models)
            ]

        # Handle cases with no positive labels (i.e., TrivialRejector)
        for i in np.argwhere(prevalence == 0).flatten():
            self.models[i] = TrivialRejector()

        # Train the models
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(self.models[c].fit)(X, y[:, c]) for c in range(nC)
        )

        self.training_time = time() - tini

    def predict(self, X):
        """
        Makes predictions for the given data.

        Parameters:
        - X (array-like): Feature matrix to predict.

        Returns:
        - array-like: Predicted labels for each sample.
        """
        return np.vstack([model.predict(X) for model in self.models]).T

    def predict_proba(self, X):
        """
        Predicts class probabilities for the given data.

        Parameters:
        - X (array-like): Feature matrix.

        Returns:
        - array-like: Predicted probabilities for each class for each sample.
        """
        return np.vstack([model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(X.shape[0]) 
                          for model in self.models]).T

    def best_params(self):
        """
        Returns the best parameters for each fitted model if GridSearchCV was used.

        Returns:
        - list of dicts: Best parameters for each classifier.
        """
        return [model.best_params_ if isinstance(model, GridSearchCV) else None for model in self.models]


class TrivialRejector:
    """
    A trivial classifier that predicts the negative class for all samples. Used when no positive samples are available.

    Methods:
    - fit(X, y): Fits the classifier. Does nothing.
    - predict(X): Predicts zeros for all samples.
    - predict_proba(X): Returns zero probabilities for all samples.
    """
    def fit(self,*args,**kwargs): return self
    def predict(self, X): return np.zeros(X.shape[0])
    def predict_proba(self, X): return np.zeros(X.shape[0])




class MLClassifier_orig:
    """
    Multi-label, multiclass classifier. Support for LinearSVC and LogisticRegression models, with 
    individual optimizations per binary problem (binary classifier per class with results averaged 
    across classes (see metrics.py) and functionalities for optimizing hyperparameters on a per-label basis. 
    Technically it should work with with any classifier that adheres to the scikit-learn API but we test
    with LinearSVC and LogisticRegression classifiers only - again classifying in a binary fashion per 
    class and then aggregating results. This code is structured to allow detailed monitoring and modification 
    of machine learning processes, specifically tailored for scenarios involving multiple labels and potentially 
    sparse class distributions. The use of GridSearchCV enables optimization across different configurations, 
    targeting best practices in model tuning.

    Parameters:
    - n_jobs (int): Number of parallel jobs to run for fitting models.
    - dataset_name (str): Name of the dataset, used for logging and output.
    - pretrained (bool): Indicates if pretrained embeddings or features are used.
    - supervised (bool): Indicates if supervised learning methods like feature selection are applied.
    - estimator (object): An instance of a scikit-learn classifier like LinearSVC or LogisticRegression.
    - verbose (bool): Enable detailed logging if set to True.
    - scoring (str or callable): Scoring method to use for evaluating the performance of the models.

    Methods:
    - fit(X, y, **grid_search_params): Fits the classifier on the provided dataset.
    - predict(X): Predicts labels for the given features.
    - predict_proba(X): Predicts class probabilities for the given features.
    - best_params(): Returns the best parameters found by GridSearchCV for each label, if applicable.
    """


    def __init__(self, n_jobs=1, estimator=LinearSVC, scoring='accuracy', *args, **kwargs):

        print("Initializing MLClassifier...")
        
        self.n_jobs = n_jobs

        print("n_jobs:", self.n_jobs)
        
        self.estimator = estimator
        self.scoring = scoring
        self.args = args
        self.kwargs = kwargs

        self.verbose = False if 'verbose' not in self.kwargs else self.kwargs['verbose']

        print("estimator type:", {self.estimator.__name__})
        print("estimator args:", self.kwargs)
    

    def fit(self, X, y, **grid_search_params):
        """
        Fits the classifier to the data. Can perform grid search to find the best parameters for each label's classifier.

        Parameters:
        - X (array-like): Feature matrix.
        - y (array-like): Target label matrix.
        - **grid_search_params (dict): Parameters for GridSearchCV like 'param_grid' and 'cv' (cross-validation strategy).

        Raises:
        - AssertionError: If the data is not in the expected multi-label binary format.
        """

        print("fitting ML Classifier...")
        
        print("X, y:", X.shape, y.shape)
        print("grid_search_params:", grid_search_params)

        # Suppress specific sklearn warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)

        tini = time()
        assert len(y.shape)==2 and set(np.unique(y).tolist()) == {0,1}, 'data format is not multi-label'
        
        nD,nC = y.shape
        prevalence = np.sum(y, axis=0)

        print(f"Instantiating {nC} instances of {self.estimator.__name__} with kwargs: {self.kwargs}")

        self.svms = np.array([self.estimator(*self.args, **self.kwargs) for _ in range(nC)])

        if grid_search_params and grid_search_params['param_grid']:
        
            print('grid_search activated with: {}'.format(grid_search_params))
            
            # Grid search cannot be performed if the category prevalence is less than the parameter cv.
            # In those cases we place a svm instead of a gridsearchcv
            cv = 5 if 'cv' not in grid_search_params else grid_search_params['cv']
            
            assert isinstance(cv, int), 'cv must be an int (other policies are not supported yet)'

            self.svms = [GridSearchCV(svm_i, refit=True, **grid_search_params, scoring=self.scoring) if prevalence[i]>=cv else svm_i
                         for i,svm_i in enumerate(self.svms)]
            
        for i in np.argwhere(prevalence==0).flatten():
            self.svms[i] = TrivialRejector()
        
        self.svms = Parallel(n_jobs=self.n_jobs)(
            delayed(self.svms[c].fit)(X,y[:,c]) for c,svm in enumerate(self.svms)
        )

        self.training_time = time() - tini


    def predict(self, X):
        """
        Makes predictions for the given data.

        Parameters:
        - X (array-like): Feature matrix to predict.

        Returns:
        - array-like: Predicted labels for each sample.
        """
        return np.vstack(list(map(lambda svmi: svmi.predict(X), self.svms))).T


    def predict_proba(self, X):
        """
        Predicts class probabilities for the given data.

        Parameters:
        - X (array-like): Feature matrix.

        Returns:
        - array-like: Predicted probabilities for each class for each sample.
        """
        return np.vstack(map(lambda svmi: svmi.predict_proba(X)[:,np.argwhere(svmi.classes_==1)[0,0]], self.svms)).T


    def _print(self, msg):
        """
        Prints a message to stdout if verbose is enabled.

        Parameters:
        - msg (str): The message to print.
        """
        if self.verbose>0:
            print(msg)


    def best_params(self):
        """
        Returns the best parameters for each fitted model if GridSearchCV was used.

        Returns:
        - list of dicts: Best parameters for each classifier.
        """
        return [svmi.best_params_ if isinstance(svmi, GridSearchCV) else None for svmi in self.svms]


class TrivialRejector:
    """
    A trivial classifier that predicts the negative class for all samples. Used when no positive samples are available.

    Methods:
    - fit(X, y): Fits the classifier. Does nothing.
    - predict(X): Predicts zeros for all samples.
    - predict_proba(X): Returns zero probabilities for all samples.
    """
    def fit(self,*args,**kwargs): return self
    def predict(self, X): return np.zeros(X.shape[0])
    def predict_proba(self, X): return np.zeros(X.shape[0])

