from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

import numpy as np

from joblib import Parallel, delayed

from time import time

import os
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress specific sklearn warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


class MLSVC:
    """
    Multi-Label Support Vector Machine, with individual optimizations per binary problem.
    """
    
    def __init__(self, n_jobs=1, dataset_name='unassigned', pretrained=False, supervised=False, estimator=LinearSVC, verbose=False):

        self.verbose = verbose

        self._print("MLSVC:__init__()")

        self.n_jobs = n_jobs
        self.estimator = estimator

        # for the plot
        self.performance_metrics = []
        self.dataset_name = dataset_name
        self.pretrained = pretrained
        self.supervised = supervised
        

    def fit(self, X, y, **grid_search_params):

        print()
        print("\n---MLSVC::fit() ---")
        print("X, y:", X.shape, y.shape)
        print("grid_search_params:", grid_search_params)

        tini = time()
        assert len(y.shape)==2 and set(np.unique(y).tolist()) == {0,1}, 'data format is not multi-label'
        
        nD,nC = y.shape
        prevalence = np.sum(y, axis=0)
        #self.svms = np.array([self.estimator(*self.args, **self.kwargs) for _ in range(nC)])
        self.svms = np.array([self.estimator() for _ in range(nC)])

        #if 'param_grid' in grid_search_params:
        if grid_search_params and grid_search_params['param_grid']:
            self._print('grid_search activated with: {}'.format(grid_search_params))
            
            # Grid search cannot be performed if the category prevalence is less than the parameter cv.
            # In those cases we place a svm instead of a gridsearchcv
            cv = 5 if 'cv' not in grid_search_params else grid_search_params['cv']
            
            assert isinstance(cv, int), 'cv must be an int (other policies are not supported yet)'

            """
            scoring = {
                'precision': make_scorer(precision_score, average='micro'),
                'recall': make_scorer(recall_score, average='micro'),
                'f1': make_scorer(f1_score, average='micro')
            }
            grid_search_params.update({'scoring': scoring, 'refit': 'f1', 'verbose': True})

            self.svms = [GridSearchCV(svm_i, **grid_search_params) if prevalence[i]>=cv else svm_i
                        for i,svm_i in enumerate(self.svms)]
            """

            self.svms = [GridSearchCV(svm_i, refit=True, **grid_search_params) if prevalence[i]>=cv else svm_i
                         for i,svm_i in enumerate(self.svms)]
            
        for i in np.argwhere(prevalence==0).flatten():
            self.svms[i] = TrivialRejector()

        
        self.svms = Parallel(n_jobs=self.n_jobs)(
            delayed(self.svms[c].fit)(X,y[:,c]) for c,svm in enumerate(self.svms)
        )
        
        
        #self._print(f"Starting parallel training with {self.n_jobs} jobs...")
        """
        self.svms = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_and_collect)(svm, X, y[:, c], c, nC) for c, svm in enumerate(self.svms)
        )
        """
        
        """
        for c, svm in enumerate(self.svms):
            self._train_and_collect(svm, X, y[:, c], c, nC)
        """

        self.training_time = time() - tini


    def _train_and_collect(self, svm, X, y, label_index, nC):
  
        print("_train_and_collect()[svm X y label_index]:", {svm}, X.shape, y.shape, label_index)

        self._print(f"Training label {label_index+1}/{nC}...")
        #self._print(f"Training label {label_index+1}/{len(y)}...")  # len(y) replaces y.shape[1]
        
        svm.fit(X, y)
    
        if isinstance(svm, GridSearchCV):

            best_score = svm.best_score_
            best_params = svm.best_params_

            self.performance_metrics.append((label_index, best_score, best_params))

            self._print(f"Best Score for label {label_index}: {best_score} with params {best_params}")
        
        return svm


    def visualize_performance(self, output_dir='../output'):
        
        self._print(f"visualize_performance()")
        
        if self.performance_metrics:
            labels, scores, params = zip(*self.performance_metrics)
            plt.figure(figsize=(10, 5))
            plt.bar(labels, scores, color='blue')
            
            plt.xlabel('Label Index')
            plt.ylabel('Best Grid Search Score')
            
            # Enhanced title with configuration details
            title = f'Performance for {self.estimator.__name__} on {self.dataset_name}\n'
            title += f'Params: max_iter={self.kwargs.get("max_iter", "N/A")}, '
            title += f'Pretrained: {self.pretrained}, Supervised: {self.supervised}'
            plt.title(title)

            # Filename generation based on class attributes and major parameters
            filename = f"{self.estimator.__name__}_params_{self.kwargs['max_iter']}_ds_{self.dataset_name}"
            filename += f"_pretrained_{self.pretrained}_supervised_{self.supervised}.png"
            filename = filename.replace(' ', '_').lower()  # Make filename conform to common standards

            if not os.path.exists(output_dir_dir):
                os.makedirs(save_dir)
            plot_path = os.path.join(output_dir, filename)
            plt.savefig(plot_path)
            plt.close()
            self._print(f"Performance plot saved to {plot_path}")
        else:
            self._print("No performance metrics to visualize.")

    def predict(self, X):
        return np.vstack(list(map(lambda svmi: svmi.predict(X), self.svms))).T


    def predict_proba(self, X):
        return np.vstack(map(lambda svmi: svmi.predict_proba(X)[:,np.argwhere(svmi.classes_==1)[0,0]], self.svms)).T


    def _print(self, msg):
        if self.verbose>0:
            print(msg)


    def best_params(self):
        return [svmi.best_params_ if isinstance(svmi, GridSearchCV) else None for svmi in self.svms]


class TrivialRejector:
    def fit(self,*args,**kwargs): return self
    def predict(self, X): return np.zeros(X.shape[0])
    def predict_proba(self, X): return np.zeros(X.shape[0])

