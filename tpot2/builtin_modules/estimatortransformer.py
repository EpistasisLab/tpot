from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import available_if
import numpy as np
from sklearn.utils.validation import check_is_fitted

class EstimatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, method='auto', passthrough=False, cross_val_predict_cv=0):
        """
        A class for using a sklearn estimator as a transformer.

        Parameters
        ----------
        estimator : sklear.base. BaseEstimator
            The estimator to use as a transformer.
        method : str, default='auto'
            The method to use for the transformation. If 'auto', will try to use predict_proba, decision_function, or predict in that order.
            - predict_proba: use the predict_proba method of the estimator.
            - decision_function: use the decision_function method of the estimator.
            - predict: use the predict method of the estimator.
        passthrough : bool, default=False
            Whether to pass the original input through.
        cross_val_predict_cv : int, default=0
            Number of folds to use for the cross_val_predict function for inner classifiers and regressors. Estimators will still be fit on the full dataset, but the following node will get the outputs from cross_val_predict.

            - 0-1 : When set to 0 or 1, the cross_val_predict function will not be used. The next layer will get the outputs from fitting and transforming the full dataset.
            - >=2 : When fitting pipelines with inner classifiers or regressors, they will still be fit on the full dataset.
                    However, the output to the next node will come from cross_val_predict with the specified number of folds.

        """
        self.estimator = estimator
        self.method = method
        self.passthrough = passthrough
        self.cross_val_predict_cv = cross_val_predict_cv
    
    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        #Does not do cross val predict, just uses the estimator to transform the data. This is used for the actual transformation in practice, so the real transformation without fitting is needed
        if self.method == 'auto':
            if hasattr(self.estimator, 'predict_proba'):
                method = 'predict_proba'
            elif hasattr(self.estimator, 'decision_function'):
                method = 'decision_function'
            elif hasattr(self.estimator, 'predict'):
                method = 'predict'
            else:
                raise ValueError('Estimator has no valid method')
        else:
            method = self.method
        
        output = getattr(self.estimator, method)(X)
        output=np.array(output)
        
        if len(output.shape) == 1:
            output = output.reshape(-1,1)

        if self.passthrough:
            return np.hstack((output, X))
        else:
            return output
        

        
    def fit_transform(self, X, y=None):
        #Does use cross_val_predict if cross_val_predict_cv is greater than 0. this function is only used in training the model. 
        self.estimator.fit(X,y)

        if self.method == 'auto':
            if hasattr(self.estimator, 'predict_proba'):
                method = 'predict_proba'
            elif hasattr(self.estimator, 'decision_function'):
                method = 'decision_function'
            elif hasattr(self.estimator, 'predict'):
                method = 'predict'
            else:
                raise ValueError('Estimator has no valid method')
        else:
            method = self.method
        
        if self.cross_val_predict_cv > 0:
            output = cross_val_predict(self.estimator, X, y=y, cv=self.cross_val_predict_cv)
        else:
            output = getattr(self.estimator, method)(X)
            #reshape if needed
        
        if len(output.shape) == 1:
            output = output.reshape(-1,1)

        output=np.array(output)
        if self.passthrough:
            return np.hstack((output, X))
        else:
            return output
    
    def _estimator_has(attr):
        '''Check if we can delegate a method to the underlying estimator.
        First, we check the first fitted final estimator if available, otherwise we
        check the unfitted final estimator.
        '''
        return  lambda self: (self.estimator is not None and
            hasattr(self.estimator, attr)
        )

    @available_if(_estimator_has('predict'))
    def predict(self, X, **predict_params):
        check_is_fitted(self.estimator)
        #X = check_array(X)

        preds = self.estimator.predict(X,**predict_params)
        return preds

    @available_if(_estimator_has('predict_proba'))
    def predict_proba(self, X, **predict_params):
        check_is_fitted(self.estimator)
        #X = check_array(X)
        return self.estimator.predict_proba(X,**predict_params)

    @available_if(_estimator_has('decision_function'))
    def decision_function(self, X, **predict_params):
        check_is_fitted(self.estimator)
        #X = check_array(X)
        return self.estimator.decision_function(X,**predict_params)

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return check_is_fitted(self.estimator)


    # @property
    # def _estimator_type(self):
    #     return self.estimator._estimator_type
    

    
    @property
    def classes_(self):
        """The classes labels. Only exist if the last step is a classifier."""
        return self.estimator._classes