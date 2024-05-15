from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import available_if
import numpy as np
from sklearn.utils.validation import check_is_fitted

class EstimatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, method='auto', passthrough=False, cross_val_predict_cv=0):
        self.estimator = estimator
        self.method = method
        self.passthrough = passthrough
        self.cross_val_predict_cv = cross_val_predict_cv
    
    def fit(self, X, y=None):
        return self.estimator.fit(X, y)
    
    def transform(self, X):
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