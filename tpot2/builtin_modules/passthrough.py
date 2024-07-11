from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Passthrough(TransformerMixin,BaseEstimator):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        return X


class SkipTransformer(TransformerMixin,BaseEstimator):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        #empty array of same shape as X
        return np.array([]).reshape(X.shape[0],0)
    
