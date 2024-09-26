from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Passthrough(TransformerMixin,BaseEstimator):
    """
    A transformer that does nothing. It just passes the input array as is.
    """

    def fit(self, X=None, y=None):
        """
        Nothing to fit, just returns self.
        """
        return self

    def transform(self, X):
        """
        returns the input array as is.
        """
        return X


class SkipTransformer(TransformerMixin,BaseEstimator):
    """
    A transformer returns an empty array. When combined with FeatureUnion, it can be used to skip a branch.
    """
    def fit(self, X=None, y=None):
        """
        Nothing to fit, just returns self.
        """
        return self

    def transform(self, X):
        """
        returns an empty array.
        """
        return np.array([]).reshape(X.shape[0],0)
    
