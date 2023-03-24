from sklearn.base import BaseEstimator, TransformerMixin

class Passthrough(TransformerMixin,BaseEstimator):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        return X
