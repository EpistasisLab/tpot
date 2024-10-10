import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

def select_features(X, min_unique=10,):
    """
    Given a DataFrame or numpy array, return a list of column indices that have more than min_unique unique values.

    Parameters
    ----------
    X: DataFrame or numpy array
        Data to select features from
    min_unique: int, default=10
        Minimum number of unique values a column must have to be selected

    Returns
    -------
    list
        List of column indices that have more than min_unique unique values
    
    """

    if isinstance(X, pd.DataFrame):
        return [col for col in X.columns if len(X[col].unique()) > min_unique]
    else:
        return [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) > min_unique]

class PassKBinsDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5,  encode='onehot-dense', strategy='quantile', subsample=None, random_state=None):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.subsample = subsample
        self.random_state = random_state
        """
        Same as sklearn.preprocessing.KBinsDiscretizer, but passes through columns that are not discretized due to having fewer than n_bins unique values instead of ignoring them.
        See sklearn.preprocessing.KBinsDiscretizer for more information.
        """

    def fit(self, X, y=None):
        # Identify columns with more than n unique values
        # Create a ColumnTransformer to select and discretize the chosen columns
        self.selected_columns_ = select_features(X, min_unique=10)
        if isinstance(X, pd.DataFrame):
            self.not_selected_columns_ = [col for col in X.columns if col not in self.selected_columns_]
        else:
            self.not_selected_columns_ = [i for i in range(X.shape[1]) if i not in self.selected_columns_]
        
        enc = KBinsDiscretizer(n_bins=self.n_bins, encode=self.encode, strategy=self.strategy, subsample=self.subsample, random_state=self.random_state)
        self.transformer = ColumnTransformer([
            ('discretizer', enc, self.selected_columns_),
            ('passthrough', 'passthrough', self.not_selected_columns_)
        ])
        self.transformer.fit(X)
        return self

    def transform(self, X):
        return self.transformer.transform(X)