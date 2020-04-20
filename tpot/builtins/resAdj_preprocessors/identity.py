"""
AUTHOR
Elisabetta Manduchi

SCOPE
Modification of ZeroCount which handles indicator and adjY columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class identity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X
