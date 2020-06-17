"""
AUTHOR
Elisabetta Manduchi

SCOPE
Modification of Normalizer which handles indicator and adjY columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Normalizer
import re

class resAdjNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, norm='l2'):
        self.norm = norm

    def fit(self, X, y=None, **fit_params):
        X_train = pd.DataFrame.copy(X)
        for col in X_train.columns:

            if re.match(r'^indicator', str(col)) or re.match(r'^adjY', str(col)):
                X_train.drop(col, axis=1, inplace=True)
        est = Normalizer(norm=self.norm)
        self.transformer = est.fit(X_train)
        return self

    def transform(self, X):
        tmp_X = pd.DataFrame.copy(X)
        for col in tmp_X.columns:
            if re.match(r'^indicator', str(col)) or re.match(r'^adjY', str(col)):
                tmp_X.drop(col, axis=1, inplace=True)
        X_test_red = self.transformer.transform(tmp_X)

        indX = X.filter(regex='indicator')
        if indX.shape[1] == 0:
            raise ValueError("X has no indicator columns")

        adjY = X.filter(regex='adjY')
        if (adjY.shape[1] == 0):
            raise ValueError("X has no adjY columns")

        X_test_red = pd.DataFrame(X_test_red, index=indX.index)

        X_test = pd.concat([X_test_red, indX, adjY], axis = 1)
        return X_test
