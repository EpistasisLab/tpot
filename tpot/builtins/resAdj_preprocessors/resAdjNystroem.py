"""
AUTHOR
Elisabetta Manduchi

SCOPE
Modification of Nystroem which handles indicator and adjY columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.kernel_approximation import Nystroem
import re

class resAdjNystroem(BaseEstimator, TransformerMixin):
    def __init__(self, kernel='rbf', gamma=None, n_components=10, random_state=None):
        self.kernel = kernel
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        X_train = pd.DataFrame.copy(X)
        for col in X_train.columns:

            if re.match(r'^indicator', str(col)) or re.match(r'^adjY', str(col)):
                X_train.drop(col, axis=1, inplace=True)
        est = Nystroem(kernel=self.kernel, gamma=self.gamma,
                       n_components=self.n_components, random_state=self.random_state)
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
