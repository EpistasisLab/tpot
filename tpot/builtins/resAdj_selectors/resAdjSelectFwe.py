"""
AUTHOR
Elisabetta Manduchi

DATE
April 9, 2020

SCOPE
Modification of SelectFwe which handles indicator and adjY columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, SelectorMixin
from sklearn.feature_selection import SelectFwe
import re

class resAdjSelectFwe(BaseEstimator, SelectorMixin):
    def __init__(self, alpha=0.05, score_funct='f_regression'):
        self.alpha = alpha
        self.score_func = score_func

    def fit(self, X, y=None, **fit_params):
        X_train = pd.DataFrame.copy(X)
        for col in X_train.columns:
            if re.match(r'^indicator', col) or re.match(r'^adjY', col):
                X_train.drop(col, axis=1, inplace=True)

        indX = X.filter(regex='indicator')
        if indX.shape[1] == 0:
            raise ValueError("X has no indicator columns")
        adjY = X.filter(regex='adjY')
        if (adjY.shape[1] == 0):
            raise ValueError("X has no adjY columns")
        for col in indX.columns:
            if sum(indX[col])==0:
                i = col.split('_')[1]
                y_train = X['adjY_' + i]
                break

        est = SelectFwe(alpha=self.alpha, score_func=self.score_func, **fit_params)
        self.transformer = est.fit(X_train, y_train)
        return self

    def transform(self, X):
        tmp_X = pd.DataFrame.copy(X)
        for col in tmp_X.columns:
            if re.match(r'^indicator', col) or re.match(r'^adjY', col):
                tmp_X.drop(col, axis=1, inplace=True)
        X_test_red = self.transformer.transform(tmp_X)

        indX = X.filter(regex='indicator')
        if indX.shape[1] == 0:
            raise ValueError("X has no indicator columns")

        adjY = X.filter(regex='adjY')
        if (adjY.shape[1] == 0):
            raise ValueError("X has no adjY columns")

        X_test = pd.concat([X_test_red, indX, adjY], axis = 1)
        return X_test
