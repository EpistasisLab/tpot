"""
AUTHOR
Elisabetta Manduchi

SCOPE
Modification of SelectPercentile which handles indicator and adjY columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectPercentile, f_regression
try:
    from sklearn.feature_selection._base import SelectorMixin
except ImportError:
    from sklearn.feature_selection.base import SelectorMixin
import re

class resAdjSelectPercentile(BaseEstimator, SelectorMixin):
    def __init__(self, percentile=10, score_func=f_regression):
        self.percentile = percentile
        self.score_func = score_func

    def fit(self, X, y=None, **fit_params):
        X_train = pd.DataFrame.copy(X)
        for col in X_train.columns:
            if re.match(r'^indicator', str(col)) or re.match(r'^adjY', str(col)):
                X_train.drop(col, axis=1, inplace=True)

        indX = X.filter(regex='indicator')
        if indX.shape[1] == 0:
            raise ValueError("X has no indicator columns")
        adjY = X.filter(regex='adjY')
        if (adjY.shape[1] == 0):
            raise ValueError("X has no adjY columns")

        y_train = y
        for col in indX.columns:
            if sum(indX[col])==0:
                i = col.split('_')[1]
                y_train = X['adjY_' + i]
                break

        est = SelectPercentile(percentile=self.percentile, score_func=self.score_func, **fit_params)
        self.transformer = est.fit(X_train, y_train)
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

        mask_cols = np.ones(indX.shape[1] + adjY.shape[1], dtype=bool)
        self.mask = np.hstack((self.transformer._get_support_mask(), mask_cols))

        return X_test

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected
        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """

        return self.mask
