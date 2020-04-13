"""
AUTHOR
Elisabetta Manduchi

DATE
April 8, 2020

SCOPE
Modification of AdjLinearSVR which handles indicator and adjY columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import LinearSVR
import re

class resAdjLinearSVR(BaseEstimator, RegressorMixin):
    def __init__(self, loss='epsilon_insensitive', dual=True,
                 tol=1e-4, C=1., epsilon=None, random_state=None):
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.random_state = random_state

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
        est = LinearSVR(loss=self.loss, dual=self.dual, tol=self.tol, C=self.C,
                        epsilon=self.epsilon, random_state=self.random_state)
        self.estimator = est.fit(X_train, y_train)
        return self


    def predict(self, X):
        X_test = pd.DataFrame.copy(X)
        for col in X_test.columns:
            if re.match(r'^indicator', col) or re.match(r'^adjY', col):
                X_test.drop(col, axis=1, inplace=True)

        return self.estimator.predict(X_test)
