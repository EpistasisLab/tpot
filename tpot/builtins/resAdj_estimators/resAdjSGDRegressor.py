"""
AUTHOR
Elisabetta Manduchi

SCOPE
Modification of SGDRegressor which handles indicator and adjY columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import SGDRegressor
import re

class resAdjSGDRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, loss='squared_loss', penalty='elasticnet',
                 alpha=0.001, learning_rate='invscaling',
                 fit_intercept=True, l1_ratio=0.25,
                 eta0=0.01, power_t=0.5,
                 random_state=None):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.l1_ratio = l1_ratio
        self.eta0 = eta0
        self.power_t = power_t
        self.random_state = random_state

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
        est = SGDRegressor(loss=self.loss, penalty=self.penalty,
                           alpha=self.alpha, learning_rate=self.learning_rate,
                           fit_intercept=self.fit_intercept,
                           l1_ratio=self.l1_ratio, eta0=self.eta0,
                           power_t=self.power_t, random_state=self.random_state)
        self.estimator = est.fit(X_train, y_train)
        return self


    def predict(self, X):
        X_test = pd.DataFrame.copy(X)
        for col in X_test.columns:
            if re.match(r'^indicator', str(col)) or re.match(r'^adjY', str(col)):
                X_test.drop(col, axis=1, inplace=True)

        return self.estimator.predict(X_test)
