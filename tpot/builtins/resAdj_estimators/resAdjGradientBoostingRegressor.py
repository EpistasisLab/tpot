"""
AUTHOR
Elisabetta Manduchi

SCOPE
Modification of GradientBoostingRegressor which handles indicator and adjY columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
import re

class resAdjGradientBoostingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, loss='ls', learning_rate=0.1,
                 max_depth=3, min_samples_split=2, min_samples_leaf=1,
                 subsample=0.05, max_features=None, alpha=0.9, random_state=None):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.alpha = alpha
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
        est = GradientBoostingRegressor(loss=self.loss,
                                        learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators,
                                        subsample=self.subsample,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        max_features=self.max_features,
                                        alpha=self.alpha,
                                        random_state=self.random_state)
        self.estimator = est.fit(X_train, y_train)
        return self


    def predict(self, X):
        X_test = pd.DataFrame.copy(X)
        for col in X_test.columns:
            if re.match(r'^indicator', str(col)) or re.match(r'^adjY', str(col)):
                X_test.drop(col, axis=1, inplace=True)

        return self.estimator.predict(X_test)
