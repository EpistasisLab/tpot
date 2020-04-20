"""
AUTHOR
Elisabetta Manduchi

SCOPE
Modification of XGBRegressor which handles indicator and adjY columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
import re

class resAdjXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=3,
                 learning_rate=1e-1, subsample=0.05,
                 min_child_weight=2, nthread=1,
                 objective='reg:squarederror',
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.nthread = nthread
        self.objective = objective
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
        est = XGBRegressor(n_estimators=self.n_estimators,
                           max_depth=self.max_depth,
                           learning_rate=self.learning_rate,
                           subsample=self.subsample,
                           min_child_weight=self.min_child_weight,
                           nthread=self.nthread,
                           objective=self.objective,
                           random_state=self.random_state)
        self.estimator = est.fit(X_train, y_train)
        return self


    def predict(self, X):
        X_test = pd.DataFrame.copy(X)
        for col in X_test.columns:
            if re.match(r'^indicator', str(col)) or re.match(r'^adjY', str(col)):
                X_test.drop(col, axis=1, inplace=True)

        return self.estimator.predict(X_test)
