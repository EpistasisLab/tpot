# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_array
from sklearn.linear_model import LinearRegression, LogisticRegression


class MetaEstimator(BaseEstimator, ClassifierMixin):
    """Meta-transformer for adding predictions and/or class probabilities as synthetic feature(s).

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    """

    def __init__(self, estimator, A=None, C=None):
        """Create a StackingEstimator object.

        Parameters
        ----------
        estimator: object with fit, predict, and predict_proba methods.
            The estimator to generate synthetic features from.
        A: pd.DataFrame, shape(n_samples_test, n_conf_covariates)
            Each column of A correspond to a non-confounding covariate.
        C: pd.DataFrame, shape(n_samples_test, n_conf_covariates)
            Each column of C correspond to a confounding covariate.

        """
        self.estimator = estimator
        self.A = A
        self.C = C


    def fit(self, X, y=None, **fit_params):
        """Fit the StackingEstimator meta-transformer.

        Parameters
        ----------
        X: pd.DataFrame of shape (n_samples, n_features)
            The training input samples.
        y: array-like, shape (n_samples,)
            The target values (integers that correspond to classes in classification, real numbers in regression).
        fit_params:
            Other estimator-specific parameters.

        Returns
        -------
        self: object
            Returns a copy of the estimator
        """
        if self.C is None:
            X_train_adj = X
            C_train = None
        else:
            X_train_adj = np.zeros(X.shape)
            self.col_ests = [] # store estimator for each columns
            self.values_list = [] # store values for each columns
            C_train = self.C.loc[X.index,:]

            for col in range(X.shape[1]):
                X_train_col = X.iloc[:, col].values.reshape((-1,1)) # np.ndarray
                # test information cannot be used in fit() function
                # may be values should be provided as a parameter in __init__ above
                # here values was stored into self.values_list and can be used in predict
                # function below for test dataset
                dosage = np.hstack((X_train_col!=0, X_train_col!=1, X_train_col!=2))
                if X_train_col[np.all(dosage, axis=1).reshape((-1, 1))].shape[0]>0:
                    values = 'dosage'
                else:
                    values = 'ternary'
                self.values_list.append(values)
                if values == 'dosage':
                    regr = LinearRegression(),
                    regr.fit(C_train, X_train_col)
                    X_train_adj[:, col:(col+1)] = X_train_col - regr.predict(C_train)
                    self.col_ests.append(regr)
                else:
                    clf = LogisticRegression(penalty='none',
                                            solver='lbfgs',
                                            multi_class='auto')
                    clf.fit(C_train, np.ravel(X_train_col))
                    X_train_adj[:, col:(col+1)] = X_train_col - clf.predict_proba(C_train)[:, 1:2]-2*clf.predict_proba(C_train)[:, 2:3]
                    self.col_ests.append(regr)
        if self.A is not None:
            A_train = self.A.loc[X.index,:]

        if self.A is None and self.C is None:
            rasie(ValueError, "At least one of A_train and C_train must be specified")
        elif C_train is None and A_train is not None:
            B_train = A_train
        elif A_train is None and C_train is not None:
            B_train = C_train
        else:
            B_train = np.hstack((A_train, C_train))

        self.B_clf = LogisticRegression(penalty='none', solver='lbfgs').
        self.B_clf.fit(B_train, y_train)
        pi_train = np.ravel(self.B_clf.predict_proba(B_train)[:, 1])
        y_train_adj = y_train - pi_train

        self.estimator.fit(X_train_adj, y_train_adj, **fit_params)
        return self

    def predict(self, X):
        """Transform data by adding two synthetic feature(s).

        Parameters
        ----------
        X: pd.DataFrame, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        y_pred: array-like, shape (n_samples, )
        """
        if self.C is None:
            X_test_adj = X
            C_test = None
        else:
            X_test_adj = np.zeros(X.shape)
            C_test = self.C.loc[X.index,:]
            for values, est in zip(self.values_list, self.col_ests):
                if values == 'dosage':
                    X_test_adj[:, col:(col+1)] = X_test_adj - est.predict(C_test)
                else:
                    X_test_adj[:, col:(col+1)] = X_test_adj - est.predict_proba(C_test)[:, 1:2]-2*est.predict_proba(C_test)[:, 2:3]
        if self.A is not None:
            A_test = self.A.loc[X.index,:]
        if self.A is None and self.C is None:
            rasie(ValueError, "At least one of A_train and C_train must be specified")
        elif C_train is None and A_train is not None:
            B_test = A_test
        elif A_train is None and C_train is not None:
            B_test = C_test
        else:
            B_test = np.hstack((A_test, C_test))
        pi_test = np.ravel(self.B_clf.predict_proba(B_test)[:, 1])
        y_test_adj_pred = self.estimator.predict(X_test_adj)
        y_test_adj_pred_pi = y_test_adj_pred + pi_test
        # make a array of 0 for redefined prediction of y
        y_pred = np.zeros(y_test_adj_pred_pi.shape, dtype=int)
        # assume that y_adj_pred_pi > 0.5 then pred_y is 1 unless it is 0
        y_pred[np.where(y_test_adj_pred_pi > 0.5)] = 1
        return y_pred
