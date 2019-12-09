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
        A: a string for a list of columns delimited by ";" for A, e.g "N1;N2"
            columns of A correspond to a non-confounding covariate.
        C: a string for a list of columns delimited by ";" for C, e.g "N4;N5"
            columns of C correspond correspond to a confounding covariate.

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
        if self.A is None and self.C is None:
            rasie(ValueError, "At least one of A_train and C_train must be specified")
        X_train = pd.DataFrame.copy(X)
        if self.A is not None:
            self.A_list = self.A.split(';')
            X_train.drop(self.A_list, axis=1, inplace=True)
        if self.C is not None:
            self.C_list = self.C.split(';')
            X_train.drop(self.C_list, axis=1, inplace=True)
        if self.C is None:
            X_train_adj = X_train
            C_train = None
        else:
            X_train_adj = np.zeros(X_train.shape)
            self.col_ests = [] # store estimator for each columns
            self.values_list = [] # store values for each columns
            C_train = X[self.C_list].values

            for col in range(X_train.shape[1]):
                X_train_col = X_train.iloc[:, col].values.reshape((-1,1)) # np.ndarray

                # test information cannot be used in fit() function
                # may be values should be provided as a parameter in __init__ above
                # here values was stored into self.values_list and can be used in predict
                # function below for test dataset !!! need pre_provide
                dosage = np.hstack((X_train_col!=0, X_train_col!=1, X_train_col!=2))
                if X_train_col[np.all(dosage, axis=1).reshape((-1, 1))].shape[0]>0:
                    values = 'dosage'
                else:
                    values = 'ternary'
                self.values_list.append(values)
                if values == 'dosage':
                    regr = LinearRegression(),
                    regr.fit(C_train, X_train_col.reshape((-1,)))
                    X_train_adj[:, col:(col+1)] = X_train_col - regr.predict(C_train)
                    self.col_ests.append(regr)
                else:
                    clf = LogisticRegression(penalty='none',
                                            solver='lbfgs',
                                            multi_class='auto')
                    clf.fit(C_train, X_train_col.reshape((-1,)).astype(np.int32))
                    clf_pred_proba = clf.predict_proba(C_train)

                    X_train_col_adj = X_train_col
                    # clf.classes_ should return an array of genotypes in this column
                    # like array([0, 1, 2]) or array([0, 1])
                    for gt in clf.classes_:
                        gt = int(gt)
                        X_train_col_adj = X_train_col_adj - gt*clf_pred_proba[:, gt:gt+1]
                    X_train_adj[:, col:(col+1)] = X_train_col_adj
                    self.col_ests.append(clf)
        if self.A is not None:
            A_train = X[self.A_list].values


        if C_train is None and A_train is not None:
            B_train = A_train
        elif A_train is None and C_train is not None:
            B_train = C_train
        else:
            B_train = np.hstack((A_train, C_train))

        self.B_clf = LogisticRegression(penalty='none', solver='lbfgs')
        self.B_clf.fit(B_train, y)
        pi_train = np.ravel(self.B_clf.predict_proba(B_train)[:, 1])
        y_train_adj = y - pi_train

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
        X_test = pd.DataFrame.copy(X)
        if self.A is not None:
            X_test.drop(self.A_list, axis=1, inplace=True)
        if self.C is not None:
            X_test.drop(self.C_list, axis=1, inplace=True)
        if self.C is None:
            X_test_adj = X_test
            C_test = None
        else:
            X_test_adj = np.zeros(X_test.shape)
            C_test = X[self.C_list].values
            for values, est, col in zip(self.values_list, self.col_ests, range(X_test.shape[1])):
                X_test_col = X_test.iloc[:, col].values.reshape((-1,1))
                if values == 'dosage':
                    X_test_adj[:, col:(col+1)] = X_test_adj - est.predict(C_test)
                else:
                    clf_pred_proba = est.predict_proba(C_test)
                    X_test_col_adj = X_test_col
                    for gt in est.classes_:
                        X_test_col_adj = X_test_col_adj - gt*clf_pred_proba[:, gt:gt+1]
                    X_test_adj[:, col:(col+1)] = X_test_col_adj

        if self.A is not None:
            A_test = X[self.A_list].values
        if self.A is None and self.C is None:
            rasie(ValueError, "At least one of A_train and C_train must be specified")
        elif C_test is None and A_test is not None:
            B_test = A_test
        elif A_test is None and C_test is not None:
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
