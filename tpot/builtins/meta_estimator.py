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

    def __init__(self, estimator, A=None, C=None, mode="classification"):
        """Create a StackingEstimator object.

        Parameters
        ----------
        estimator: object with fit, predict, and predict_proba methods.
            The estimator to generate synthetic features from.
        A: a list of columns for A, e.g ["N1", "N2"]
            columns of A correspond to a non-confounding covariate.
        C:  a list of columns for C, e.g ["N4", "N5"]
            columns of C correspond correspond to a confounding covariate.
        mode: strings
            use of MetaEstimator for classification or regression problem
        """
        self.estimator = estimator
        self.A = A
        self.C = C
        self.mode = mode


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
            raise(ValueError, "At least one of A_train and C_train must be specified")
        X_train = pd.DataFrame.copy(X)
        if self.A is not None:
            X_train.drop(self.A, axis=1, inplace=True)
        if self.C is not None:
            X_train.drop(self.C, axis=1, inplace=True)
        if self.C is None:
            X_train_adj = X_train
            C_train = None
        else:
            X_train_adj = np.zeros(X_train.shape)
            self.col_ests = [] # store estimator for each columns
            self.values_list = [] # store values for each columns
            C_train = X[self.C].values

            for col in range(X_train.shape[1]):
                X_train_col = X_train.iloc[:, col].values.reshape((-1, 1)) # np.ndarray

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
                    regr = LinearRegression()
                    regr.fit(C_train, X_train_col.reshape((-1,)))
                    est_pred = regr.predict(C_train).reshape((-1, 1))
                    X_train_adj[:, col:(col+1)] = X_train_col - est_pred
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
                    for gt_idx, gt in enumerate(clf.classes_):
                        gt = int(gt)
                        X_train_col_adj = X_train_col_adj - gt*clf_pred_proba[:, gt_idx:gt_idx+1]
                    X_train_adj[:, col:(col+1)] = X_train_col_adj
                    self.col_ests.append(clf)
        if self.A is not None:
            A_train = X[self.A].values


        if C_train is None and A_train is not None:
            B_train = A_train
        elif A_train is None and C_train is not None:
            B_train = C_train
        else:
            B_train = np.hstack((A_train, C_train))
        if self.mode == "classification":
            self.B_est = LogisticRegression(penalty='none', solver='lbfgs')
            self.B_est.fit(B_train, y)
            pi_train = np.ravel(self.B_est.predict_proba(B_train)[:, 1])
        elif self.mode == "regression":
            self.B_est = LinearRegression()
            self.B_est.fit(B_train, y)
            pi_train = np.ravel(self.B_est.predict(B_train).reshape((-1, 1)))
        else:
            raise(ValueError, "Unknown Mode! It should classification or regression")
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
            X_test.drop(self.A, axis=1, inplace=True)
        if self.C is not None:
            X_test.drop(self.C, axis=1, inplace=True)
        if self.C is None:
            X_test_adj = X_test
            C_test = None
        else:
            X_test_adj = np.zeros(X_test.shape)
            C_test = X[self.C].values
            for values, est, col in zip(self.values_list, self.col_ests, range(X_test.shape[1])):
                X_test_col = X_test.iloc[:, col].values.reshape((-1,1))
                if values == 'dosage':
                    est_pred = est.predict(C_test).reshape((-1, 1))
                    X_test_adj[:, col:(col+1)] = X_test_col - est_pred
                else:
                    clf_pred_proba = est.predict_proba(C_test)
                    X_test_col_adj = X_test_col
                    for gt_idx, gt in enumerate(est.classes_):
                        gt = int(gt)
                        X_test_col_adj = X_test_col_adj - gt*clf_pred_proba[:, gt_idx:gt_idx+1]
                    X_test_adj[:, col:(col+1)] = X_test_col_adj

        if self.A is not None:
            A_test = X[self.A].values
        if self.A is None and self.C is None:
            raise(ValueError, "At least one of A_train and C_train must be specified")
        elif C_test is None and A_test is not None:
            B_test = A_test
        elif A_test is None and C_test is not None:
            B_test = C_test
        else:
            B_test = np.hstack((A_test, C_test))
        if self.mode == "classification":
            pi_test = np.ravel(self.B_est.predict_proba(B_test)[:, 1])
        elif self.mode == "regression":
            pi_test = np.ravel(self.B_est.predict(B_test).reshape((-1, 1)))
        y_test_adj_pred = self.estimator.predict(X_test_adj)
        y_test_adj_pred_pi = y_test_adj_pred + pi_test
        # make a array of 0 for redefined prediction of y
        y_pred = np.zeros(y_test_adj_pred_pi.shape, dtype=int)
        # assume that y_adj_pred_pi > 0.5 then pred_y is 1 unless it is 0
        y_pred[np.where(y_test_adj_pred_pi > 0.5)] = 1
        return y_pred
'''
def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y if it is a classification problem.
            R^2 of self.predict(X) wrt. y. is it is a regression problem.
        """
        if self.mode == "classification":

            from sklearn.metrics import accuracy_score
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        elif self.mode == "regression":

            from sklearn.metrics import r2_score
            from sklearn.metrics._regression import _check_reg_targets
            y_pred = self.predict(X)
            # XXX: Remove the check in 0.23
            y_type, _, _, _ = _check_reg_targets(y, y_pred, None)
            if y_type == 'continuous-multioutput':
                warnings.warn("The default value of multioutput (not exposed in "
                              "score method) will change from 'variance_weighted' "
                              "to 'uniform_average' in 0.23 to keep consistent "
                              "with 'metrics.r2_score'. To specify the default "
                              "value manually and avoid the warning, please "
                              "either call 'metrics.r2_score' directly or make a "
                              "custom scorer with 'metrics.make_scorer' (the "
                              "built-in scorer 'r2' uses "
                              "multioutput='uniform_average').", FutureWarning)
            return r2_score(y, y_pred, sample_weight=sample_weight,
                            multioutput='variance_weighted')
'''

class MetaRegressor(BaseEstimator, RegressorMixin):
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
        A: a list of columns for A, e.g ["N1", "N2"]
            columns of A correspond to a non-confounding covariate.
        C:  a list of columns for C, e.g ["N4", "N5"]
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
            raise(ValueError, "At least one of A_train and C_train must be specified")
        X_train = pd.DataFrame.copy(X)
        if self.A is not None:
            X_train.drop(self.A, axis=1, inplace=True)
        if self.C is not None:
            X_train.drop(self.C, axis=1, inplace=True)
        if self.C is None:
            X_train_adj = X_train
            C_train = None
        else:
            X_train_adj = np.zeros(X_train.shape)
            self.col_ests = [] # store estimator for each columns
            self.values_list = [] # store values for each columns
            C_train = X[self.C].values

            for col in range(X_train.shape[1]):
                X_train_col = X_train.iloc[:, col].values.reshape((-1, 1)) # np.ndarray

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
                    regr = LinearRegression()
                    regr.fit(C_train, X_train_col.reshape((-1,)))
                    est_pred = regr.predict(C_train).reshape((-1, 1))
                    X_train_adj[:, col:(col+1)] = X_train_col - est_pred
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
                    for gt_idx, gt in enumerate(clf.classes_):
                        gt = int(gt)
                        X_train_col_adj = X_train_col_adj - gt*clf_pred_proba[:, gt_idx:gt_idx+1]
                    X_train_adj[:, col:(col+1)] = X_train_col_adj
                    self.col_ests.append(clf)
        if self.A is not None:
            A_train = X[self.A].values


        if C_train is None and A_train is not None:
            B_train = A_train
        elif A_train is None and C_train is not None:
            B_train = C_train
        else:
            B_train = np.hstack((A_train, C_train))
        self.B_est = LinearRegression()
        self.B_est.fit(B_train, y)
        pi_train = np.ravel(self.B_est.predict(B_train).reshape((-1, 1)))

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
            X_test.drop(self.A, axis=1, inplace=True)
        if self.C is not None:
            X_test.drop(self.C, axis=1, inplace=True)
        if self.C is None:
            X_test_adj = X_test
            C_test = None
        else:
            X_test_adj = np.zeros(X_test.shape)
            C_test = X[self.C].values
            for values, est, col in zip(self.values_list, self.col_ests, range(X_test.shape[1])):
                X_test_col = X_test.iloc[:, col].values.reshape((-1,1))
                if values == 'dosage':
                    est_pred = est.predict(C_test).reshape((-1, 1))
                    X_test_adj[:, col:(col+1)] = X_test_col - est_pred
                else:
                    clf_pred_proba = est.predict_proba(C_test)
                    X_test_col_adj = X_test_col
                    for gt_idx, gt in enumerate(est.classes_):
                        gt = int(gt)
                        X_test_col_adj = X_test_col_adj - gt*clf_pred_proba[:, gt_idx:gt_idx+1]
                    X_test_adj[:, col:(col+1)] = X_test_col_adj

        if self.A is not None:
            A_test = X[self.A].values
        if self.A is None and self.C is None:
            raise(ValueError, "At least one of A_train and C_train must be specified")
        elif C_test is None and A_test is not None:
            B_test = A_test
        elif A_test is None and C_test is not None:
            B_test = C_test
        else:
            B_test = np.hstack((A_test, C_test))

        pi_test = np.ravel(self.B_est.predict(B_test).reshape((-1, 1)))
        y_test_adj_pred = self.estimator.predict(X_test_adj)
        y_test_adj_pred_pi = y_test_adj_pred + pi_test
        # make a array of 0 for redefined prediction of y
        y_pred = np.zeros(y_test_adj_pred_pi.shape, dtype=int)
        # assume that y_adj_pred_pi > 0.5 then pred_y is 1 unless it is 0
        y_pred[np.where(y_test_adj_pred_pi > 0.5)] = 1
        return y_pred
