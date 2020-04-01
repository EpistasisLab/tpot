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


def X_adj_fit(X, C, col_type):
    """transform X by a list of confounding features.

    Parameters
    ----------
    X : pd.Series
    C: pd.DataFrame for a confounding covariate.
    col_type: 'logistic' or 'linear'

    Return
    ----------
    X_adj: transformed/adjusted X
    est: estimator for a column
    """
    X_col = X.values # 1 D np.ndarray
    if col_type == 'linear':
        est = LinearRegression()
        est.fit(C, X_col)
        est_pred = est.predict(C)
        X_adj = X_col - est_pred

    elif col_type == 'logistic':
        est = LogisticRegression(penalty='none',
                                solver='lbfgs',
                                multi_class='auto')
        est.fit(C, X_col.astype(np.int32))
        clf_pred_proba = est.predict_proba(C)
        X_adj = X_col
        # clf.classes_ should return an array of genotypes in this column
        # like array([0, 1, 2]) or array([0, 1])
        for gt_idx, gt in enumerate(est.classes_):
            gt = int(gt)
            X_adj = X_adj - gt*clf_pred_proba[:, gt_idx]
    else:
        raise(ValueError, "Wrong column type! It should be 'logistic' or 'linear'!")
    return X_adj.reshape(-1, 1), est

def X_adj_predict(X, C, col_type, est):
    """transform X by a list of confounding features.

    Parameters
    ----------
    X : pd.Series
    C: pd.DataFrame for a confounding covariate.
    col_type: 'logistic' or 'linear'
    est: estimator for a column

    Return
    ----------
    X_adj: transformed/adjusted X

    """
    X_col = X.values # 1 D np.ndarray
    if col_type == 'linear':
        est_pred = est.predict(C)
        X_adj = X_col - est_pred

    elif col_type == 'logistic':
        clf_pred_proba = est.predict_proba(C)
        X_adj = X_col
        # clf.classes_ should return an array of genotypes in this column
        # like array([0, 1, 2]) or array([0, 1])
        for gt_idx, gt in enumerate(est.classes_):
            gt = int(gt)
            X_adj = X_adj - gt*clf_pred_proba[:, gt_idx]
    else:
        raise(ValueError, "Wroing column type! "
                        "It should be 'logistic' or 'linear'!")
    return X_adj.reshape(-1, 1)


class MetaEstimator(BaseEstimator):
    """Meta-transformer for adding predictions and/or class probabilities as synthetic feature(s).

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    """


    def __init__(self, estimator, A=None, C=None, adj_list=None, mode="classification"):
        """Create a StackingEstimator object.

        Parameters
        ----------
        estimator: object with fit, predict, and predict_proba methods.
            The estimator to generate synthetic features from.
        A: a list of columns for A, e.g ["N1", "N2"]
            columns of A correspond to a non-confounding covariate.
        C:  a list of columns for C, e.g ["N4", "N5"]
            columns of C correspond to a confounding covariate.
        adj_list: a csv file with header row
            1st column lists features for adjustment
            2nd column is feature type, logistic or linear
            3rd column is a list of confounding covariates separate by ";"
            e.g.
            Feature,Type,Coavariates
            N1,logistic,N13;N14;N15
            N2,logistic,N13
            N3,logistic,N13
            N4,logistic,N14;N15
            N5,logistic,N14;N15
        mode: string
            "classification" or "regression"
        """
        self.estimator = estimator
        self.A = A
        self.C = C
        self.adj_list = adj_list
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
        if self.mode == "classification":
            self.B_est = LogisticRegression(penalty='none',
                                            solver='lbfgs',
                                            multi_class='auto')
        elif self.mode == "regression":
            self.B_est = LinearRegression()
        else:
            raise(ValueError, "Wrong mode!")

        if self.A is None and self.C is None:
            raise(ValueError, "At least one of A_train and C_train must be specified")
        X_train = pd.DataFrame.copy(X)
        if self.A is not None:
            A_train = X[self.A].values
            X_train.drop(self.A, axis=1, inplace=True)
        if self.C is None:
            X_train_adj = X_train.values
            B_train = A_train
        else:
            C_train = X[self.C].values
            if self.A is not None:
                B_train = np.hstack((A_train, C_train))
            else:
                B_train = C_train
            X_train.drop(self.C, axis=1, inplace=True)
            X_train_adj = X_train.values
            self.col_ests = {}
            if self.adj_list is not None: # if None then no adjustment
                self.adj_df = pd.read_csv(self.adj_list)
                # overlap features
                self.comm_features = [a for a in self.adj_df.Feature if a in X_train.columns]
                if self.comm_features:
                    X_train_unsel = X_train.drop(self.comm_features, axis=1).values
                    X_subset_adj = np.array([]) # make a empty array
                    self.sub_adj_df = self.adj_df[self.adj_df['Feature'].isin(self.comm_features)]
                    for _, row in self.sub_adj_df.iterrows():
                        subC = row['Coavariates'].split(";")
                        tmp_C = X[subC].values
                        tmp_X = X_train[row['Feature']]
                        tmp_X_adj, est = X_adj_fit(tmp_X, tmp_C, row['Type'])
                        self.col_ests[row['Feature']] = est
                        if X_subset_adj.size == 0:
                            X_subset_adj = tmp_X_adj
                        else:
                            X_subset_adj = np.hstack((X_subset_adj, tmp_X_adj))
                    X_train_adj = np.hstack((X_subset_adj, X_train_unsel))

        self.B_est.fit(B_train, y)
        pi_train = self.adj_function(B_train)
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
            A_test = X[self.A].values
            X_test.drop(self.A, axis=1, inplace=True)
        if self.C is None:
            X_test_adj = X_test.values
            B_test = A_test
        else:
            C_test = X[self.C].values
            if self.A is not None:
                B_test = np.hstack((A_test, C_test))
            else:
                B_test = C_test
            X_test.drop(self.C, axis=1, inplace=True)
            X_test_adj = X_test.values
            if self.adj_list is not None: # if None then no adjustment
                if self.comm_features:
                    # features without adjustment
                    X_test_unsel = X_test.drop(self.comm_features, axis=1).values
                    # features for adjustment
                    X_subset_adj = np.array([]) # make a empty array
                    for _, row in self.sub_adj_df.iterrows():
                        subC = row['Coavariates'].split(";")
                        tmp_C = X[subC].values
                        tmp_X = X_test[row['Feature']]
                        tmp_X_adj = X_adj_predict(tmp_X, tmp_C,
                                                row['Type'],
                                                self.col_ests[row['Feature']])

                        if X_subset_adj.size == 0:
                            X_subset_adj = tmp_X_adj
                        else:
                            X_subset_adj = np.hstack((X_subset_adj, tmp_X_adj))
                    X_test_adj = np.hstack((X_subset_adj, X_test_unsel))


        y_test_adj_pred = self.estimator.predict(X_test_adj)
        pi_test = self.adj_function(B_test)
        y_test_adj_pred_pi = y_test_adj_pred + pi_test
        return self.recode_function(y_test_adj_pred_pi)


    def adj_function(self, B):
        """Adjust y based on different application.
        This is for classification
        """
        if self.mode == "classification":
            B_est_pred_proba = self.B_est.predict_proba(B)
            pi = np.zeros((B_est_pred_proba.shape[0], ))
            for gt_idx, gt in enumerate(self.B_est.classes_):
                gt = int(gt)
                pi = pi + gt*B_est_pred_proba[:, gt_idx]
        else:
            pi = self.B_est.predict(B)
        return pi


    def recode_function(self, y_test_adj_pred_pi):
        """recode y for prediction"""
        # define min. max value in B_est.classes_
        if self.mode == "classification":
            min_c, max_c = min(self.B_est.classes_), max(self.B_est.classes_)
            y_pred = np.rint(y_test_adj_pred_pi)
            y_pred[np.where(y_test_adj_pred_pi<min_c)] = min_c
            y_pred[np.where(y_test_adj_pred_pi>max_c)] = max_c
        else:
            y_pred = y_test_adj_pred_pi
        return y_pred

class MetaClassifier(MetaEstimator, ClassifierMixin):
    mode = "classification"

class MetaRegressor(MetaEstimator, RegressorMixin):
    mode = "regression"
