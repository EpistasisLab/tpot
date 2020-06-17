"""
AUTHORS
Weixuan Fu, Elisabetta Manduchi

SCOPE
Transforms features by residual covariate adjustment.
Can specify which features should be adjusted by which covariates.
Removes covariate columns, but retains indicator and adjY columns.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
import re

def X_adj_fit(X, C, col_type):
    """
    Transform X by a list of covariates.

    Parameters
    ----------
    X : pd.Series
    C: pd.DataFrame of covariates.
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
       # NEED TO MAKE PROVISIONS IN CASE OF NO CONVERGENCE

    elif col_type == 'logistic':
        est = LogisticRegression(penalty='none',
                                 solver='lbfgs',
                                 multi_class='auto',
                                 max_iter=500)
        est.fit(C, X_col.astype(np.int32))
       # NEED TO MAKE PROVISIONS IN CASE OF NO CONVERGENCE

    else:
        raise ValueError("Wrong column type! It should be 'logistic' or 'linear'!")
    return est

def X_adj_predict(X, C, col_type, est):
    """
    Transform X by a list of covariates.
    Parameters
    ----------
    X : pd.Series
    C: pd.DataFrame for a confounding covariate.
    col_type: 'logistic' or 'linear'
    est: estimator
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
        for gt_idx, gt in enumerate(est.classes_):
            gt = int(gt)
            X_adj = X_adj - gt*clf_pred_proba[:, gt_idx]
    else:
        raise ValueError("Wrong column type! It should be 'logistic' or 'linear'!")
    return X_adj.reshape(-1, 1)

class resAdjTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, C=None, adj_list=None):
        """
        Parameters
        ----------
        C:  a list of columns for C, e.g ['N13', 'N14', 'N15']
        columns of C correspond to covariates to adjust features by
        adj_list: a csv file with header row
        1st column (Feature) feature to adjust
        2nd column (Type) adjustment type: 'logistic' or 'linear'
        3rd column (Covariates) list of confounding covariates separated by ";"
        e.g.
        Feature,Type,Covariates
        N1,logistic,N13;N14;N15
        N2,logistic,N13
        N3,logistic,N13
        N4,logistic,N14;N15
        N5,logistic,N14;N15
        """

        self.C = C
        self.adj_list = adj_list

    def fit(self, X, y=None, **fit_params):
        """
        Fit the StackingEstimator meta-transformer

        Parameters
        ----------
        X : array-like
        y: None. Ignored variable.
        fit_params: other estimator-specific parameters

        Returns
        -------
        self: object
        Returns an estimator for each feature that needs to be adjusted
        """

        if self.C is None:
            raise ValueError('X is missing the covariate columns')
        if self.adj_list is None:
            raise ValueError('No adjustment information given')

        X_train = pd.DataFrame.copy(X)
        X_train.drop(self.C, axis=1, inplace=True)
        self.adj_df = pd.read_csv(self.adj_list)
        self.col_ests = {}
        for a in self.adj_df.Feature:
            if re.match(r'^indicator', a) or re.match(r'^adjY', a):
                raise ValueError("indicator and adjY columns of X should not be adjusted")
        self.comm_features = [a for a in self.adj_df.Feature if a in X_train.columns]
        if self.comm_features:
            self.sub_adj_df = self.adj_df[self.adj_df['Feature'].isin(self.comm_features)]
            for _, row in self.sub_adj_df.iterrows():
                subC = row['Covariates'].split(";")
                tmp_C = X[subC].values
                tmp_X = X_train[row['Feature']]
                est = X_adj_fit(tmp_X, tmp_C, row['Type'])
                self.col_ests[row['Feature']] = est

        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like
        """

        if self.C is None:
            raise ValueError('X is missing the covariate columns')
        if self.adj_list is None:
            raise ValueError('No adjustment information given')

        X_test = pd.DataFrame.copy(X)
        X_test.drop(self.C, axis=1, inplace=True)
        X_test_adj = X_test.values
        self.adj_df = pd.read_csv(self.adj_list)

        if self.comm_features:
            # features not to adjust

            X_test_unsel = X_test.drop(self.comm_features, axis=1)
            # features to adjust
            X_subset_adj = np.array([]) # make a empty array
            for _, row in self.sub_adj_df.iterrows():
                subC = row['Covariates'].split(";")
                tmp_C = X[subC].values
                tmp_X = X_test[row['Feature']]
                tmp_X_adj = X_adj_predict(tmp_X, tmp_C, row['Type'],
                                          self.col_ests[row['Feature']])
                if X_subset_adj.size == 0:
                    X_subset_adj = tmp_X_adj
                else:
                    X_subset_adj = np.hstack((X_subset_adj, tmp_X_adj))

            X_subset_adj = pd.DataFrame(X_subset_adj, index=X_test_unsel.index)
            X_test_adj = pd.concat([X_subset_adj, X_test_unsel], axis=1)
        return X_test_adj
