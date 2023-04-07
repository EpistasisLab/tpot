# -*- coding: utf-8 -*-

"""Copyright (c) 2015 The auto-sklearn developers. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the auto-sklearn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""
#TODO support np arrays

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import OneHotEncoder
import sklearn
import sklearn.impute

import pandas as pd
from pandas.api.types import is_numeric_dtype



def auto_select_categorical_features(X):

    if not isinstance(X, pd.DataFrame):
        return []
    
    feature_mask = []
    for column in X.columns:
        feature_mask.append(is_numeric_dtype(X[column]))

    return feature_mask


def _X_selected(X, selected):
    """Split X into selected features and other features"""
    X_sel = X[X.columns[selected]]
    X_not_sel = X.drop(X.columns[selected], axis=1)
    return X_sel, X_not_sel



class NumericImpute(BaseEstimator, TransformerMixin):


    def __init__(self, categorical_features='auto'):
        self.categorical_features = categorical_features


    def fit(self, X, y=None):
        """Fit OneHotEncoder to X, then transform X.

        Equivalent to self.fit(X).transform(X), but more convenient and more
        efficient. See fit for the parameters, transform for the return value.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Dense array or sparse matrix.
        y: array-like {n_samples,} (Optional, ignored)
            Feature labels
        """
        
        if self.categorical_features == "auto":
            self.categorical_features_ = auto_select_categorical_features(X)

        #TODO make this more consistent with sklearn baseimputer/baseencoder
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                # check if the column name is not a string
                if not isinstance(col, str):
                    # if it's not a string, rename the column with "X" prefix
                    X.rename(columns={col: f"X{col}"}, inplace=True)

        if sum(self.categorical_features_) == 0:
            return self
        
        self.enc = sklearn.impute.SimpleImputer(strategy='most_frequent')
        if isinstance(X, pd.DataFrame):
            self.enc.set_output(transform="pandas")

        if sum(self.categorical_features_) == X.shape[1]:
            X_sel = self.enc.fit(X)
        else:
            X_sel, X_not_sel = _X_selected(X, self.categorical_features_)
            X_sel = self.enc.fit(X_sel)
        
        return self
  
    def transform(self, X):
        """Transform X using one-hot encoding.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Dense array or sparse matrix.

        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array, dtype=int
            Transformed input.
        """

    
        if sum(self.categorical_features_) == 0:
            return X
        
        #TODO make this more consistent with sklearn baseimputer/baseencoder
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                # check if the column name is not a string
                if not isinstance(col, str):
                    # if it's not a string, rename the column with "X" prefix
                    X.rename(columns={col: f"X{col}"}, inplace=True)

        if sum(self.categorical_features_) == X.shape[1]:
            return self.enc.transform(X)
        else:

            X_sel, X_not_sel= _X_selected(X, self.categorical_features_)
            X_sel = self.enc.transform(X_sel)
            
            #If X is dataframe
            if isinstance(X, pd.DataFrame):
            
                X_sel = pd.DataFrame(X_sel, columns=self.enc.get_feature_names_out())
                return pd.concat([X_not_sel.reset_index(drop=True), X_sel.reset_index(drop=True)], axis=1)
            else:
                return np.hstack((X_not_sel, X_sel))
