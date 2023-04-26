# %%
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
import sklearn.compose


def auto_select_categorical_features(X):

    if not isinstance(X, pd.DataFrame):
        return []
    
    feature_mask = []
    for column in X.columns:
        feature_mask.append(not is_numeric_dtype(X[column]))

    return feature_mask


def get_mask_from_categorical_features(X, categorical_features):
    """
    Params 
    ------
    X: pd.DataFrame or np.array
        Dataframe to be processed
    categorical_features: list
        List of categorical features. If X is a dataframe, this should be a list of column names. If X is a numpy array, this should be a list of column indices

    Returns
    -------
    mask: list
        List of booleans indicating which columns are categorical

    """

    if isinstance(X, pd.DataFrame):
        mask = [col in categorical_features for col in X.columns]
    elif isinstance(X, np.ndarray):
        mask = [i in categorical_features for i in range(X.shape[1])]
    else:
        raise TypeError("X must be either a pandas DataFrame or a numpy array")
    return mask



class ColumnImputer(BaseEstimator, TransformerMixin):
    def __init__(self,  columns,         
                        missing_values=np.nan,
                        strategy="mean",
                        fill_value=None,
                        copy=True,
                        add_indicator=False,
                        keep_empty_features=False,):
        
        self.columns = columns
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features


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
        
        self.imputer = sklearn.impute.SimpleImputer(missing_values=self.missing_values,
                                                    strategy=self.strategy,
                                                    fill_value=self.fill_value,
                                                    copy=self.copy,
                                                    add_indicator=self.add_indicator,
                                                    keep_empty_features=self.keep_empty_features)
        
        if isinstance(X, pd.DataFrame):
            self.imputer.set_output(transform="pandas")

        self.imputer_column_transformer = sklearn.compose.ColumnTransformer(transformers=[(self.strategy, self.imputer, self.columns)], remainder='passthrough') 

        if isinstance(X, pd.DataFrame):
            self.imputer_column_transformer.set_output(transform="pandas")

        return self.imputer_column_transformer.fit(X, y)

  
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
        return self.imputer_column_transformer.transform(X)


class AutoImputer(BaseEstimator, TransformerMixin):


    def __init__(self, impute_type="categorical", categorical_features='auto', numeric_features='auto', categorical_strategy='most_frequent', numeric_strategy='median'):
        """
        Params
        ------
        categorical_features: list or 'auto'
            List of categorical features. If 'auto', will automatically select categorical features based on pandas dtype
        impute_type: str
            Type of imputation to perform. Either 'categorical' or 'numeric'
            - 'categorical': impute categorical features only
            - 'numeric': impute numeric features only
            - 'all': impute all features
        """
        
        self.impute_type = impute_type
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.categorical_strategy = categorical_strategy
        self.numeric_strategy = numeric_strategy
    
    def fit(self, X, y=None):

        if self.categorical_features == "auto":
            self.categorical_features_ = auto_select_categorical_features(X)
        else:
            self.categorical_features_ = get_mask_from_categorical_features(X, self.categorical_features)

        if self.numeric_features == "auto":
            self.numeric_features_ = ~np.array(self.categorical_features_)
        else:
            self.numeric_features_ = ~np.array(get_mask_from_categorical_features(X, self.numeric_features))
        
        if self.impute_type == "categorical":
            self.imputer_ = ColumnImputer(columns=self.categorical_features_, strategy=self.categorical_strategy)
        elif self.impute_type == "numeric":
            self.imputer_ = ColumnImputer(columns=self.numeric_features_, strategy=self.numeric_strategy)
        elif self.impute_type == "all":
            #make pipeline of categorical then numeric

            self.cat_imputer = sklearn.impute.SimpleImputer(strategy=self.categorical_strategy)
            if isinstance(X, pd.DataFrame):
                self.cat_imputer.set_output(transform="pandas")
            self.numeric_imputer = sklearn.impute.SimpleImputer(strategy=self.numeric_strategy)
            if isinstance(X, pd.DataFrame):
                self.numeric_imputer.set_output(transform="pandas")


            self.imputer_ = sklearn.compose.ColumnTransformer(transformers=[(self.categorical_strategy, self.cat_imputer, self.categorical_features_),
                                                                            (self.numeric_strategy, self.numeric_imputer, self.numeric_features_)])
            
            
            if isinstance(X, pd.DataFrame):
                self.imputer_.set_output(transform="pandas")

        return self.imputer_.fit(X, y)
    
    def transform(self, X):
        return self.imputer_.transform(X)

