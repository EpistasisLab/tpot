"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Gabriel Ketron (gabriel.ketron@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

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


class ColumnSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self,  columns="all",         
                        missing_values=np.nan,
                        strategy="mean",
                        fill_value=None,
                        copy=True,
                        add_indicator=False,
                        keep_empty_features=False,):
        """"
        A wrapper for SimpleImputer that allows for imputation of specific columns in a DataFrame or np array.
        Passes through columns that are not imputed.

        Parameters
        ----------
        columns : str, list, default='all'
            Determines which columns to impute with sklearn.impute.SimpleImputer.
            - 'categorical' : Automatically select categorical features
            - 'numeric' : Automatically select numeric features
            - 'all' : Select all features
            - list : A list of columns to select

        # See documentation from sklearn.impute.SimpleImputer for the following parameters
        missing_values, strategy, fill_value, copy, add_indicator, keep_empty_features
               
        """
        
        self.columns = columns
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features


    def fit(self, X, y=None):
        if (self.columns == "categorical" or self.columns == "numeric") and not isinstance(X, pd.DataFrame):
            raise ValueError(f"Invalid value for columns: {self.columns}. "
                             "Only 'all' or <list> is supported for np arrays")

        if self.columns == "categorical":
            self.columns_ = list(X.select_dtypes(exclude='number').columns)
        elif self.columns == "numeric":
            self.columns_ =  [col for col in X.columns if is_numeric_dtype(X[col])]
        elif self.columns == "all":
            if isinstance(X, pd.DataFrame):
                self.columns_ = X.columns
            else:
                self.columns_ = list(range(X.shape[1]))
        elif isinstance(self.columns, list):
            self.columns_ = self.columns
        else:
            raise ValueError(f"Invalid value for columns: {self.columns}")
        
        if len(self.columns_) == 0:
            return self
        
        self.imputer = sklearn.impute.SimpleImputer(missing_values=self.missing_values,
                                                    strategy=self.strategy,
                                                    fill_value=self.fill_value,
                                                    copy=self.copy,
                                                    add_indicator=self.add_indicator,
                                                    keep_empty_features=self.keep_empty_features)
        
        if isinstance(X, pd.DataFrame):
            self.imputer.set_output(transform="pandas")

        if isinstance(X, pd.DataFrame):
            self.imputer.fit(X[self.columns_], y)
        else:
            self.imputer.fit(X[:, self.columns_], y)

        return self

    def transform(self, X):
        if len(self.columns_) == 0:
            return X

        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X[self.columns_] = self.imputer.transform(X[self.columns_])
            return X
        else:
            X = np.copy(X)
            X[:, self.columns_] = self.imputer.transform(X[:, self.columns_])
            return X


