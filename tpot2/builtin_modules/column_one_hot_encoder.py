import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import OneHotEncoder
import sklearn

import pandas as pd
from pandas.api.types import is_numeric_dtype



def auto_select_categorical_features(X, min_unique=10,):

    if isinstance(X, pd.DataFrame):
        return [col for col in X.columns if len(X[col].unique()) < min_unique]
    else:
        return [i for i in range(X.shape[1]) if len(np.unique(X[:, i])) < min_unique]



def _X_selected(X, selected):
    """Split X into selected features and other features"""
    
    if isinstance(X, pd.DataFrame):
        X_sel = X[selected]
        X_not_sel = X.drop(selected, axis=1)
    else:
        X_sel = X[:, selected]
        X_not_sel = np.delete(X, selected, axis=1)

    return X_sel, X_not_sel



class ColumnOneHotEncoder(BaseEstimator, TransformerMixin):


    def __init__(self, columns='auto', drop=None, handle_unknown='infrequent_if_exist', sparse_output=False, min_frequency=None,max_categories=None):
        '''
        
        Parameters
        ----------

        columns : str, list, default='auto'
            Determines which columns to onehot encode with sklearn.preprocessing.OneHotEncoder.
            - 'auto' : Automatically select categorical features based on columns with less than 10 unique values
            - 'categorical' : Automatically select categorical features
            - 'numeric' : Automatically select numeric features
            - 'all' : Select all features
            - list : A list of columns to select
        
        drop, handle_unknown, sparse_output, min_frequency, max_categories : see sklearn.preprocessing.OneHotEncoder

        '''

        self.columns = columns
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.min_frequency = min_frequency
        self.max_categories = max_categories



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
        
        if (self.columns == "categorical" or self.columns == "numeric") and not isinstance(X, pd.DataFrame):
            raise ValueError(f"Invalid value for columns: {self.columns}. "
                             "Only 'all' or <list> is supported for np arrays")

        if self.columns == "categorical":
            self.columns_ = list(X.select_dtypes(exclude='number').columns)
        elif self.columns == "numeric":
            self.columns_ =  [col for col in X.columns if is_numeric_dtype(X[col])]
        elif self.columns == "auto":
            self.columns_ = auto_select_categorical_features(X)
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
        
        self.enc = sklearn.preprocessing.OneHotEncoder( categories='auto',   
                                                        drop = self.drop,
                                                        handle_unknown = self.handle_unknown,
                                                        sparse_output = self.sparse_output,
                                                        min_frequency = self.min_frequency,
                                                        max_categories = self.max_categories)

        #TODO make this more consistent with sklearn baseimputer/baseencoder
        if isinstance(X, pd.DataFrame):
            self.enc.set_output(transform="pandas")
            for col in X.columns:
                # check if the column name is not a string
                if not isinstance(col, str):
                    # if it's not a string, rename the column with "X" prefix
                    X.rename(columns={col: f"X{col}"}, inplace=True)


        if len(self.columns_) == X.shape[1]:
            X_sel = self.enc.fit(X)
        else:
            X_sel, X_not_sel = _X_selected(X, self.columns_)
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

    
        if len(self.columns_) == 0:
            return X
        
        #TODO make this more consistent with sklearn baseimputer/baseencoder
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                # check if the column name is not a string
                if not isinstance(col, str):
                    # if it's not a string, rename the column with "X" prefix
                    X.rename(columns={col: f"X{col}"}, inplace=True)

        if len(self.columns_) == X.shape[1]:
            return self.enc.transform(X)
        else:

            X_sel, X_not_sel= _X_selected(X, self.columns_)
            X_sel = self.enc.transform(X_sel)
            
            #If X is dataframe
            if isinstance(X, pd.DataFrame):
            
                X_sel = pd.DataFrame(X_sel, columns=self.enc.get_feature_names_out())
                return pd.concat([X_not_sel.reset_index(drop=True), X_sel.reset_index(drop=True)], axis=1)
            else:
                return np.hstack((X_not_sel, X_sel))
