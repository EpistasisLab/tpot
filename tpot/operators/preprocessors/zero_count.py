# -*- coding: utf-8 -*-

"""
Copyright 2016 Randal S. Olson

This file is part of the TPOT library.

The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along
with the TPOT library. If not, see http://www.gnu.org/licenses/.

"""

import numpy as np

from .base import Preprocessor
from sklearn.base import BaseEstimator


class ZeroCount(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, input_df):
        feature_cols_only = input_df.drop(self.non_feature_columns, axis=1)
        num_features = len(feature_cols_only.columns.values)

        modified_df = input_df.copy()
        modified_df['non_zero'] = feature_cols_only.\
            apply(lambda row: np.count_nonzero(row), axis=1).astype(np.float64)
        modified_df['zero_col'] = feature_cols_only.\
            apply(lambda row: (num_features - np.count_nonzero(row)), axis=1).astype(np.float64)

        return modified_df.copy()


class TPOTZeroCount(Preprocessor):
    """Uses scikit-learn's ======== to transform the feature set

    Parameters
    ----------
    None

    """
    import_hash = {'tpot.operators.preprocessors': ['ZeroCount']}
    sklearn_class = ZeroCount

    def __init__(self):
        pass

    def preprocess_args(self):
        return {}
