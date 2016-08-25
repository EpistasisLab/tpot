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
from sklearn.utils import check_array


class ZeroCount(BaseEstimator):

    """Preprocessor that adds two virtual features to the dataset, one for the count of zero values in the feature set, and one for the count of non-zeros in the feature set"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Dummy function to fit in with the sklearn API"""
        return self

    def transform(self, X, y=None):
        """Transform data by adding two virtual features

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components
            is the number of components.
        y: None
            Unused

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features)
            The transformed feature set
        """
        X = check_array(X)
        n_features = X.shape[1]

        X_transformed = np.copy(X)

        non_zero = np.apply_along_axis(lambda row: np.count_nonzero(row),
                                        axis=1, arr=X_transformed)
        zero_col = np.apply_along_axis(lambda row: (n_features - np.count_nonzero(row)),
                                        axis=1, arr=X_transformed)

        X_transformed = np.insert(X_transformed, n_features, non_zero, axis=1)
        X_transformed = np.insert(X_transformed, n_features + 1, zero_col, axis=1)

        return X_transformed


class TPOTZeroCount(Preprocessor):

    """Uses TPOT's ZeroCount to transform the feature set"""

    import_hash = {'tpot.operators.preprocessors': ['ZeroCount']}
    sklearn_class = ZeroCount
    arg_types = ()

    def __init__(self):
        """Creates a new TPOTZeroCount instance"""
        pass

    def preprocess_args(self):
        """Preprocesses the arguments in case they need to be constrained in some way"""
        return {}
