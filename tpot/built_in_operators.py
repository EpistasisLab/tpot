# -*- coding: utf-8 -*-

"""Copyright 2015-Present Randal S. Olson.

This file is part of the TPOT library.

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
from sklearn.base import BaseEstimator
from sklearn.utils import check_array


class ZeroCount(BaseEstimator):
    """Adds the count of zeros and count of non-zeros per sample as features."""

    def fit(self, X, y=None):
        """Dummy function to fit in with the sklearn API."""
        return self

    def transform(self, X, y=None):
        """Transform data by adding two virtual features.

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

        non_zero = np.apply_along_axis(
            lambda row: np.count_nonzero(row),
            axis=1,
            arr=X_transformed
        )
        zero_col = np.apply_along_axis(
            lambda row: (n_features - np.count_nonzero(row)),
            axis=1,
            arr=X_transformed
        )

        X_transformed = np.insert(X_transformed, n_features, non_zero, axis=1)
        X_transformed = np.insert(X_transformed, n_features + 1, zero_col, axis=1)

        return X_transformed


class CombineDFs(object):
    """Combine two DataFrames."""

    @property
    def __name__(self):
        """Instance name is the same as the class name."""
        return self.__class__.__name__
