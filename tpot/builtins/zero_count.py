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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


class ZeroCount(BaseEstimator, TransformerMixin):
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

        non_zero_vector = np.count_nonzero(X_transformed, axis=1)
        non_zero = np.reshape(non_zero_vector, (-1, 1))
        zero_col = np.reshape(n_features - non_zero_vector, (-1, 1))

        X_transformed = np.hstack((non_zero, X_transformed))
        X_transformed = np.hstack((zero_col, X_transformed))

        return X_transformed
