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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

from .one_hot_encoder import OneHotEncoder, auto_select_categorical_features


class CategoricalSelector(BaseEstimator, TransformerMixin):
    """Meta-transformer for selecting categorical features and transform them using OneHotEncoder.

    Parameters
    ----------

    threshold : int, default=10
        Maximum number of unique values per feature to consider the feature
        to be categorical.

    minimum_fraction: float, default=None
        Minimum fraction of unique values in a feature to consider the feature
        to be categorical.
    """

    def __init__(self, threshold=10, minimum_fraction=None):
        """Create a CategoricalSelector object."""
        self.threshold=threshold
        self.minimum_fraction = minimum_fraction


    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like
        """
        X = check_array(X, accept_sparse='csr')
        return self


    def transform(self, X):
        """Select categorical features and transform them using OneHotEncoder.

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        X_transformed: array-like, {n_samples, n_components}
        """
        selected = auto_select_categorical_features(X, threshold=self.threshold)
        n_features = X.shape[1]
        ind = np.arange(n_features)
        sel = np.zeros(n_features, dtype=bool)
        sel[np.asarray(selected)] = True
        X_sel = X[:, ind[sel]]
        ohe = OneHotEncoder(categorical_features='all', sparse=False, minimum_fraction=self.minimum_fraction)

        return ohe.fit_transform(X_sel)
