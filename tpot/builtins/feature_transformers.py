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
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

#from .one_hot_encoder import OneHotEncoder

def auto_select_categorical_features(X, threshold=10):
    """Make a feature mask of categorical features in X.

    Features with less than 10 unique values are considered categorical.

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Dense array or sparse matrix.

    threshold : int
        Maximum number of unique values per feature to consider the feature
        to be categorical.

    Returns
    -------
    feature_mask : array of booleans of size {n_features, }
    """
    feature_mask = []

    for column in range(X.shape[1]):
        if sparse.issparse(X):
            indptr_start = X.indptr[column]
            indptr_end = X.indptr[column + 1]
            unique = np.unique(X.data[indptr_start:indptr_end])
        else:
            unique = np.unique(X[:, column])

        feature_mask.append(len(unique) <= threshold)

    return feature_mask

class CategoricalSelector(BaseEstimator, TransformerMixin):
    """Meta-transformer for selecting categorical features and transform them using OneHotEncoder.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    """

    def __init__(self, threshold):
        """Create a StackingEstimator object.

        Parameters
        ----------

        threshold : int, default=10
            Maximum number of unique values per feature to consider the feature
            to be categorical.
        """
        threshold=self.threshold

    def fit(self, X, y=None):
        """Fit CategoricalSelector to X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_feature)
            Input array of type int.

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def transform(self, X):
        """Transform data by adding two synthetic feature(s).

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features + 1) or (n_samples, n_features + 1 + n_classes) for classifier with predict_proba attribute
            The transformed feature set.
        """

        X = check_array(X)
        X_transformed = np.copy(X)
        categorical_features = auto_select_categorical_features(X, threshold=self.threshold)

        return X_transformed
