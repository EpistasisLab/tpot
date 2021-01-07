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
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.decomposition import PCA
from sklearn.compose import make_column_transformer

from .one_hot_encoder import OneHotEncoder, auto_select_categorical_features, _X_selected


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
        self.threshold = threshold
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
        array-like, {n_samples, n_components}
        """
        selected = auto_select_categorical_features(X, threshold=self.threshold)
        X_sel, _, n_selected, _ = _X_selected(X, selected)

        if n_selected == 0:
            # No features selected.
            raise ValueError('No categorical feature was found!')
        else:
            ohe = OneHotEncoder(categorical_features='all', sparse=False, minimum_fraction=self.minimum_fraction)
            return ohe.fit_transform(X_sel)


class ContinuousSelector(BaseEstimator, TransformerMixin):
    """Meta-transformer for selecting continuous features and transform them using PCA.

    Parameters
    ----------

    threshold : int, default=10
        Maximum number of unique values per feature to consider the feature
        to be categorical.

    svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < X.shape[1]
        randomized :
            run randomized SVD by the method of Halko et al.

    iterated_power : int >= 0, or 'auto', (default 'auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.

    """

    def __init__(self, threshold=10, svd_solver='randomized' ,iterated_power='auto', random_state=42):
        """Create a ContinuousSelector object."""
        self.threshold = threshold
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.random_state = random_state


    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like
        """
        X = check_array(X)
        return self


    def transform(self, X):
        """Select continuous features and transform them using PCA.

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        array-like, {n_samples, n_components}
        """
        selected = auto_select_categorical_features(X, threshold=self.threshold)
        _, X_sel, n_selected, _ = _X_selected(X, selected)

        if n_selected == 0:
            # No features selected.
            raise ValueError('No continuous feature was found!')
        else:
            pca = PCA(svd_solver=self.svd_solver, iterated_power=self.iterated_power, random_state=self.random_state)
            return pca.fit_transform(X_sel)


class ColumnTransformer(BaseEstimator, TransformerMixin):
    """Wrapper around sklearn.compose.ColumnTransformer, for ease of use with evolutionary algo.

    Parameters
    ----------

    include_col_{n} : bool
        Whether to include col n or not. Example: include_col_5 = True

    transformer_{n} : sklearn transformer object
        Whether to include transformer n or not. Example: transformer_0 = StandardScaler()

    choice: int
        Which transformer to use. For example, choice = 1 will use transformer_1

    remainder: 'drop' | 'passthrough'
        'drop' will drop the unselected columns, 'passthrough' will leave them unchanged
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cols = []
        self.transformer = None
        self.col_transformer = None

        if not kwargs:
            return
        transformers_temp = {}
        # parse cols and transformer
        for k, v in kwargs.items():
            if 'include_col_' in k:
                if v:
                    self.cols.append(int(k.split('_')[-1]))
            elif 'transformer_' in k:
                transformers_temp[int(k.split('_')[-1])] = v

        if 'choice' in kwargs and kwargs['choice'] is not None:
            self.transformer = transformers_temp[kwargs['choice']]
            self.col_transformer = make_column_transformer((self.transformer, self.cols), remainder=kwargs['remainder'])

    def fit(self, X, y=None):
        """Call underlying fit method of transformer.
        Parameters
        ----------
        X : array-like
        """
        if self.transformer:
            cols = [c for c in self.cols if c < X.shape[1]]
            self.col_transformer = make_column_transformer((self.transformer, cols), remainder=self.kwargs['remainder'])
        if self.col_transformer:
            self.col_transformer.fit(X, y)
        return self


    def transform(self, X):
        """Call underlying transform method of transformer.

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        array-like, {n_samples, n_components}
        """
        if self.col_transformer:
            r = self.col_transformer.transform(X)
            return r
        elif self.transformer:
            raise NotFittedError("This ColumnTransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")            
        return X

