#!/usr/bin/env python3
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
from sklearn.neural_network import MLPClassifier, MLPRegressor


class EmbeddingEstimator(TransformerMixin, BaseEstimator):
    """Meta-transformer for creating neural network embeddings as features.
    """

    def __init__(self, estimator, embedding_layer=None, backend=None):
        """Create a StackingEstimator object.

        Parameters
        ----------
        estimator: neural network model; either from sklearn or Keras-like.
            The estimator to generate embeddings.
        embedding_layer: the particular layer used as the embedding. 
            By default we use the second last layer. Layers are counted with
            input layer being `0th` layer; negative indices are allowed. 
        backend: (optional), the backend we use to query the neural network. 
            Not required if using scikit-learn interface.
            Currently only supports keras-like interface (incl. tensorflow)
        """
        second_last_layer = -2
        self.estimator = estimator
        self.embedding_layer = (
            second_last_layer if embedding_layer is None else embedding_layer
        )
        self.backend = backend

    def fit(self, X, y=None, **fit_params):
        """Fit the StackingEstimator meta-transformer.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like, shape (n_samples,)
            The target values (integers that correspond to classes in classification, real numbers in regression).
        fit_params:
            Other estimator-specific parameters.

        Returns
        -------
        self: object
            Returns a copy of the estimator
        """
        if not issubclass(self.estimator.__class__, MLPClassifier) and not issubclass(
            self.estimator.__class__, MLPRegressor
        ):
            input_shape = X.shape[1]
            self.estimator.sk_params["input_shape"] = input_shape
            self.estimator.check_params(self.estimator.sk_params)
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        """Transform data by adding embedding as features.

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features + embedding) where embedding is the size of the embedding layer
            The transformed feature set.
        """
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, MLPClassifier) or issubclass(
            self.estimator.__class__, MLPRegressor
        ):
            X_transformed = np.hstack(
                (self._embedding_mlp(self.estimator, X), X_transformed)
            )
        else:
            X_transformed = np.hstack(
                (self._embedding_keras(self.estimator, X), X_transformed)
            )

        return X_transformed

    def _embedding_mlp(self, estimator, X):
        # see also BaseMultilayerPerceptron._predict from
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/multilayer_perceptron.py
        X = check_array(X, accept_sparse=["csr", "csc", "coo"])

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = estimator.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [X.shape[1]] + hidden_layer_sizes + [estimator.n_outputs_]

        # Initialize layers
        activations = [X]

        for i in range(estimator.n_layers_ - 1):
            activations.append(np.empty((X.shape[0], layer_units[i + 1])))
        # forward propagate
        estimator._forward_pass(activations)
        y_embedding = activations[self.embedding_layer]

        return y_embedding

    def _embedding_keras(self, estimator, X):
        X = check_array(X, accept_sparse=["csr", "csc", "coo"])
        get_embedding = self.backend.function(
            [estimator.model.layers[0].input],
            [estimator.model.layers[self.embedding_layer].output],
        )
        return get_embedding([X])[0]
