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

from functools import partial

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

try:
    import os
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import Model as KerasModel
    from tensorflow.keras import backend as keras_backend
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


DEFAULT_HIDDEN_LAYERS = (100,)  # same as sklearn.neural_net

def _build_model(
        input_size,
        output_size, # num classes for classification or 1 for regression
        hidden_layer_sizes=DEFAULT_HIDDEN_LAYERS,
        optimizer='adam',
        loss='categorical_crossentropy',
        kernel_initializer='normal',
        kernel_regularizer='l2',
        hidden_layer_activation='relu',
        output_layer_activation='softmax',
        metrics=['accuracy'],
    ):
    if all((0<x<1 for x in hidden_layer_sizes)):
        # relative_layer sizes
        # use ceil to insure that no layer ends up being < 1
        hidden_layer_sizes = np.ceil(np.array(hidden_layer_sizes)*input_size)
    elif all((x>1 and isinstance(x, int) for x in hidden_layer_sizes)):
        hidden_layer_sizes = np.array(hidden_layer_sizes, dtype=int)
    else:
        raise ValueError(
            "`hidden_layer_sizes` must be an iterable of int x>1 or floats 0<x<1"
        )
    model = Sequential()
    # add input layer
    model.add(Dense(
        hidden_layer_sizes[0],
        activation=hidden_layer_activation,
        input_dim=input_size
    ))
    # add hidden layers
    for layer_size in hidden_layer_sizes[1:]:
        model.add(Dense(
            layer_size,
            activation=hidden_layer_activation,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
        ))
    # add output layer
    model.add(Dense(output_size,
        activation=output_layer_activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    ))
    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics
    )
    return model

class DeepLearningClassifier(KerasClassifier):
    _build_model = partial(
        _build_model,
        output_layer_activation='softmax',
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    def __init__(self, **sk_params):
        # crete a prototype with empty input/output sizes
        # this is needed for sk_params to be processed correctly
        self.__call__ = partial(
            self._build_model,
            input_size=None,
            output_size=None
        )
        super().__init__(**sk_params)
    
    def fit(self, X, y=None, **fit_params):
        self.__call__ = partial(
            self._build_model,
            input_size=X.shape[1],
            output_size=np.unique(y).size
        )
        self.history = super().fit(X, y, **fit_params)
        return self

class DeepLearningRegressor(KerasRegressor):
    _build_model = partial(
        _build_model,
        output_layer_activation='linear',
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_squared_error'],
    )

    def __init__(self, **sk_params):
        # crete a prototype with empty input/output sizes
        # this is needed for sk_params to be processed correctly
        self.__call__ = partial(
            self._build_model,
            input_size=None,
            output_size=None
        )
        super().__init__(**sk_params)
    
    def fit(self, X, y=None, **fit_params):
        self.__call__ = partial(
            self._build_model,
            input_size=X.shape[1],
            output_size=1
        )
        self.history = super().fit(X, y, **fit_params)
        return self

class DeepLearningTransformer(TransformerMixin, DeepLearningClassifier):
    """Meta-transformer for creating neural network embeddings as features.
    """

    def __init__(self, embedding_layer=-2, **sk_params):
        """Create a StackingEstimator object.

        Parameters
        ----------
        estimator: the Keras neural network model used to generate embeddings.
        embedding_layer: the particular layer used as the embedding. 
            By default we use the second last layer. Layers are counted with
            input layer being `0th` layer; negative indices are allowed.
        backend: (optional), the backend we use to query the neural network. 
            Currently only supports keras-like interface (incl. tensorflow)
        """
        # validate embedding_layer
        if 'hidden_layer_sizes' in sk_params:
            test_list = sk_params['hidden_layer_sizes']
        else:
            test_list = DEFAULT_HIDDEN_LAYERS
        assert embedding_layer not in (-1, len(test_list)-1),\
            "Can not use output layer for embedding"
        assert embedding_layer not in (0, -len(test_list)),\
            "Can not use input layer for embedding"
        try:
            test_list[embedding_layer]
        except IndexError:
            raise ValueError(
                f"`embedding_layer` ({embedding_layer}) is not a valid index"
                f" of `hidden_layer_sizes` ({test_list})"
            )
        if embedding_layer < 0:
            self.embedding_layer = embedding_layer - 1  # adjust for output layer
        else:
            self.embedding_layer = embedding_layer
        super().__init__(**sk_params)

    def fit(self, *args, **kwargs):
        model = super().fit(*args, **kwargs)
        self._model = model
        return model

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
        X_transformed = np.hstack((self._embedding_keras(X), X_transformed))

        return X_transformed

    def _embedding_keras(self, X):
        X = check_array(X, accept_sparse=["csr", "csc", "coo"])
        get_embedding = keras_backend.function(
            [self._model.model.layers[0].input],
            [self._model.model.layers[self.embedding_layer].output],
        )
        return get_embedding([X])[0]
