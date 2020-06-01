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

# Note: There are quite a few pylint messages disabled in this file. In
# general, this usually should be avoided. However, in some cases it is
# necessary: e.g., we use `X` and `y` to refer to data and labels in compliance
# with the scikit-learn API, but pylint doesn't like short variable names.

# pylint: disable=redefined-outer-name
# pylint: disable=not-callable

from abc import abstractmethod

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, assert_all_finite, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target

try:
    import torch
    from torch import nn
    from torch.autograd import Variable
    from torch.optim import Adam
    from torch.utils.data import TensorDataset, DataLoader
except ModuleNotFoundError:
    raise

def _pytorch_model_is_fully_initialized(clf: BaseEstimator):
    if all([
        hasattr(clf, 'network'),
        hasattr(clf, 'loss_function'),
        hasattr(clf, 'optimizer'),
        hasattr(clf, 'data_loader'),
        hasattr(clf, 'train_dset_len'),
        hasattr(clf, 'device')
    ]):
        return True
    else:
        return False

def _get_cuda_device_if_available():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class PytorchEstimator(BaseEstimator):
    """Base class for Pytorch-based estimators (currently only classifiers) for
    use in TPOT.

    In the future, these will be merged into TPOT's main code base.
    """

    @abstractmethod
    def fit(self, X, y): # pragma: no cover
        pass

    @abstractmethod
    def transform(self, X): # pragma: no cover
        pass

    def predict(self, X):
        return self.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class PytorchClassifier(PytorchEstimator, ClassifierMixin):
    @abstractmethod
    def _init_model(self, X, y): # pragma: no cover
        pass

    def fit(self, X, y):
        """Generalizable method for fitting a PyTorch estimator to a training
        set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """
        # pylint: disable=no-member

        self._init_model(X, y)

        assert _pytorch_model_is_fully_initialized(self)

        for epoch in range(self.num_epochs):
            for i, (samples, labels) in enumerate(self.data_loader):
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.network(samples)

                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                if self.verbose and ((i + 1) % 100 == 0):
                    print(
                        "Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f"
                        % (
                            epoch + 1,
                            self.num_epochs,
                            i + 1,
                            self.train_dset_len // self.batch_size,
                            loss.item(),
                        )
                    )

        # pylint: disable=attribute-defined-outside-init
        self.is_fitted_ = True
        return self

    def validate_inputs(self, X, y):
        # Things we don't want to allow until we've tested them:
        # - Sparse inputs
        # - Multiclass outputs (e.g., more than 2 classes in `y`)
        # - Non-finite inputs
        # - Complex inputs

        X, y = check_X_y(X, y, accept_sparse=False, allow_nd=False)

        assert_all_finite(X, y)

        if type_of_target(y) != 'binary':
            raise ValueError("Non-binary targets not supported")

        if np.any(np.iscomplex(X)) or np.any(np.iscomplex(y)):
            raise ValueError("Complex data not supported")
        if np.issubdtype(X.dtype, np.object_) or np.issubdtype(y.dtype, np.object_):
            try:
                X = X.astype(float)
                y = y.astype(int)
            except (TypeError, ValueError):
                raise ValueError("argument must be a string.* number")

        return (X, y)

    def predict(self, X):
        # pylint: disable=no-member

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        predictions = np.empty(len(X), dtype=int)
        for i, rows in enumerate(X):
            rows = Variable(rows.view(-1, self.input_size))
            outputs = self.network(rows)

            _, predicted = torch.max(outputs.data, 1)
            predictions[i] = int(predicted)
        return predictions.reshape(-1, 1)

    def transform(self, X):
        return self.predict(X)



class _LR(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, input_size, num_classes):
        super(_LR, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

class _MLP(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, input_size, num_classes):
        super(_MLP, self).__init__()

        self.hidden_size = round((input_size+num_classes)/2)

        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        hidden = self.fc1(x)
        r1 = self.relu(hidden)
        out = self.fc2(r1)
        return out


class PytorchLRClassifier(PytorchClassifier):
    """Logistic Regression classifier, implemented in PyTorch, for use with
    TPOT.

    For examples on standalone use (i.e., non-TPOT) refer to:
    https://github.com/trang1618/tpot-nn/blob/master/tpot_nn/estimator_sandbox.py
    """

    def __init__(
        self,
        num_epochs=10,
        batch_size=16,
        learning_rate=0.02,
        weight_decay=1e-4,
        verbose=False
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose

        self.input_size = None
        self.num_classes = None
        self.network = None
        self.loss_function = None
        self.optimizer = None
        self.data_loader = None
        self.train_dset_len = None
        self.device = None

    def _init_model(self, X, y):
        device = _get_cuda_device_if_available()

        X, y = self.validate_inputs(X, y)

        self.input_size = X.shape[-1]
        self.num_classes = len(set(y))

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        train_dset = TensorDataset(X, y)

        # Set parameters of the network
        self.network = _LR(self.input_size, self.num_classes).to(device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.data_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self.train_dset_len = len(train_dset)
        self.device = device

    def _more_tags(self):
        return {'non_deterministic': True, 'binary_only': True}

class PytorchMLPClassifier(PytorchClassifier):
    """Multilayer Perceptron, implemented in PyTorch, for use with TPOT.
    """

    def __init__(
        self,
        num_epochs=10,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0,
        verbose=False
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose

        self.input_size = None
        self.num_classes = None
        self.network = None
        self.loss_function = None
        self.optimizer = None
        self.data_loader = None
        self.train_dset_len = None
        self.device = None

    def _init_model(self, X, y):
        device = _get_cuda_device_if_available()

        X, y = self.validate_inputs(X, y)

        self.input_size = X.shape[-1]
        self.num_classes = len(set(y))

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        train_dset = TensorDataset(X, y)

        # Set parameters of the network
        self.network = _MLP(self.input_size, self.num_classes).to(device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.data_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self.train_dset_len = len(train_dset)
        self.device = device

    def _more_tags(self):
        return {'non_deterministic': True, 'binary_only': True}
