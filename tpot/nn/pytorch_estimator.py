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

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset

import numpy as np
from pandas import to_numeric
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import ipdb


class PytorchEstimator(ClassifierMixin, BaseEstimator):
    """Base class for Pytorch-based estimators (currently only classifiers) for
    use in TPOT.

    In the future, these will be merged into TPOT's main code base.
    """
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class PytorchClassifier(PytorchEstimator):
    def predict(self, X):
        return self.transform(X)
    

class LR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


class PytorchLRClassifier(PytorchClassifier):
    """Logistic Regression classifier, implemented in PyTorch, for use with
    TPOT.
    """

    class PytorchDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __getitem__(self, i):
            return (self.X[i, :], self.y[i, :])

        def __len__(self):
            return self.X.shape[0]

    def __init__(
            self,
            penalty="l2",
            num_epochs=10,
            batch_size=2,
            learning_rate=0.01,
            num_classes=2):
        super().__init__()
        self.penalty = penalty
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes

    def get_params(self, deep=True):
        return {
            "penalty": self.penalty,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_classes": self.num_classes
        }

    def fit(self, X, y):
        """Based on code from
        https://www.kaggle.com/negation/pytorch-logistic-regression-tutorial
        """

        X, y = check_X_y(X, y, accept_sparse=False)

        # Make sure all labels are between 0 and (n_classes - 1)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        if np.any(np.iscomplex(X)) or np.any(np.iscomplex(y)):
            raise ValueError("Complex data not supported")

        # if (not np.issubdtype(X.dtype, np.number)) or (not np.issubdtype(y.dtype, np.number)):
        #     raise ValueError("Data must be numeric")
        if np.issubdtype(X.dtype, np.object_) or np.issubdtype(y.dtype, np.object_):
            try:
                X = X.astype(float)
                y = y.astype(int)
            except TypeError:
                raise TypeError("argument must be a string.* number")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y are of incompatible lengths")

        self.input_size = X.shape[-1]
        num_classes = len(set(y))

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        train_dset = TensorDataset(X, y)

        train_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

        self.model = LR(self.input_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            for i, (rows, labels) in enumerate(train_loader):
                rows = Variable(rows)
                labels = Variable(labels)

                optimizer.zero_grad()
                outputs = self.model(rows)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(
                        "Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f"
                        % (
                            epoch + 1,
                            self.num_epochs,
                            i + 1,
                            len(train_dset) // self.batch_size,
                            loss.item(),
                        )
                    )

        self.is_fitted_ = True
        return self

    def forward(self, x):
        out = self.linear(x)
        return out

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        # if np.issubdtype(X.dtype, np.object_):
        #     X = X.astype(float)
        
        X = torch.tensor(X, dtype=torch.float32)
        predictions = np.empty(len(X), dtype=int)
        for i, rows in enumerate(X):
            rows = Variable(rows.view(-1, self.input_size))
            outputs = self.model(rows)
            #ipdb.set_trace()
            _, predicted = torch.max(outputs.data, 1)
            predictions[i] = int(predicted)

        # Convert class predictions back to their original integer labels
        predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions

    def transform(self, X):
        return self.predict(X)

    def _more_tags(self):
        return {'non_deterministic': True, 'binary_only': True}


class PytorchMLP(PytorchEstimator):
    """Multilayer Perceptron, implemented in PyTorch, for use with TPOT.
    """
    def __init__(self, num_epochs=5):
        super().__init__()
        self.num_epochs = num_epochs

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
