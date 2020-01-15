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

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset, TensorDataset

import numpy as np
from sklearn.utils.validation import assert_all_finite, check_X_y, check_array, check_is_fitted

from .pytorch_estimator import PytorchClassifier

class _LR(nn.Module):
    def __init__(self, input_size, num_classes, penalty):
        super(_LR, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

class PytorchLRClassifier(PytorchClassifier):
    """Logistic Regression classifier, implemented in PyTorch, for use with
    TPOT.
    """

    def __init__(
        self,
        penalty="l2",
        num_epochs=10,
        batch_size=2,
        learning_rate=0.01,
        num_classes=2,
    ):
        super().__init__()
        self.penalty = penalty
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes

    def _validate_inputs(self, X, y):
        """Ensure that users have provided valid input, converting datatypes
        where possible.
        
        """
        X, y = check_X_y(X, y, accept_sparse=False)

        # Data assertions
        assert_all_finite(X, y)

        # dtype assertions
        if np.any(np.iscomplex(X)) or np.any(np.iscomplex(y)):
            raise ValueError("Complex data not supported")
        if np.issubdtype(X.dtype, np.object_) or np.issubdtype(y.dtype, np.object_):
            try:
                X = X.astype(float)
                y = y.astype(int)
            except TypeError:
                raise TypeError("argument must be a string.* number")

        return (X, y)

    def fit(self, X, y):
        """Fit the Pytorch Logistic Regression Classifier using a training set.
        
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
        X, y = self._validate_inputs(X, y)

        self.input_size = X.shape[-1]
        num_classes = len(set(y))

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        train_dset = TensorDataset(X, y)

        train_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

        self.model = _LR(self.input_size, num_classes, self.penalty)
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            #print("Epoch: [%d/%d]" % (epoch+1, self.num_epochs))
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

    # def forward(self, x):
    #     out = self.linear(x)
    #     return out

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        X = torch.tensor(X, dtype=torch.float32)
        predictions = np.empty(len(X), dtype=int)
        for i, rows in enumerate(X):
            rows = Variable(rows.view(-1, self.input_size))
            outputs = self.model(rows)

            _, predicted = torch.max(outputs.data, 1)
            predictions[i] = int(predicted)
        return predictions.reshape(-1, 1)

    def transform(self, X):
        return self.predict(X)

    def _more_tags(self):
        return {'non_deterministic': True, 'binary_only': True}