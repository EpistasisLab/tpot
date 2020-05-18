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

from tpot import TPOTClassifier
from tpot.config import classifier_config_nn

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from nose.tools import nottest, assert_raises

train_test_split = nottest(train_test_split)

# Set up data used in tests
input_data = pd.read_csv(
    'tests/tests.csv',
    sep=',',
    dtype=np.float64,
)
pd_features = input_data.drop('class', axis=1)
pd_target = input_data['class']

multiclass_X, multiclass_y = make_classification(
    n_samples=50,
    n_features=20,
    n_classes=3,
    n_clusters_per_class=1
)


clf = TPOTClassifier(
    random_state=42,
    population_size=1,
    generations=1,
    config_dict=classifier_config_nn,
    template='PytorchLRClassifier'
)
assert_raises(ValueError, clf.fit(multiclass_X, multiclass_y))

# Tests

def test_nn_conf_dict():
    """NN: Assert that we can instantiate a TPOT classifier with the NN config dict."""
    clf = TPOTClassifier(config_dict=classifier_config_nn)
    assert clf.config_dict == classifier_config_nn

def test_nn_errors_on_multiclass():
    """NN: Assert that TPOT-NN throws an error when you try to pass training data with > 2 classes."""
clf = TPOTClassifier(
    random_state=42,
    population_size=1,
    generations=1,
    config_dict=classifier_config_nn,
    template='PytorchLRClassifier'
)
assert_raises(ValueError, clf.fit(multiclass_X, multiclass_y))

def test_pytorch_lr_classifier():
    """"""
    clf = TPOTClassifier(
        random_state=42,
        population_size=1,
        generations=1,
        config_dict=classifier_config_nn,
        template='PytorchLRClassifier'
    )
    clf.fit(pd_features, pd_target)

def test_pytorch_mlp_classifier():
    """"""
    clf = TPOTClassifier(
        random_state=42,
        population_size=1,
        generations=1,
        config_dict=classifier_config_nn,
        template='PytorchMLPClassifier'
    )
    clf.fit(pd_features, pd_target)
