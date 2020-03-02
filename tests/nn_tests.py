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

from tpot import TPOTClassifier, TPOTRegressor
from tpot.config.classifier_nn import classifier_config_nn
import tpot.nn

import numpy as np
import scipy as sp
import nose
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator
from nose.tools import nottest

train_test_split = nottest(train_test_split)

np.random.seed(42)
random.seed(42)


# Set up testing data

input_data = pd.read_csv(
    'tests/tests.csv',
    sep=',',
    dtype=np.float64,
)
pd_features = input_data.drop('class', axis=1)
pd_target = input_data['class']


# Tests

def test_conf_dict_nn():
    """Assert that TPOT can assign TPOT-NN config dictionary to a TPOTClassifier"""
    tpot_obj = TPOTClassifier(config_dict='TPOT NN')
    assert tpot_obj.config_dict == classifier_config_nn

# We need to run tests on the TPOT-NN models directly, since they are
# implemented natively.