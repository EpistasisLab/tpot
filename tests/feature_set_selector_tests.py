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
import pandas as pd
from tpot.builtins import FeatureSetSelector
from nose.tools import assert_raises

test_data = pd.read_csv("tests/tests.csv")
test_X = test_data.drop("class", axis=1)


def test_FeatureSetSelector_1():
    """Assert that the StackingEstimator returns transformed X based on test feature list 1."""
    ds = FeatureSetSelector(subset_list="tests/subset_test.csv", sel_subset="test_subset_1")
    ds.fit(test_X, y=None)
    transformed_X = ds.transform(test_X)

    assert transformed_X.shape[0] == test_X.shape[0]
    assert transformed_X.shape[1] != test_X.shape[1]
    assert transformed_X.shape[1] == 5
    assert np.array_equal(transformed_X, test_X[ds.feat_list].values)

def test_FeatureSetSelector_2():
    """Assert that the StackingEstimator returns transformed X based on test feature list 2."""
    ds = FeatureSetSelector(subset_list="tests/subset_test.csv", sel_subset="test_subset_2")
    ds.fit(test_X, y=None)
    transformed_X = ds.transform(test_X)

    assert transformed_X.shape[0] == test_X.shape[0]
    assert transformed_X.shape[1] != test_X.shape[1]
    assert transformed_X.shape[1] == 6
    assert np.array_equal(transformed_X, test_X[ds.feat_list].values)

def test_FeatureSetSelector_3():
    """Assert that the StackingEstimator returns transformed X based on 2 subsets' names"""
    ds = FeatureSetSelector(subset_list="tests/subset_test.csv", sel_subset=["test_subset_1", "test_subset_2"])
    ds.fit(test_X, y=None)
    transformed_X = ds.transform(test_X)

    assert transformed_X.shape[0] == test_X.shape[0]
    assert transformed_X.shape[1] != test_X.shape[1]
    assert transformed_X.shape[1] == 7
    assert np.array_equal(transformed_X, test_X[ds.feat_list].values)

def test_FeatureSetSelector_4():
    """Assert that the StackingEstimator returns transformed X based on 2 subsets' indexs"""
    ds = FeatureSetSelector(subset_list="tests/subset_test.csv", sel_subset=[0, 1])
    ds.fit(test_X, y=None)
    transformed_X = ds.transform(test_X)

    assert transformed_X.shape[0] == test_X.shape[0]
    assert transformed_X.shape[1] != test_X.shape[1]
    assert transformed_X.shape[1] == 7
    assert np.array_equal(transformed_X, test_X[ds.feat_list].values)

def test_FeatureSetSelector_5():
    """Assert that the StackingEstimator returns transformed X seleced based on test feature list 1's index."""
    ds = FeatureSetSelector(subset_list="tests/subset_test.csv", sel_subset=0)
    ds.fit(test_X, y=None)
    transformed_X = ds.transform(test_X)

    assert transformed_X.shape[0] == test_X.shape[0]
    assert transformed_X.shape[1] != test_X.shape[1]
    assert transformed_X.shape[1] == 5
    assert np.array_equal(transformed_X, test_X[ds.feat_list].values)

def test_FeatureSetSelector_6():
    """Assert that the _get_support_mask function returns correct mask."""
    ds = FeatureSetSelector(subset_list="tests/subset_test.csv", sel_subset="test_subset_1")
    ds.fit(test_X, y=None)
    mask = ds._get_support_mask()
    get_mask = ds.get_support()

    assert mask.shape[0] == 30
    assert np.count_nonzero(mask) == 5
    assert np.array_equal(get_mask, mask)

def test_FeatureSetSelector_7():
    """Assert that the StackingEstimator works as expected when input X is np.array."""
    ds = FeatureSetSelector(subset_list="tests/subset_test.csv", sel_subset="test_subset_1")
    ds.fit(test_X.values, y=None)
    transformed_X = ds.transform(test_X.values)
    str_feat_list = [str(i+2) for i in ds.feat_list_idx]


    assert transformed_X.shape[0] == test_X.shape[0]
    assert transformed_X.shape[1] != test_X.shape[1]
    assert transformed_X.shape[1] == 5
    assert np.array_equal(transformed_X, test_X.values[:, ds.feat_list_idx])
    assert np.array_equal(transformed_X, test_X[str_feat_list].values)


def test_FeatureSetSelector_8():
    """Assert that the StackingEstimator rasies ValueError when features are not available."""
    ds = FeatureSetSelector(subset_list="tests/subset_test.csv", sel_subset="test_subset_4")
    assert_raises(ValueError, ds.fit, test_X)


def test_FeatureSetSelector_9():
    """Assert that the StackingEstimator __name__ returns correct class name."""
    ds = FeatureSetSelector(subset_list="tests/subset_test.csv", sel_subset="test_subset_4")
    assert ds.__name__ == 'FeatureSetSelector'
