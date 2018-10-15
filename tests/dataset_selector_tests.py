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
from tpot.builtins import DatasetSelector

test_data = pd.read_csv("tests/tests.csv")
test_X = test_data.drop("class", axis=1)


def test_DatasetSelector_1():
    """Assert that the StackingEstimator returns transformed X based on test feature list 1."""
    ds = DatasetSelector(subset_list="tests/subset_test.csv", sel_subset="test_subset_1")
    ds.fit(test_X, y=None)
    transformed_X = ds.transform(test_X)

    assert transformed_X.shape[0] == test_X.shape[0]
    assert transformed_X.shape[1] != test_X.shape[1]
    assert transformed_X.shape[1] == 5
    assert np.array_equal(transformed_X, test_X[ds.feat_list].values)

def test_DatasetSelector_2():
    """Assert that the StackingEstimator returns transformed X based on test feature list 2."""
    ds = DatasetSelector(subset_list="tests/subset_test.csv", sel_subset="test_subset_2")
    ds.fit(test_X, y=None)
    transformed_X = ds.transform(test_X)

    assert transformed_X.shape[0] == test_X.shape[0]
    assert transformed_X.shape[1] != test_X.shape[1]
    assert transformed_X.shape[1] == 6
    assert np.array_equal(transformed_X, test_X[ds.feat_list].values)

def test_DatasetSelector_3():
    """Assert that the StackingEstimator returns transformed X based on 2 subsets' names"""
    ds = DatasetSelector(subset_list="tests/subset_test.csv", sel_subset=["test_subset_1", "test_subset_2"])
    ds.fit(test_X, y=None)
    transformed_X = ds.transform(test_X)

    assert transformed_X.shape[0] == test_X.shape[0]
    assert transformed_X.shape[1] != test_X.shape[1]
    assert transformed_X.shape[1] == 7
    assert np.array_equal(transformed_X, test_X[ds.feat_list].values)

def test_DatasetSelector_4():
    """Assert that the StackingEstimator returns transformed X based on 2 subsets' indexs"""
    ds = DatasetSelector(subset_list="tests/subset_test.csv", sel_subset=[0, 1])
    ds.fit(test_X, y=None)
    transformed_X = ds.transform(test_X)

    assert transformed_X.shape[0] == test_X.shape[0]
    assert transformed_X.shape[1] != test_X.shape[1]
    assert transformed_X.shape[1] == 7
    assert np.array_equal(transformed_X, test_X[ds.feat_list].values)

def test_DatasetSelector_5():
    """Assert that the StackingEstimator returns transformed X seleced based on test feature list 1's index."""
    ds = DatasetSelector(subset_list="tests/subset_test.csv", sel_subset=0)
    ds.fit(test_X, y=None)
    transformed_X = ds.transform(test_X)

    assert transformed_X.shape[0] == test_X.shape[0]
    assert transformed_X.shape[1] != test_X.shape[1]
    assert transformed_X.shape[1] == 5
    assert np.array_equal(transformed_X, test_X[ds.feat_list].values)
