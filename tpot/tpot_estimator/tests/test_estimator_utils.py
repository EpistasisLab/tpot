"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Gabriel Ketron (gabriel.ketron@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

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
import pytest
import numpy as np
import pandas as pd
from ..estimator_utils import *

def test_remove_underrepresented_classes():
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 2])
    min_count = 2
    x_result, y_result = remove_underrepresented_classes(x, y, min_count)
    np.testing.assert_array_equal(x_result, np.array([[1, 2], [5, 6]]))
    np.testing.assert_array_equal(y_result, np.array([0, 0]))

    x = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]}).T
    y = pd.Series([0, 1, 0, 2])
    min_count = 2
    x_result, y_result = remove_underrepresented_classes(x, y, min_count)
    pd.testing.assert_frame_equal(x_result, pd.DataFrame({'a': [1, 2], 'c': [5, 6]}).T)
    pd.testing.assert_series_equal(y_result, pd.Series([0, 1, 0, 2])[[0,2]])

    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    min_count = 2
    x_result, y_result = remove_underrepresented_classes(x, y, min_count)
    np.testing.assert_array_equal(x_result, np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    np.testing.assert_array_equal(y_result, np.array([0, 1, 0, 1]))

    x = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]}).T
    y = pd.Series([0, 1, 0, 1])
    min_count = 2
    x_result, y_result = remove_underrepresented_classes(x, y, min_count)
    pd.testing.assert_frame_equal(x_result, pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]}).T)
    pd.testing.assert_series_equal(y_result, pd.Series([0, 1, 0, 1]))


def test_check_if_y_is_encoded():
    assert check_if_y_is_encoded([0, 1, 2, 3]) == True
    assert check_if_y_is_encoded([0, 1, 3, 4]) == False
    assert check_if_y_is_encoded([0, 2, 3]) == False
    assert check_if_y_is_encoded([0]) == True
    assert check_if_y_is_encoded([0,0,0,0,1,1,1,1]) == True
    assert check_if_y_is_encoded([0,0,0,0,1,1,1,1,3]) == False
    assert check_if_y_is_encoded([1,1,1,1,2,2,2,2]) == False
