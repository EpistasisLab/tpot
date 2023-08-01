import pytest
import numpy as np
import pandas as pd

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

test_remove_underrepresented_classes()