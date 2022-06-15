# -*- coding: utf-8 -*-

"""
Copyright (c) 2014, Matthias Feurer
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np
import scipy.sparse
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.datasets import load_iris, load_boston
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from nose.tools import assert_equal

from tpot.builtins import OneHotEncoder, auto_select_categorical_features, _transform_selected


iris_data = load_iris().data

dense1 = np.array([[0, 1, 0],
                   [0, 0, 0],
                   [1, 1, 0]])
dense1_1h = np.array([[1, 0, 0, 1, 1],
                     [1, 0, 1, 0, 1],
                     [0, 1, 0, 1, 1]])
dense1_1h_minimum_fraction = np.array([[0, 1, 0, 1, 1],
                                       [0, 1, 1, 0, 1],
                                       [1, 0, 0, 1, 1]])

# Including NaNs
dense2 = np.array([[0, np.NaN, 0],
                   [np.NaN, 0, 2],
                   [1, 1, 1],
                   [np.NaN, 0, 1]])
dense2_1h = np.array([[0, 1, 0, 1, 0, 0, 1, 0, 0],
                      [1, 0, 0, 0, 1, 0, 0, 0, 1],
                      [0, 0, 1, 0, 0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 1, 0, 0, 1, 0]])

dense2_1h_minimum_fraction = np.array([[1, 0, 1, 0, 1, 0],
                                       [0, 1, 0, 1, 1, 0],
                                       [1, 0, 1, 0, 0, 1],
                                       [0, 1, 0, 1, 0, 1]])

dense2_partial_1h = np.array([[0., 1., 0., 1., 0., 0., 0.],
                              [1., 0., 0., 0., 1., 0., 2.],
                              [0., 0., 1., 0., 0., 1., 1.],
                              [1., 0., 0., 0., 1., 0., 1.]])

dense2_1h_minimum_fraction_as_sparse = np.array([[0, 0, 1, 0, 0, 0],
                                                 [0, 1, 0, 0, 1, 0],
                                                 [1, 0, 0, 1, 0, 1],
                                                 [0, 1, 0, 0, 0, 1]])

# All NaN slice
dense3 = np.array([[0, 1, np.NaN],
                   [1, 0, np.NaN]])
dense3_1h = np.array([[1, 0, 0, 1, 1],
                      [0, 1, 1, 0, 1]])

sparse1 = scipy.sparse.csc_matrix(([3, 2, 1, 1, 2, 3],
                                   ((1, 4, 5, 2, 3, 5),
                                    (0, 0, 0, 1, 1, 1))), shape=(6, 2))
sparse1_1h = scipy.sparse.csc_matrix(([1, 1, 1, 1, 1, 1],
                                      ((5, 4, 1, 2, 3, 5),
                                       (0, 1, 2, 3, 4, 5))), shape=(6, 6))
sparse1_paratial_1h = scipy.sparse.csc_matrix(([1, 1, 1, 1, 2, 3],
                                               ((5, 4, 1, 2, 3, 5),
                                                (0, 1, 2, 3, 3, 3))),
                                              shape=(6, 4))

# All zeros slice
sparse2 = scipy.sparse.csc_matrix(([2, 1, 0, 0, 0, 0],
                                   ((1, 4, 5, 2, 3, 5),
                                    (0, 0, 0, 1, 1, 1))), shape=(6, 2))
sparse2_1h = scipy.sparse.csc_matrix(([1, 1, 1, 1, 1, 1],
                                      ((5, 4, 1, 2, 3, 5),
                                       (0, 1, 2, 3, 3, 3))), shape=(6, 4))

sparse2_csr = scipy.sparse.csr_matrix(([2, 1, 0, 0, 0, 0],
                                      ((1, 4, 5, 2, 3, 5),
                                       (0, 0, 0, 1, 1, 1))), shape=(6, 2))
sparse2_csr_1h = scipy.sparse.csr_matrix(([1, 1, 1, 1, 1, 1],
                                         ((5, 4, 1, 2, 3, 5),
                                          (0, 1, 2, 3, 3, 3))), shape=(6, 4))


def fit_then_transform(expected, input, categorical_features='all',
                       minimum_fraction=None):
    # Test fit_transform
    ohe = OneHotEncoder(categorical_features=categorical_features,
                        minimum_fraction=minimum_fraction)
    transformation = ohe.fit_transform(input.copy())
    assert_array_almost_equal(expected.astype(float),
                              transformation.todense())

    # Test fit, and afterwards transform
    ohe2 = OneHotEncoder(categorical_features=categorical_features,
                         minimum_fraction=minimum_fraction)
    ohe2.fit(input.copy())
    transformation = ohe2.transform(input.copy())
    assert_array_almost_equal(expected, transformation.todense())


def fit_then_transform_dense(expected, input,
                             categorical_features='all',
                             minimum_fraction=None):
    ohe = OneHotEncoder(categorical_features=categorical_features,
                        sparse=False, minimum_fraction=minimum_fraction)
    transformation = ohe.fit_transform(input.copy())
    assert_array_almost_equal(expected, transformation)

    ohe2 = OneHotEncoder(categorical_features=categorical_features,
                         sparse=False, minimum_fraction=minimum_fraction)
    ohe2.fit(input.copy())
    transformation = ohe2.transform(input.copy())
    assert_array_almost_equal(expected, transformation)


def test_auto_detect_categorical():
    """Assert that automatic selection of categorical features works as expected with a threshold of 10."""
    selected = auto_select_categorical_features(iris_data[0:16, :], threshold=10)
    expected = [False, False, True, True]

    assert_equal(selected, expected)


def test_dense1():
    """Test fit_transform a dense matrix."""
    fit_then_transform(dense1_1h, dense1)
    fit_then_transform_dense(dense1_1h, dense1)


def test_dense1_minimum_fraction():
    """Test fit_transform a dense matrix with minimum_fraction=0.5."""
    fit_then_transform(dense1_1h_minimum_fraction, dense1, minimum_fraction=0.5)
    fit_then_transform_dense(dense1_1h_minimum_fraction, dense1, minimum_fraction=0.5)


def test_dense2():
    """Test fit_transform a dense matrix including NaNs."""
    fit_then_transform(dense2_1h, dense2)
    fit_then_transform_dense(dense2_1h, dense2)


def test_dense2_minimum_fraction():
    """Test fit_transform a dense matrix including NaNs with minimum_fraction=0.5"""
    fit_then_transform(
        dense2_1h_minimum_fraction,
        dense2,
        minimum_fraction=0.3
    )
    fit_then_transform_dense(
        dense2_1h_minimum_fraction,
        dense2,
        minimum_fraction=0.3
    )


def test_dense2_with_non_sparse_components():
    """Test fit_transform a dense matrix including NaNs with specifying categorical_features."""
    fit_then_transform(
        dense2_partial_1h,
        dense2,
        categorical_features=[True, True, False]
    )
    fit_then_transform_dense(
        dense2_partial_1h,
        dense2,
        categorical_features=[True, True, False]
    )


def test_sparse_on_dense2_minimum_fraction():
    """Test fit_transform a dense matrix with minimum_fraction as sparse"""
    sparse = scipy.sparse.csr_matrix(dense2)
    fit_then_transform(
        dense2_1h_minimum_fraction_as_sparse,
        sparse,
        minimum_fraction=0.5
    )
    fit_then_transform_dense(
        dense2_1h_minimum_fraction_as_sparse,
        sparse,
        minimum_fraction=0.5
    )


# Minimum fraction is not too interesting here...
def test_dense3():
    """Test fit_transform a dense matrix including all NaN slice."""
    fit_then_transform(dense3_1h, dense3)
    fit_then_transform_dense(dense3_1h, dense3)


def test_sparse1():
    """Test fit_transform a sparse matrix."""
    fit_then_transform(sparse1_1h.todense(), sparse1)
    fit_then_transform_dense(sparse1_1h.todense(), sparse1)


def test_sparse1_minimum_fraction():
    """Test fit_transform a sparse matrix with minimum_fraction=0.5."""
    expected = np.array([[0, 1, 0, 0, 1, 1],
                         [0, 0, 1, 1, 0, 1]], dtype=float).transpose()
    fit_then_transform(
        expected,
        sparse1,
        minimum_fraction=0.5
    )
    fit_then_transform_dense(
        expected,
        sparse1,
        minimum_fraction=0.5
    )


def test_sparse1_with_non_sparse_components():
    """Test fit_transform a sparse matrix with specifying categorical_features."""
    fit_then_transform(
        sparse1_paratial_1h.todense(),
        sparse1,
        categorical_features=[True, False]
    )


def test_sparse2():
    """Test fit_transform a sparse matrix including all zeros slice."""
    fit_then_transform(sparse2_1h.todense(), sparse2)
    fit_then_transform_dense(sparse2_1h.todense(), sparse2)


def test_sparse2_minimum_fraction():
    """Test fit_transform a sparse matrix including all zeros slice with minimum_fraction=0.5."""
    expected = np.array([[0, 1, 0, 0, 1, 1],
                         [0, 0, 1, 1, 0, 1]], dtype=float).transpose()
    fit_then_transform(
        expected,
        sparse2,
        minimum_fraction=0.5
    )
    fit_then_transform_dense(
        expected,
        sparse2,
        minimum_fraction=0.5
    )


def test_sparse2_csr():
    """Test fit_transform another sparse matrix including all zeros slice."""
    fit_then_transform(sparse2_csr_1h.todense(), sparse2_csr)
    fit_then_transform_dense(sparse2_csr_1h.todense(), sparse2_csr)


def test_transform():
    """Test OneHotEncoder with both dense and sparse matrixes."""
    input = np.array(((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))).transpose()
    ohe = OneHotEncoder()
    ohe.fit(input)
    test_data = np.array(((0, 1, 2, 6), (0, 1, 6, 7))).transpose()
    output = ohe.transform(test_data).todense()
    assert np.sum(output) == 5

    input = np.array(((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))).transpose()
    ips = scipy.sparse.csr_matrix(input)
    ohe = OneHotEncoder()
    ohe.fit(ips)
    test_data = np.array(((0, 1, 2, 6), (0, 1, 6, 7))).transpose()
    tds = scipy.sparse.csr_matrix(test_data)
    output = ohe.transform(tds).todense()
    assert np.sum(output) == 3


def test_transform_selected():
    """Assert _transform_selected return original X when selected is empty list"""
    ohe = OneHotEncoder(categorical_features=[])
    X = _transform_selected(
            dense1,
            ohe._fit_transform,
            ohe.categorical_features,
            copy=True
        )
    assert np.allclose(X, dense1)


def test_transform_selected_2():
    """Assert _transform_selected return original X when selected is a list of False values"""
    ohe = OneHotEncoder(categorical_features=[False, False, False])
    X = _transform_selected(
            dense1,
            ohe._fit_transform,
            ohe.categorical_features,
            copy=True
        )
    assert np.allclose(X, dense1)


def test_k_fold_cv():
    """Test OneHotEncoder with categorical_features='auto'."""
    boston = load_boston()
    clf = make_pipeline(
        OneHotEncoder(
            categorical_features='auto',
            sparse=False,
            minimum_fraction=0.05
        ),
        LinearRegression()
    )

    cross_val_score(clf, boston.data, boston.target, cv=KFold(n_splits=10, shuffle=True))


def test_refit_on_new_data():
    """Test that OneHotEncoder can refit on two data sets."""
    ohe = OneHotEncoder()
    ohe.fit(dense1)
    ohe.fit(dense2)
