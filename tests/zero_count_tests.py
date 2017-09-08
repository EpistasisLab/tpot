# -*- coding: utf-8 -*-

"""Copyright 2015-Present Randal S. Olson.

This file is part of the TPOT library.

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
from tpot.builtins import ZeroCount

X = np.array([[0, 1, 7, 0, 0],
            [3, 0, 0, 2, 19],
            [0, 1, 3, 4, 5],
            [5, 0, 0, 0, 0]])

def test_ZeroCount():
    """Assert that ZeroCount operator returns correct transformed X."""
    op = ZeroCount()
    X_transformed = op.transform(X)
    zero_col = np.array([3, 2, 1, 4])
    non_zero = np.array([2, 3, 4, 1])

    assert np.allclose(zero_col, X_transformed[:, 0])
    assert np.allclose(non_zero, X_transformed[:, 1])


def test_ZeroCount_fit():
    """Assert that fit() ZeroCount operator does nothing."""
    op = ZeroCount()
    ret_op = op.fit(X)
    
    assert ret_op==op
