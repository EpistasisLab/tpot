# -*- coding: utf-8 -*-

"""
Copyright 2016 Randal S. Olson
This file is part of the TPOT library.
The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.
The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License
along with the TPOT library. If not, see http://www.gnu.org/licenses/.
"""

from .base import (
    EvaluateEstimator,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
)
from toolz import (
    partial, pipe,
)
from traitlets import (
    Int,
)


class knnc(EvaluateEstimator):
    """Fits a k-nearest neighbor classifier
    Parameters
    ----------
    n_neighbors: int
        Number of neighbors to use by default for k_neighbors queries; must
        be a positive value
    """
    model = KNeighborsClassifier
    n_neighbors = Int(3).tag(
        df=True,
        apply=lambda df, x: pipe(
            x, partial(min, len(df.ix[True])-1), partial(max, 2)
        )
    )
