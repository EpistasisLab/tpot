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

from sklearn.cluster import (
    FeatureAgglomeration,
)
from toolz import (
    partial,
    curry,
)
from traitlets import (
    Int,
)

_affinity_types = [
    'euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed',
]
_linkage_types = [
    'ward', 'complete', 'average',
]

attr = curry(
    lambda value, index: value[index % len(value)]
)


class feat_agg(EvaluateEstimator):
    """Fits a C-support vector classifier
    Parameters
    ----------
    C: float
        Penalty parameter C of the error term; must be a positive value
    """
    model = FeatureAgglomeration
    n_clusters = Int(1).tag(
        apply=partial(max, 1),
    )
    affinity = Int(1).tag(
        apply=attr(_affinity_types),
    )
    linkage = Int(1).tag(
        apply=attr(_linkage_types),
    )
