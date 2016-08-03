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
details. You should have received a copy of the GNU General Public License along
with the TPOT library. If not, see http://www.gnu.org/licenses/.

"""

from .base import Preprocessor
from sklearn.cluster import FeatureAgglomeration


class TPOTFeatureAgglomeration(Preprocessor):
    """Uses scikit-learn's Nystroem to transform the feature set

    Parameters
    ----------
    affinity: int
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" is accepted.
        Input integer is used to select one of the above strings.
    linkage: int
        Can be one of the following values:
            "ward", "complete", "average"
        Input integer is used to select one of the above strings.

    """
    import_hash = {'sklearn.cluster': ['FeatureAgglomeration']}
    sklearn_class = FeatureAgglomeration
    arg_types = (int, int)

    def __init__(self):
        pass

    def preprocess_args(self, affinity, linkage):
        linkage_types = ['ward', 'complete', 'average']
        linkage_name = linkage_types[linkage % len(linkage_types)]

        affinity_types = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
        affinity_name = 'euclidean' if linkage_name == 'ward' else affinity_types[affinity % len(affinity_types)]

        return {
            'affinity': affinity_name,
            'linkage': linkage_name
        }
