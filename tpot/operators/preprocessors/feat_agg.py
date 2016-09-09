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
from ..base import DEAPType
from sklearn.cluster import FeatureAgglomeration


class FeatAggLinkage(DEAPType):
    """The linkage criterion determines which distance to use between sets of features"""

    values = ['ward', 'complete', 'average']


class FeatAggAffinity(DEAPType):
    """Metric used to compute the linkage"""

    values = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']


class TPOTFeatureAgglomeration(Preprocessor):
    """Uses scikit-learn's Nystroem to transform the feature set"""

    import_hash = {'sklearn.cluster': ['FeatureAgglomeration']}
    sklearn_class = FeatureAgglomeration
    arg_types = (FeatAggLinkage, FeatAggAffinity)

    def __init__(self):
        pass

    def preprocess_args(self, linkage, affinity):
        if linkage == 'ward':
            affinity = 'euclidean'

        return {
            'affinity': affinity,
            'linkage': linkage
        }
