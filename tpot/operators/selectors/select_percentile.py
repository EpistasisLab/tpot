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

from .base import Selector
from sklearn.feature_selection import SelectPercentile, f_classif


class TPOTSelectPercentile(Selector):
    """Uses scikit-learn's SelectPercentile to transform the feature set

    Parameters
    ----------
    percentile: int
        The features that belong in the top percentile to keep from the original
        set of features in the training data

    """
    import_hash = {'sklearn.feature_selection': ['SelectPercentile', 'f_classif']}
    sklearn_class = SelectPercentile
    arg_types = (int, )

    def __init__(self):
        pass

    def preprocess_args(self, percentile):
        percentile = max(min(99, percentile), 1)

        return {
            'score_func': f_classif,
            'percentile': percentile
        }
