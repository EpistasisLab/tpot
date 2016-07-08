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
from sklearn.feature_selection import VarianceThreshold


class TPOTVarianceThreshold(Selector):
    """Uses scikit-learn's VarianceThreshold to transform the feature set

    Parameters
    ----------
    threshold: float
        The variance threshold that removes features that fall under the threshold

    """
    import_hash = {'sklearn.feature_selection': ['VarianceThreshold']}
    sklearn_class = VarianceThreshold
    arg_types = (float, )

    def __init__(self):
        pass

    def preprocess_args(self, threshold):
        return {
            'threshold': threshold
        }
