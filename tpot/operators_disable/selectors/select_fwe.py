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
from sklearn.feature_selection import SelectFwe, f_classif


class TPOTSelectFwe(Selector):
    """Uses scikit-learn's SelectFwe to transform the feature set

    Parameters
    ----------
    alpha: float in the range [0.001, 0.05]
        The highest uncorrected p-value for features to keep

    """
    import_hash = {'sklearn.feature_selection': ['SelectFwe', 'f_classif']}
    sklearn_class = SelectFwe
    arg_types = (float, )

    def __init__(self):
        pass

    def preprocess_args(self, alpha):
        alpha = max(min(0.05, alpha), 0.001)

        return {
            'score_func': f_classif,
            'alpha': alpha
        }
