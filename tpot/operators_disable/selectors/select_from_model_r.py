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
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor


class TPOTSelectFromModelR(Selector):
    """Uses scikit-learn's ExtraTreesRegressor combined with SelectFromModel
    to transform the feature set.

    Parameters
    ----------
    threshold: float
        Features whose importance is greater or equal are kept while the others
        are discarded.
    max_features: float
        For the ExtraTreesRegressor:
        The number of features to consider when looking for the best split

    """
    import_hash = {
        'sklearn.feature_selection': ['SelectFromModel'],
        'sklearn.ensemble':          ['ExtraTreesRegressor']
    }
    sklearn_class = SelectFromModel
    arg_types = (float, float)
    classification = False

    def __init__(self):
        pass

    def preprocess_args(self, threshold, max_features):
        threshold = min(1., max(0., threshold))

        max_features = min(1., max(0., max_features))

        return {
            'estimator': ExtraTreesRegressor(max_features=max_features),
            'threshold': threshold
        }
