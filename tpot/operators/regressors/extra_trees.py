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

from .base import Regressor
from sklearn.ensemble import ExtraTreesRegressor


class TPOTExtraTreesRegressor(Regressor):
    """Fits an Extra Trees Regressor

    Parameters
    ----------
    criterion: int
        Integer that is used to select from the list of valid criteria,
        either 'gini', or 'entropy'
    max_features: float
        The number of features to consider when looking for the best split

    """
    import_hash = {'sklearn.ensemble': ['ExtraTreesRegressor']}
    sklearn_class = ExtraTreesRegressor
    arg_types = (float, )

    def __init__(self):
        pass

    def preprocess_args(self, max_features):
        max_features = min(1., max(0., max_features))

        return {
            'max_features': max_features,
            'n_estimators': 500
        }
