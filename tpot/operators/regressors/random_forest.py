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
from ..gp_types import MinSamplesSplit, MinSamplesLeaf, MaxFeatures, Bool
from sklearn.ensemble import RandomForestRegressor


class TPOTRandomForestClassifier(Regressor):
    """Fits a random forest Regressor"""

    import_hash = {'sklearn.ensemble': ['RandomForestRegressor']}
    sklearn_class = RandomForestRegressor
    arg_types = (MaxFeatures, MinSamplesLeaf, MinSamplesSplit, Bool)

    def __init__(self):
        pass

    def preprocess_args(self, max_features, min_samples_leaf, min_samples_split, bootstrap):
        return {
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split,
            'bootstrap': bootstrap,
            'n_estimators': 500
        }
