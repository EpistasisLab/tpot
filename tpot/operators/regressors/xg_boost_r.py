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
from ..gp_types import MaxDepth, MinChildWeight, SubSample, LearningRate
from xgboost import XGBRegressor


class TPOTXGBRegressor(Regressor):
    """Fits an XGBoost Regressor"""

    import_hash = {'xgboost': ['XGBRegressor']}
    sklearn_class = XGBRegressor
    arg_types = (MaxDepth, MinChildWeight, SubSample, LearningRate)

    def __init__(self):
        pass

    def preprocess_args(self, max_depth, min_child_weight, subsample, learning_rate):
        return {
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'n_estimators': 500
        }
