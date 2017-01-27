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
from xgboost import XGBRegressor


class TPOTXGBRegressor(Regressor):
    """Fits an XGBoost Regressor

    Parameters
    ----------
    max_depth: int
        Maximum tree depth for base learners
    min_child_weight: int
        Minimum sum of instance weight(hessian) needed in a child
    learning_rate: float
        Shrinks the contribution of each tree by learning_rate
    subsample: float
        Subsample ratio of the training instance
    """
    import_hash = {'xgboost': ['XGBRegressor']}
    sklearn_class = XGBRegressor
    arg_types = (int, int, float, float)

    def __init__(self):
        pass

    def preprocess_args(self, max_depth, min_child_weight, learning_rate, subsample):
        max_depth = min(10, max(max_depth, 1))
        min_child_weight = min(20, max(min_child_weight, 1))
        learning_rate = min(1., max(learning_rate, 0.0001))
        subsample = min(1., max(subsample, 0.05))

        return {
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'n_estimators': 500
        }
