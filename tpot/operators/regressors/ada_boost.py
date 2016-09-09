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
from ..base import DEAPType
from ..gp_types import LearningRate, MaxDepth
from sklearn.ensemble import AdaBoostRegressor


class AdaBoostRLoss(DEAPType):
    """The loss function to use when updating the weights after each boosting iteration"""

    values = ['linear', 'square', 'exponential']


class TPOTAdaBoostRegressor(Regressor):
    """Fits an AdaBoost Regressor"""

    import_hash = {'sklearn.ensemble': ['AdaBoostRegressor']}
    sklearn_class = AdaBoostRegressor
    arg_types = (LearningRate, MaxDepth, AdaBoostRLoss)

    def __init__(self):
        pass

    def preprocess_args(self, learning_rate, max_depth, loss):
        return {
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'loss': loss,
            'n_estimators': 500
        }
