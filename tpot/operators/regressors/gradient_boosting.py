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

import numpy as np

from .base import Regressor
from ..base import DEAPType
from ..gp_types import LearningRate, MaxDepth, MinSamplesLeaf, MinSamplesSplit, SubSample, MaxFeatures
from sklearn.ensemble import GradientBoostingRegressor


class GBRLoss(DEAPType):
    """Loss function to use"""

    values = ['ls', 'lad', 'huber', 'quantile']


class GBAlpha(DEAPType):
    """The alpha-quantile of the huber loss function and the quantile loss function"""

    values = [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


class TPOTGradientBRegressor(Regressor):
    """Fits a Gradient Boosting Regressor"""

    import_hash = {'sklearn.ensemble': ['GradientBoostingRegressor']}
    sklearn_class = GradientBoostingRegressor
    arg_types = (GBRLoss, LearningRate, MaxDepth, MinSamplesLeaf, MinSamplesSplit, SubSample, MaxFeatures, GBAlpha)

    def __init__(self):
        pass

    def preprocess_args(self, loss, learning_rate, max_depth, min_leaf, min_split, subsample, max_features, alpha):
        return {
            'loss': loss,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_leaf': min_leaf,
            'min_samples_split': min_split,
            'subsample': subsample,
            'max_features': max_features,
            'alpha': alpha,
            'n_estimators': 500
        }
