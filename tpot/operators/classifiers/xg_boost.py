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

from .base import Classifier
from ..gp_types import MaxDepth, LearningRate, SubSample, MinChildWeight
from xgboost import XGBClassifier


class TPOTXGBClassifier(Classifier):
    """Fits an XGBoost Classifier"""

    import_hash = {'xgboost': ['XGBClassifier']}
    sklearn_class = XGBClassifier
    arg_types = (MaxDepth, LearningRate, SubSample, MinChildWeight)

    def __init__(self):
        pass

    def preprocess_args(self, max_depth, learning_rate, subsample, min_child_weight):
        return {
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'n_estimators': 500
        }
