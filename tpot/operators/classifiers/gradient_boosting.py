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
from ..gp_types import LearningRate, MaxDepth, MinSamplesLeaf, MinSamplesSplit, SubSample, MaxFeatures
from sklearn.ensemble import GradientBoostingClassifier


class TPOTGradientBoosting(Classifier):
    """Fits a Gradient Boosting classifier"""

    import_hash = {'sklearn.ensemble': ['GradientBoostingClassifier']}
    sklearn_class = GradientBoostingClassifier
    arg_types = (LearningRate, MaxDepth, MinSamplesLeaf, MinSamplesSplit, SubSample, MaxFeatures)

    def __init__(self):
        pass

    def preprocess_args(self, learning_rate, max_depth, min_samples_leaf, min_samples_split, subsample, max_features):
        return {
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split,
            'subsample': subsample,
            'max_features': max_features,
            'n_estimators': 500
        }
