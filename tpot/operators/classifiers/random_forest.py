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
from ..gp_types import ClassCriterion, MinSamplesSplit, MinSamplesLeaf, MaxDepth, Bool
from sklearn.ensemble import RandomForestClassifier


class TPOTRandomForestClassifier(Classifier):
    """Fits a random forest classifier"""

    import_hash = {'sklearn.ensemble': ['RandomForestClassifier']}
    sklearn_class = RandomForestClassifier
    arg_types = (ClassCriterion, MaxDepth, MinSamplesLeaf, MinSamplesSplit, Bool)

    def __init__(self):
        pass

    def preprocess_args(self, criterion, max_depth, min_samples_leaf, min_samples_split, bootstrap):
        return {
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split,
            'bootstrap': bootstrap,
            'n_estimators': 500
        }
