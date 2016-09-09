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
from ..gp_types import SelectorThreshold, MaxFeatures, ClassCriterion
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier


class TPOTSelectFromModel(Selector):
    """Uses scikit-learn's ExtraTreesClassifier combined with SelectFromModel
    to transform the feature set"""

    import_hash = {
        'sklearn.feature_selection': ['SelectFromModel'],
        'sklearn.ensemble':          ['ExtraTreesClassifier']
    }
    sklearn_class = SelectFromModel
    arg_types = (SelectorThreshold, ClassCriterion, MaxFeatures)
    regression = False  # Can not be used in regression due to ExtraTreesClassifier

    def __init__(self):
        pass

    def preprocess_args(self, threshold, criterion, max_features):
        return {
            'estimator': ExtraTreesClassifier(criterion=criterion, max_features=max_features),
            'threshold': threshold
        }
