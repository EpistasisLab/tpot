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
from sklearn.ensemble import ExtraTreesClassifier


class TPOTExtraTreesClassifier(Classifier):
    """Fits an Extra Trees Classifier

    Parameters
    ----------
    criterion: int
        Integer that is used to select from the list of valid criteria,
        either 'gini', or 'entropy'
    max_features: float
        The number of features to consider when looking for the best split
    min_weight_fraction_leaf: float
        The minimum weighted fraction of the input samples required to be at a leaf node.

    """
    import_hash = {'sklearn.ensemble': ['ExtraTreesClassifier']}
    sklearn_class = ExtraTreesClassifier
    arg_types = (int, float, float)

    def __init__(self):
        pass

    def preprocess_args(self, criterion, max_features, min_weight_fraction_leaf):
        # Select criterion string from list of valid parameters
        criterion_values = ['gini', 'entropy']
        criterion_selection = criterion_values[criterion % len(criterion_values)]

        min_weight = min(0.5, max(0., min_weight_fraction_leaf))
        max_features = min(1., max(0., max_features))

        return {
            'min_weight_fraction_leaf': min_weight,
            'criterion': criterion_selection,
            'max_features': max_features,
            'n_estimators': 500
        }
