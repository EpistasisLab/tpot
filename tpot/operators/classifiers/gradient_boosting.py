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
from sklearn.ensemble import GradientBoostingClassifier


class TPOTGradientBoosting(Classifier):
    """Fits a Gradient Boosting classifier

    Parameters
    ----------
    learning_rate: float
        Shrinks the contribution of each tree by learning_rate
    max_features: float
        Maximum number of features to use (proportion of total features)

    """
    import_hash = {'sklearn.ensemble': ['GradientBoostingClassifier']}
    sklearn_class = GradientBoostingClassifier
    arg_types = (float, float)

    def __init__(self):
        pass

    def preprocess_args(self, learning_rate, max_features):
        learning_rate = min(1., max(learning_rate, 0.0001))
        max_features = min(1., max(0., learning_rate))

        return {
            'learning_rate': learning_rate,
            'max_features': max_features,
            'n_estimators': 500
        }
