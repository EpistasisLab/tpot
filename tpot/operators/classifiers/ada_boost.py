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
from sklearn.ensemble import AdaBoostClassifier


class TPOTAdaBoostClassifier(Classifier):
    """Fits an AdaBoost classifier

    Parameters
    ----------
    learning_rate: float
        Learning rate shrinks the contribution of each classifier by learning_rate.

    """
    import_hash = {'sklearn.ensemble': ['AdaBoostClassifier']}
    sklearn_class = AdaBoostClassifier
    arg_types = (float, )

    def __init__(self):
        pass

    def preprocess_args(self, learning_rate):
        learning_rate = min(1., max(0.0001, learning_rate))

        return {
            'learning_rate': learning_rate,
            'n_estimators': 500
        }
