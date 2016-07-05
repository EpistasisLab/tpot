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
from sklearn.tree import DecisionTreeClassifier


class TPOTDecisionTreeClassifier(Classifier):
    """Fits a decision tree classifier

    Parameters
    ----------
    min_weight_fraction_leaf: float
        The minimum weighted fraction of the input samples required to be at a leaf node.

    """
    import_hash = {'sklearn.tree': ['DecisionTreeClassifier']}
    sklearn_class = DecisionTreeClassifier

    def __init__(self):
        pass

    def preprocess_args(self, min_weight_fraction_leaf: float):
        min_weight = min(0.5, max(0., min_weight_fraction_leaf))

        return {
            'min_weight_fraction_leaf': min_weight
        }
