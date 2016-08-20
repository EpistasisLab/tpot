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
from sklearn.linear_model import PassiveAggressiveClassifier


class TPOTPassiveAggressive(Classifier):
    """Fits a Passive Aggressive Classifier

    Parameters
    ----------
    C: float
        Penalty parameter C of the error term.
    loss: int
        Integer used to determine the loss function (either 'hinge' or 'squared_hinge')

    """
    import_hash = {'sklearn.linear_model': ['PassiveAggressiveClassifier']}
    sklearn_class = PassiveAggressiveClassifier
    arg_types = (float, int)

    def __init__(self):
        pass

    def preprocess_args(self, C, loss):
        loss_values = ['hinge', 'squared_hinge']
        loss_selection = loss_values[loss % len(loss_values)]

        C = min(1., max(0.0001, C))

        return {
            'C': C,
            'loss': loss_selection,
            'fit_intercept': True
        }
