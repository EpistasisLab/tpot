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

from .base import Regressor
from sklearn.linear_model import PassiveAggressiveRegressor


class TPOTPassiveAggressiveR(Regressor):
    """Fits a Passive Aggressive Regressor

    Parameters
    ----------
    C: float
        Penalty parameter C of the error term.
    loss: int
        Integer used to determine the loss function
        (either 'epsilon_insensitive' or 'squared_epsilon_insensitive')

    """
    import_hash = {'sklearn.linear_model': ['PassiveAggressiveRegressor']}
    sklearn_class = PassiveAggressiveRegressor
    arg_types = (float, int)

    def __init__(self):
        pass

    def preprocess_args(self, C, loss):
        loss_values = ['epsilon_insensitive', 'squared_epsilon_insensitive']
        loss_selection = loss_values[loss % len(loss_values)]

        C = min(1., max(0.0001, C))

        return {
            'C': C,
            'loss': loss_selection,
            'fit_intercept': True
        }
