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
from ..base import DEAPType
from ..gp_types import Bool, Tol, CType
from sklearn.svm import LinearSVR


class LinearSVRLoss(DEAPType):
    """Loss function to use"""

    values = ['epsilon_insensitive', 'squared_epsilon_insensitive']


class Epsilon(DEAPType):
    """Epsilon parameter in the epsilon-insensitive loss function"""

    values = [1e-4, 1e-3, 1e-2, 1e-1, 1.]


class TPOTLinearSVR(Regressor):
    """Fits a Linear Support Vector Regressor"""

    import_hash = {'sklearn.svm': ['LinearSVR']}
    sklearn_class = LinearSVR
    arg_types = (LinearSVRLoss, Bool, Tol, CType, Epsilon)

    def __init__(self):
        pass

    def preprocess_args(self, loss, dual, tol, C, epsilon):
        if not dual and loss == 'epsilon_insensitive':
            dual = True

        return {
            'C': C,
            'dual': dual,
            'loss': loss,
            'tol': tol,
            'epsilon': epsilon
        }
