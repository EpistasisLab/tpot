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

from ...gp_types import Bool
from .base import Regressor
from sklearn.svm import LinearSVR


class TPOTLinearSVR(Regressor):
    """Fits a Linear Support Vector Regressor

    Parameters
    ----------
    C: float
        Penalty parameter C of the error term.
    dual: bool
        Select the algorithm to either solve the dual or primal optimization problem.

    """
    import_hash = {'sklearn.svm': ['LinearSVR']}
    sklearn_class = LinearSVR
    arg_types = (float, Bool)

    def __init__(self):
        pass

    def preprocess_args(self, C, dual):
        C = min(25., max(0.0001, C))

        return {
            'C': C,
            'dual': dual
        }
