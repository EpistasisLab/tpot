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
from .base import Classifier
from sklearn.linear_model import LogisticRegression


class TPOTLogisticRegression(Classifier):
    """Fits a logistic regression classifier

    Parameters
    ----------
    C: float
        Inverse of regularization strength; must be a positive value. Like in support vector machines, smaller values specify stronger regularization.
    penalty: int
        Integer used to specify the norm used in the penalization (l1 or l2)
    solver: sag (Note: sciket-learn version > 0.17) for l1 and liblinear for l2
        Algorithm to use in the optimization problem.
        SAG = Stochastic Average Gradient descent solver.
        Note that 'sag' fast convergence is only guaranteed on features with approximately the same scale.
        'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty.
    """
    import_hash = {'sklearn.linear_model': ['LogisticRegression']}
    sklearn_class = LogisticRegression
    arg_types = (float, int)

    def __init__(self):
        pass

    def preprocess_args(self, C, penalty):
        C = min(50., max(0.0001, C))

        penalty_values = ['l1', 'l2']
        penalty_selection = penalty_values[penalty % len(penalty_values)]
        if penalty_selection == 'l1':
            solver = 'liblinear'
        else:
            solver = 'sag'
        return {
            'C': C,
            'penalty': penalty_selection,
            'solver': solver
        }
