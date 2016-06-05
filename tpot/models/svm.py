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
details. You should have received a copy of the GNU General Public License
along with the TPOT library. If not, see http://www.gnu.org/licenses/.
"""

from .base import (
    EvaluateEstimator,
)

from sklearn.svm import (
    SVC,
    LinearSVC,
)
from toolz import (
    partial,
)
from traitlets import (
    Int,
)

class svc(EvaluateEstimator):
    """Fits a C-support vector classifier
    Parameters
    ----------
    C: float
        Penalty parameter C of the error term; must be a positive value
    """
    model = SVC
    C = Int(1).tag(
        apply=partial(max, .0001)
    )


class linear_svc(EvaluateEstimator):
    """Fits a logistic regression classifier
    Parameters
    ----------
    C: float
        Inverse of regularization strength; must be a positive value. Like in
        support vector machines, smaller values specify stronger regularization
    """
    model = LinearSVC
    C = Int(1).tag(
        apply=partial(max, .0001)
    )
    fit_intercept = Int(0).tag(
        apply=lambda x: (x % 2) == 0
    )
    loss = Int(0).tag(
        apply=lambda x: ['hinge', 'squared_hinge'][x % 2]
    )
