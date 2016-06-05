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

from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
)
from toolz import (
    partial,
)
from traitlets import (
    Int,
)


class logistic_regression(EvaluateEstimator):
    """Fits a logistic regression classifier
    Parameters
    ----------
    input_df: pandas.DataFrame {n_samples, n_features+['class',
    'group', 'guess']}
        Input DataFrame for fitting the logistic regression classifier
    C: float
        Inverse of regularization strength; must be a positive value. Like in
        support vector machines, smaller values specify stronger regularization.
    """
    model = LogisticRegression
    C = Int(1).tag(
        apply=partial(max, .0001)
    )


class passive_aggressive(EvaluateEstimator):
    """Fits a Linear Support Vector Classifier
    Parameters
    ----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the classifier
    C: float
        Penalty parameter C of the error term.
    loss: int
        Integer used to determine the loss function (either 'hinge' or 'squared_hinge')
    fit_intercept : int
        Whether to calculate the intercept for this model (even for True, odd for False)
    """
    model = PassiveAggressiveClassifier
    C = Int(1).tag(
        apply=partial(max, .0001)
    )
    fit_intercept = Int(0).tag(
        apply=lambda x: (x % 2) == 0
    )
    loss = Int(0).tag(
        apply=lambda x: ['hinge', 'squared_hinge'][x % 2]
    )
