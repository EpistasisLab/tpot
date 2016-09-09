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
from ..base import DEAPType
from ..gp_types import Bool, Tol, CType, Penalty
from sklearn.svm import LinearSVC


class LinearSVCLoss(DEAPType):
    """Specifies the loss function. ‘hinge’ is the standard SVM loss
    (used e.g. by the SVC class) while ‘squared_hinge’ is the square
    of the hinge loss."""

    values = ['hinge', 'squared_hinge']


class TPOTLinearSVC(Classifier):
    """Fits a Linear Support Vector Classifier"""

    import_hash = {'sklearn.svm': ['LinearSVC']}
    sklearn_class = LinearSVC
    arg_types = (Penalty, LinearSVCLoss, Bool, Tol, CType)

    def __init__(self):
        pass

    def preprocess_args(self, penalty, loss, dual, tol, C):
        if penalty == 'l1' and loss == 'hinge':
            loss = 'squared_hinge'

        if not dual and penalty == 'l2' and loss == 'hinge':
            loss = 'squared_hinge'

        if not dual and penalty == 'l1':
            dual = True

        if penalty == 'l1' and loss == 'squared_hinge' and dual:
            dual = False

        return {
            'C': C,
            'penalty': penalty,
            'loss': loss,
            'tol': tol,
            'dual': dual
        }
