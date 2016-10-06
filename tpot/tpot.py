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

from .base import TPOTBase


class TPOTClassifier(TPOTBase):
    """TPOT estimator for classification problems"""

    scoring_function = 'balanced_accuracy'  # Classification scoring

    def _ignore_operator(self, op):
        """Filter that describes which operators are not used

        Parameters
        ----------
        op: Operator
            TPOT Pipeline operator being tested
        """
        return not op.classification


class TPOTRegressor(TPOTBase):
    """TPOT estimator for regression problems"""

    scoring_function = 'neg_mean_squared_error'  # Regression scoring

    def _ignore_operator(self, op):
        """Filter that describes which operators are not used

        Parameters
        ----------
        op: Operator
            TPOT Pipeline operator being tested
        """
        return not op.regression
