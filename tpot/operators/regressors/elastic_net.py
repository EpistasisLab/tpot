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
from sklearn.linear_model import ElasticNet


class TPOTElasticNet(Regressor):
    """Fits a Elastic Net Regressor

    Parameters
    ----------
    alpha: float
        Constant that multiplies the penalty terms.
    l1_ratio: int
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1

    """
    import_hash = {'sklearn.linear_model': ['ElasticNet']}
    sklearn_class = ElasticNet
    arg_types = (float, float)

    def __init__(self):
        pass

    def preprocess_args(self, alpha, l1_ratio):
        alpha = min(1., max(0.0001, alpha))
        l1_ratio = min(1., max(0.0001, l1_ratio))

        return {
            'alpha': alpha,
            'l1_ratio': l1_ratio
        }
