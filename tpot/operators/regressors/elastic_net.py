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

import numpy as np

from .base import Regressor
from ..base import DEAPType
from ..gp_types import Tol
from sklearn.linear_model import ElasticNet


class L1Ratio(DEAPType):
    """The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1"""

    values = np.arange(0.0, 1.01, 0.05)


class TPOTElasticNet(Regressor):
    """Fits a Elastic Net Regressor"""

    import_hash = {'sklearn.linear_model': ['ElasticNet']}
    sklearn_class = ElasticNet
    arg_types = (Tol, L1Ratio)

    def __init__(self):
        pass

    def preprocess_args(self, tol, l1_ratio):
        return {
            'tol': tol,
            'l1_ratio': l1_ratio
        }
