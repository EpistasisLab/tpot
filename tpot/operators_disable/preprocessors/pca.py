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

from .base import Preprocessor
from sklearn.decomposition import PCA

class TPOTRandomizedPCA(Preprocessor):

    """Uses scikit-learn's randomized PCA to transform the feature set

    Parameters
    ----------
    iterated_power: int
        Number of iterations for the power method. [1, 10]
    """

    import_hash = {'sklearn.decomposition': ['PCA']}
    sklearn_class = PCA
    arg_types = (int, )

    def __init__(self):
        pass

    def preprocess_args(self, iterated_power):
        iterated_power = min(10, max(1, iterated_power))

        return {
            'svd_solver': 'randomized',
            'iterated_power': iterated_power
        }
