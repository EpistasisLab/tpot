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
from sklearn.decomposition import FastICA


class TPOTFastICA(Preprocessor):
    """Uses scikit-learn's FastICA to transform the feature set

    Parameters
    ----------
    tol: float
        Tolerance on update at each iteration.

    """
    import_hash = {'sklearn.decomposition': ['FastICA']}
    sklearn_class = FastICA
    arg_types = (float, )

    def __init__(self):
        pass

    def preprocess_args(self, tol):
        tol = max(tol, 0.0001)

        return {
            'tol': tol
        }
