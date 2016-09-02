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
from ._auto_sklearn_onehotencoder import _OneHotEncoder

import numpy as np

class TPOTOneHotEncoder(Preprocessor):
    """Uses scikit-learn's OneHotEncoder to transform the feature set

    Parameters
    ----------
    None

    """
    import_hash = {'_auto_sklearn_onehotencoder': ['_OneHotEncoder']}
    # sklearn_class = OneHotEncoder
    arg_types = (int, float, bool, float)

    def __init__(self):
        pass

    def preprocess_args(self, categorical_features="all", dtype=np.float,
                 sparse=False, minimum_fraction=None):

        return {
            'categorical_features': categorical_features,
            'dtype': dtype,
            'sparse': sparse,
            'minimum_fraction': minimum_fraction
        }
