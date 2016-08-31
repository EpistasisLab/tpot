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
from sklearn.preprocessing import MaxAbsScaler

class TPOTMaxAbsScaler(Preprocessor):

    """Uses scikit-learn's MaxAbsScaler to transform all of the features by scaling them to [0, 1] relative to the feature's maximum value.
    
    Parameters
    ----------
    None
    """

    import_hash = {'sklearn.preprocessing': ['MaxAbsScaler']}
    sklearn_class = MaxAbsScaler
    arg_types = ()

    def __init__(self):
        pass

    def preprocess_args(self):
        """Preprocess the arguments in case they need to be limited to a certain value range"""
        return { }
