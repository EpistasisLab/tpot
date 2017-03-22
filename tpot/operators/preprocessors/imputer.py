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
from sklearn.preprocessing import Imputer


class TPOTImputer(Preprocessor):
    """Uses scikit-learn's Imputer to complete missing values

    Parameters
    ----------
    strategy: 'mean', 'median', or 'most_frequent'
        The norm to use to normalize each non zero sample.

    .. _Imputer Documentation:
       http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html
    """
    import_hash = {'sklearn.preprocessing': ['Imputer']}
    sklearn_class = Imputer
    arg_types = ()

    arg_types = (int, )

    def __init__(self):
        pass

    def preprocess_args(self, strategy_id):
        strategies = ['mean', 'median', 'most_frequent']
        strategy = strategies[strategy_id % len(strategies)]

        return {
            'strategy': strategy,
        }
