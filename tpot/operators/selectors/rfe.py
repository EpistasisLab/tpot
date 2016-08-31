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

from .base import Selector
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


class TPOTRFE(Selector):
    """Uses scikit-learn's RFE to transform the feature set

    Parameters
    ----------
    step: float
        The percentage of features to drop each iteration

    """
    import_hash = {'sklearn.feature_selection': ['RFE'], 'sklearn.svm': ['SVC']}
    sklearn_class = RFE
    arg_types = (float, )
    regression = False  # Can not be used in regression due to SVC estimator

    def __init__(self):
        pass

    def preprocess_args(self, step):
        step = max(min(0.99, step), 0.1)

        return {
            'step': step,
            'estimator': SVC(kernel='linear', random_state=42)
        }
