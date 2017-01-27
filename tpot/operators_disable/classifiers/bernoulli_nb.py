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
from sklearn.naive_bayes import BernoulliNB


class TPOTBernoulliNB(Classifier):
    """Fits a Bernoulli Naive Bayes Classifier

    Parameters
    ----------
    alpha: float
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    binarize: float
        Threshold for binarizing (mapping to booleans) of sample features.

    """
    import_hash = {'sklearn.naive_bayes': ['BernoulliNB']}
    sklearn_class = BernoulliNB
    arg_types = (float, float)

    def __init__(self):
        pass

    def preprocess_args(self, alpha, binarize):
        return {
            'alpha': alpha,
            'binarize': binarize,
            'fit_prior': True
        }
