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

from .base import DEAPType


class Output_DF(object):
    """Output data type of pipelines"""

    pass


class Bool(DEAPType):
    """Boolean class used for deap due to deap's poor handling of booleans"""

    values = [True, False]


class Float(DEAPType):
    """Default float set"""

    values = np.concatenate((
        [1e-6, 1e-5, 1e-4, 1e-3],
        np.arange(0., 1.01, 0.01),
        np.arange(2., 51., 1.),
        np.arange(60., 101., 10.))
    )


class Integer(DEAPType):
    """Default int set"""

    values = np.concatenate((
        np.arange(0, 51, 1),
        np.arange(60, 110, 10))
    )


class NComponents(DEAPType):
    """Number of components to keep"""

    values = np.concatenate((
        np.arange(1, 51, 1),
        np.arange(60, 110, 10))
    )


class IteratedPower(DEAPType):
    """Number of iterations for the power method"""

    values = np.arange(1, 11, 1)


class LearningRate(DEAPType):
    """Learning rate shrinks the contribution of each classifier"""

    values = [1e-3, 1e-2, 1e-1, 0.5, 1.]


class MaxDepth(DEAPType):
    """Maximum depth of a tree"""

    values = np.arange(1, 11, 1)


class Alpha(DEAPType):
    """Additive smoothing parameter"""

    values = [1e-3, 1e-2, 1e-1, 1., 10., 100.]


class MaxFeatures(DEAPType):
    """The maximum number of features to consider"""

    values = np.arange(0, 1.01, 0.05)


class SubSample(DEAPType):
    """The fraction of samples to be used for fitting the individual base learners"""

    values = np.arange(0.05, 1.01, 0.05)


class ClassCriterion(DEAPType):
    """The function to measure the quality of a split"""

    values = ['gini', 'entropy']


class MinSamplesSplit(DEAPType):
    """The minimum number of samples required to split an internal node"""

    values = np.arange(2, 21, 1)


class MinSamplesLeaf(DEAPType):
    """The minimum number of samples required to be at a leaf node"""

    values = np.arange(1, 21, 1)


class MinChildWeight(DEAPType):
    """Minimum sum of instance weight(hessian) needed in a child"""

    values = np.arange(1, 21, 1)


class Penalty(DEAPType):
    """Used to specify the norm used in the penalization"""

    values = ['l1', 'l2']


class CType(DEAPType):
    """Inverse of regularization strength; must be a positive float"""

    values = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]


class Tol(DEAPType):
    """Tolerance for stopping criteria"""

    values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


class KNWeights(DEAPType):
    """Weight function used in prediction"""

    values = ['uniform', 'distance']


class KNNeighbors(DEAPType):
    """Number of neighbors to use by default for k_neighbors queries"""

    values = np.arange(1, 101, 1)


class PType(DEAPType):
    """Power parameter for the Minkowski metric"""

    values = [1, 2]


class SelectorThreshold(DEAPType):
    """Features whose importance is greater or equal are kept while the others are discarded"""

    values = np.arange(0, 1.01, 0.05)
