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
from ..base import DEAPType
from ..gp_types import Float, NComponents
from sklearn.kernel_approximation import Nystroem


class Kernel(DEAPType):
    """Kernel type"""

    # Pulled from sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS
    values = ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid']


class TPOTNystroem(Preprocessor):
    """Uses scikit-learn's Nystroem to transform the feature set

    Parameters
    ----------
    gamma: float
        Gamma parameter for the kernels.
    n_components: int
        The number of components to keep

    """
    import_hash = {'sklearn.kernel_approximation': ['Nystroem']}
    sklearn_class = Nystroem
    arg_types = (Kernel, Float, NComponents)

    def __init__(self):
        pass

    def preprocess_args(self, kernel, gamma, n_components):
        return {
            'kernel': kernel,
            'gamma': gamma,
            'n_components': n_components
        }
