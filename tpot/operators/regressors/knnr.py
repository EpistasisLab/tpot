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

from .base import Regressor
from sklearn.neighbors import KNeighborsRegressor


class TPOTKNeighborsRegressor(Regressor):
    """Fits a k-nearest neighbor Regressor

    Parameters
    ----------
    n_neighbors: int
        Number of neighbors to use by default for k_neighbors queries; must be a positive value
    weights: int
        Selects a value from the list: ['uniform', 'distance']

    """
    import_hash = {'sklearn.neighbors': ['KNeighborsRegressor']}
    sklearn_class = KNeighborsRegressor
    arg_types = (int, int)

    def __init__(self):
        pass

    def preprocess_args(self, n_neighbors, weights):
        n_neighbors = max(min(5, n_neighbors), 2)

        weights_values = ['uniform', 'distance']
        weights_selection = weights_values[weights % len(weights_values)]

        return {
            'n_neighbors': n_neighbors,
            'weights': weights_selection
        }
