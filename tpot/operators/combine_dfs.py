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

from tpot.indices import non_feature_columns


class CombineDFs(object):
    """Operator to combine two DataFrames"""

    def __init__(self):
        pass

    @property
    def __name__(self):
        return self.__class__.__name__

    def __call__(self, input_mat1, input_mat2):
        """
        Parameters
        ----------
        input_df1: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
            Input matrix to combine
        input_df2: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
            Input matrix to combine
        Returns
        -------
        combined_features: numpy.ndarray {n_samples, n_both_features+['guess', 'group', 'class']}
            Returns a DataFrame containing the features of both input_df1 and input_df2
        """
        features1 = np.delete(input_mat1, non_feature_columns)
        features2 = np.delete(input_mat2, non_feature_columns)

        combined_features = np.concatenate([features1, features2], axis=1)

        for col in non_feature_columns:
            np.insert(combined_features, 0, input_mat1[:, col], axis=1)

        return combined_features
