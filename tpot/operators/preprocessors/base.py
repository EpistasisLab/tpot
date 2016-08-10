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
import warnings

from tpot.operators import Operator
from tpot.indices import non_feature_columns


class Preprocessor(Operator):
    """Parent class for Feature Preprocessors in TPOT"""

    root = False  # Whether this operator type can be the root of the tree

    def _call(self, input_matrix, *args, **kwargs):
        # Calculate arguments to be passed directly to sklearn
        operator_args = self.preprocess_args(*args, **kwargs)

        # Run the feature-preprocessor with args
        features = np.delete(input_matrix, non_feature_columns, axis=1)
        modified_df = self._fit_transform(features, operator_args)

        # Add non_feature_columns back to the matrix
        for col in non_feature_columns:
            modified_df = np.insert(modified_df, 0, input_matrix[:, col], axis=1)

        return modified_df

    def _fit_transform(self, features, operator_args):
        """Run the Preprocessor and return the modified DataFrame

        Parameters
        ----------
            features: numpy.ndarray
            operator_args: dict
                Dictionary of arguments to be passed to the preprocessor

        Returns
        -------
            transformed_features: numpy.ndarray

        """
        # Send arguments to preprocessor but also attempt to add in default
        # arguments defined in the Operator class
        op = self._merge_with_default_params(operator_args)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)

            op.fit(self.training_features.astype(np.float64))
            transformed_features = op.transform(features).astype(np.float64)

        return transformed_features
