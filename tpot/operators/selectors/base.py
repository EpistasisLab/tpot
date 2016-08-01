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


class Selector(Operator):
    """Parent class for Feature Selectors in TPOT"""

    root = False  # Whether this operator type can be the root of the tree

    def _call(self, input_matrix, *args, **kwargs):
        # Calculate arguments to be passed directly to sklearn
        operator_args = self.preprocess_args(*args, **kwargs)

        # Run the feature-selector with args
        return self._fit_mask(input_matrix, operator_args)

    def _fit_mask(self, input_matrix, operator_args):
        """Run the Selector and return the modified DataFrame

        Parameters
        ----------
            input_matrix: numpy.ndarray
            operator_args: dict
                Dictionary of arguments to be passed to the selector

        Returns
        -------
            modified_df: numpy.ndarrayy

        """
        # Send arguments to selector but also attempt to add in default
        # arguments defined in the Operator class
        op = self._merge_with_default_params(operator_args)

        with warnings.catch_warnings():
            # Ignore warnings about constant features
            warnings.simplefilter('ignore', category=UserWarning)
            op.fit(self.training_features, self.training_classes)

        mask = op.get_support(True)
        np.delete(input_matrix, mask, axis=1)

        return input_matrix
