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

import warnings

from tpot.operators import Operator


class Selector(Operator):
    """Parent class for Feature Selectors in TPOT"""

    root = False  # Whether this operator type can be the root of the tree

    def _call(self, input_df, *args, **kwargs):
        # Calculate arguments to be passed directly to sklearn
        operator_args = self.preprocess_args(*args, **kwargs)

        # Run the feature-selector with args
        return self._fit_mask(input_df, operator_args)

    def _fit_mask(self, input_df, operator_args):
        """Run the Selector and return the modified DataFrame

        Parameters
        ----------
            input_df: pd.DataFrame
            operator_args: dict
                Dictionary of arguments to be passed to the selector

        Returns
        -------
            modified_df: pd.DataFrame

        """
        # Send arguments to selector but also attempt to add in default
        # arguments defined in the Operator class
        op = self._merge_with_default_params(operator_args)

        training_features_df = input_df.loc[input_df['group'] == 'training'].\
            drop(self.non_feature_columns, axis=1)

        with warnings.catch_warnings():
            # Ignore warnings about constant features
            warnings.simplefilter('ignore', category=UserWarning)
            op.fit(training_features_df, self.training_classes)

        mask = op.get_support(True)
        mask_cols = list(training_features_df.iloc[:, mask].columns) + self.non_feature_columns

        return input_df[mask_cols]
