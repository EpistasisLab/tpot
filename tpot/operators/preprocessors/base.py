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
import pandas as pd
import warnings

from tpot.operators import Operator


class Preprocessor(Operator):
    """Parent class for Feature Preprocessors in TPOT"""

    def _call(self, input_df, *args, **kwargs):
        # Calculate arguments to be passed directly to sklearn
        operator_args = self.preprocess_args(*args, **kwargs)

        # Run the feature-preprocessor with args
        modified_df = self._fit_transform(input_df, operator_args)

        # Add non_feature_columns back to DataFrame
        for non_feature_column in self.non_feature_columns:
            modified_df[non_feature_column] = input_df[non_feature_column].values

        # Translate non-string column titles into strings
        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df

    def _fit_transform(self, input_df, operator_args):
        """Run the Preprocessor and return the modified DataFrame

        Parameters
        ----------
            input_df: pd.DataFrame
            operator_args: dict
                Dictionary of arguments to be passed to the preprocessor

        Returns
        -------
            modified_df: pd.DataFrame

        """
        # Send arguments to preprocessor but also attempt to add in default
        # arguments defined in the Operator class
        op = self._merge_with_default_params(operator_args)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)

            op.fit(self.training_features.values.astype(np.float64))
            transformed_features = op.transform(input_df.drop(self.non_feature_columns, axis=1).
                values.astype(np.float64))

        return pd.DataFrame(data=transformed_features)

    def export(self, *args, **kwargs):
        pass
