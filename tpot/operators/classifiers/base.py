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

import hashlib

from tpot.operators import Operator


class Classifier(Operator):
    """Parent class for classifiers in TPOT"""

    root = True  # Whether this operator type can be the root of the tree

    def _call(self, input_df, *args, **kwargs):
        # Calculate arguments to be passed directly to sklearn
        operator_args = self.preprocess_args(*args, **kwargs)

        return self._train_and_predict(input_df, operator_args)

    def _train_and_predict(self, input_df, operator_args):
        """Fits an arbitrary sklearn classifier model with a set of keyword parameters

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the k-neares
        operator_args: dict
            Input parameters to pass to the model's constructor

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated
            according to the classifier's predictions. Also adds the
            classifiers's predictions as a 'SyntheticFeature' column.

        """

        # Send arguments to classifier but also attempt to add in default
        # arguments defined in the Operator class
        clf = self._merge_with_default_params(operator_args)

        # Fit classifier to the data set
        clf.fit(self.training_features, self.training_classes)

        all_features = input_df.drop(self.non_feature_columns, axis=1).values
        input_df.loc[:, 'guess'] = clf.predict(all_features)

        # Store the guesses as a synthetic feature
        return self._add_synth_feature(input_df, operator_args)

    def _add_synth_feature(self, input_df, operator_args):
        column_names = [str(x) for x in input_df.columns.values.tolist()]

        sf_hash = '-'.join(sorted(column_names)) + \
                  str(self.sklearn_class.__class__) + \
                  '-'.join(operator_args)
        sf_identifier = 'SyntheticFeature-{}'.\
            format(hashlib.sha224(sf_hash.encode('UTF-8')).hexdigest())

        input_df.loc[:, sf_identifier] = input_df['guess'].values

        return input_df
