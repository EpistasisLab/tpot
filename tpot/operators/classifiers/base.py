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

from tpot.operators import Operator
from tpot.indices import GUESS_COL, non_feature_columns


class Classifier(Operator):
    """Parent class for classifiers in TPOT"""

    root = True  # Whether this operator type can be the root of the tree

    def _call(self, input_matrix, *args, **kwargs):
        # Calculate arguments to be passed directly to sklearn
        operator_args = self.preprocess_args(*args, **kwargs)
        return self._train_and_predict(input_matrix, operator_args)

    def _train_and_predict(self, input_matrix, operator_args):
        """Fits an arbitrary sklearn classifier model with a set of keyword parameters

        Parameters
        ----------
        input_matrix: numpy.ndarray
        operator_args: dict
            Input parameters to pass to the model's constructor

        Returns
        -------
        modified_df: numpy.ndarray
        """

        # Send arguments to classifier but also attempt to add in default
        # arguments defined in the Operator class
        clf = self._merge_with_default_params(operator_args)

        # Fit classifier to the data set
        clf.fit(self.training_features, self.training_classes)

        all_features = np.delete(input_matrix, non_feature_columns, axis=1)

        input_matrix[:, GUESS_COL] = clf.predict(all_features)

        # Store the guesses as a synthetic feature
        input_matrix = np.insert(input_matrix, input_matrix.shape[1], input_matrix[:, GUESS_COL], axis=1)
        
        if self.clf_eval_func == 'decision_function':
            if 'decision_function' in dir(clf):
                input_matrix = np.insert(input_matrix, input_matrix.shape[1], clf.decision_function(all_features).T, axis=1)
            else:
                input_matrix = np.insert(input_matrix, input_matrix.shape[1], np.array([[0.] * max(1, len(np.unique(training_classes))) for x in list(range(input_matrix.shape[0]))]).T, axis=1)
        elif self.clf_eval_func == 'predict_proba':
            if 'predict_proba' in dir(clf):
                input_matrix = np.insert(input_matrix, input_matrix.shape[1], clf.predict_proba(all_features).T, axis=1)
            else:
                input_matrix = np.insert(input_matrix, input_matrix.shape[1], np.array([[0.] * max(1, len(np.unique(training_classes))) for x in list(range(input_matrix.shape[0]))]).T, axis=1)

        return input_matrix
