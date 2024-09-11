#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.
"""
#TODO handle sparse input?

import numpy as np
import pandas as pd
import os, os.path
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin



#TODO clean this up and make sure it works
class FeatureSetSelector(BaseEstimator, SelectorMixin):
    """Select predefined feature subsets."""

    def __init__(self, sel_subset=None, name=None):
        """Create a FeatureSetSelector object.

        Parameters
        ----------
        sel_subset: list or int
            If X is a dataframe, items in sel_subset list must correspond to column names
            If X is a numpy array, items in sel_subset list must correspond to column indexes
            int: index of a single column
        Returns
        -------
        None

        """
        self.name = name
        self.sel_subset = sel_subset

        
    def fit(self, X, y=None):
        """Fit FeatureSetSelector for feature selection

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like, shape (n_samples,)
            The target values (integers that correspond to classes in classification, real numbers in regression).

        Returns
        -------
        self: object
            Returns a copy of the estimator
        """
        if isinstance(self.sel_subset, int) or isinstance(self.sel_subset, str):
            self.sel_subset = [self.sel_subset]

        #generate  self.feat_list_idx
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self.feat_list_idx = sorted([self.feature_names_in_.index(feat) for feat in self.sel_subset])

            
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = None#list(range(X.shape[1]))

            self.feat_list_idx = sorted(self.sel_subset)
        
        n_features = X.shape[1]
        self.mask = np.zeros(n_features, dtype=bool)
        self.mask[np.asarray(self.feat_list_idx)] = True

        return self

    #TODO keep returned as dataframe if input is dataframe? may not be consistent with sklearn

    # def transform(self, X):
    
    def _get_tags(self):
        tags = {"allow_nan": True, "requires_y": False}
        return tags

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected
        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """
        return self.mask

