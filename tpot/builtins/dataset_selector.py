#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 2018

@author: grixor
"""
import numpy as np
import pandas as pd
import os, os.path
from sklearn.base import BaseEstimator, TransformerMixin


class DatasetSelector(BaseEstimator, TransformerMixin):
    """Select predefined data subsets."""

    def __init__(self, subset_dir=None, sel_subset_idx=0):
        """Create a DatasetSelector object.

        Parameters
        ----------
        subset_dir: directory, required
            Path to folder that stores the feature list files. Currently,
            each file needs to be a .csv with one header row. The feature
            names in these files must match those in the (training and
            testing) dataset.
        sel_subset_idx: int, required
            Index of subset

        Returns
        -------
        None

        """
        self.subset_dir = subset_dir
        self.sel_subset_idx = sel_subset_idx

    def fit(self, X, y=None):
        """Fit DatasetSelector for feature selection

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

        self.feature_names = list(X.columns.values)
        subset_files = os.listdir(self.subset_dir)
        self.subset_i = self.subset_dir + "/" + subset_files[self.sel_subset_idx]
        features_i_df = pd.read_csv(self.subset_i, sep='\t', header=0)
        feature_i = [str(val) for val in features_i_df.values.flatten()]
        self.feat_list = list(set(feature_i).intersection(set(self.feature_names)))

        return self

    def transform(self, X):
        """Make subset after fit

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_features}
            New data, where n_samples is the number of samples and n_features is the number of features.

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features + 1) or (n_samples, n_features + 1 + n_classes) for classifier with predict_proba attribute
            The transformed feature set.
        """
        X_transformed = X[self.feat_list].values

        return X_transformed
