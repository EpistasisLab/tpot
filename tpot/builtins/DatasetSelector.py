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


class DatasetSelector(BaseEstimator):
    """Select predefined data subsets."""

    def __init__(self, subset_dir=None):
        """Create a DatasetSelector object.

        Parameters
        ----------
        subset_dir: directory, required
            Path to folder that stores the feature list files. Currently,
            each file needs to be a .csv with one header row. The feature
            names in these files must match those in the (training and
            testing) dataset.

        Returns
        -------
        None

        """
        self.subset_dir = subset_dir

    def get_subset(self, input_data, input_target):
        """Fit an optimized machine learning pipeline using TPOT.

        Uses genetic programming to optimize a machine learning pipeline that
        maximizes score on the provided features and target. Performs internal
        k-fold cross-validaton to avoid overfitting on the provided data. The
        best pipeline is then trained on the entire set of provided samples.

        Parameters
        ----------
        input_data: array-like {n_samples, n_features}
            Feature matrix

        input_target: array-like {n_samples}
            List of class labels for prediction

        Returns
        -------
        self.data_subset: object
            Returns a list of subsets of input_data

        """

        self.input_data = input_data
        self.input_target = input_target
        self.feature_names = list(self.input_data.columns.values)

        self.subset_files = os.listdir(self.subset_dir)
        self.num_subset = len(self.subset_files)
        self.feature_set = {}
        self.data_subset = {}
        self.population_size = population_size

        for i in range(self.num_subset):
            self.subset_i = self.subset_dir + "/" + self.subset_files[i]
            self.features_i_df = pd.read_csv(self.subset_i, sep='\t', header=0)
            self.feature_i = set(features_i_df.values.flatten())
            self.feature_set[i] = list(feature_i.intersection(set(self.feature_names)))
            self.data_subset[i] = self.input_data[self.feature_set[i]]

        return self.data_subset
