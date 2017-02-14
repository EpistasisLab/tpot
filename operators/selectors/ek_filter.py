# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 13:55:02 2017

@author: ansohn
"""

from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd


class EKF_Source(BaseEstimator):
    

    def __init__(self, expert_source=None, ekf_index=None, k_best=None):
        self.expert_source = expert_source
        self.ekf_index = ekf_index
        self.k_best = k_best
        self.top_features_ = None
        self.top_features_indices_ = None
        self.ekf_subset_ = None

    def fit(self, input_data, input_target):
        self.input_data = input_data
        self.input_target = input_target

        self.ekf_subset_ = pd.read_csv(self.expert_source[self.ekf_index], sep='\t', header=0)
        if len(self.ekf_subset_.columns) == 3:
            if set(self.ekf_subset_['Score']) in [set([True, False]), set([True]), set([False])]:
                self.top_features_ = (self.ekf_subset_)[(self.ekf_subset_)['Score'] == 1]

                self.top_features_indices_ = self.top_features_.index.values
            else:
            # Assume higher feature importance score means it's a better feature
                self.ekf_subset_ = self.ekf_subset_.sort_values(['Score'], ascending=False)
                self.ekf_subset_ = self.ekf_subset_[:self.k_best]
                self.top_features_ = self.ekf_subset_['Gene']
                
                self.top_features_indices_ = self.top_features_.index.values
        else:
            self.top_features_indices_ = np.genfromtxt((self.expert_source[self.ekf_index]), 
                                                        skip_header=0, dtype=np.int32)[:self.k_best]
#            self.ekf_subset_ = self.ekf_subset_[:self.k_best]
#            self.top_features_indices_ = self.ekf_subset_

        return self

        
    def transform(self, input_data):
        return input_data[:, self.top_features_indices_]
        
        
    def fit_transform(self, input_data, input_target):
        self.fit(input_data, input_target)
        return self.transform(input_data)
