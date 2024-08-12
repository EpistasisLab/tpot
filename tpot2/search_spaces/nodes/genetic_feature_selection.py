from numpy import iterable
import tpot2
import numpy as np
import sklearn
import sklearn.datasets
import numpy as np

import pandas as pd
import os, os.path
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin

from ..base import SklearnIndividual, SklearnIndividualGenerator

class MaskSelector(BaseEstimator, SelectorMixin):
    """Select predefined feature subsets."""

    def __init__(self, mask, set_output_transform=None):
        self.mask = mask
        self.set_output_transform = set_output_transform
        if set_output_transform is not None:
            self.set_output(transform=set_output_transform)

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns
        #     self.set_output(transform="pandas")
        self.is_fitted_ = True #so sklearn knows it's fitted
        return self

    def _get_tags(self):
        tags = {"allow_nan": True, "requires_y": False}
        return tags

    def _get_support_mask(self):
        return np.array(self.mask)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.get_support()]

class GeneticFeatureSelectorIndividual(SklearnIndividual):
    def __init__(   self,
                    mask,
                    start_p=0.2,
                    mutation_rate = 0.5,
                    crossover_rate = 0.5,
                    rng=None,
                ):

        self.start_p = start_p
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate_rate = 0
        self.crossover_rate_rate = 0



        rng = np.random.default_rng(rng)

        if isinstance(mask, int):
            #list of random bollean values
            self.mask = rng.choice([True, False], size=mask, p=[self.start_p,1-self.start_p])
        else:
            self.mask = mask

        # check if there are no features selected, if so select one
        if sum(self.mask) == 0:
            index = rng.choice(len(self.mask))
            self.mask[index] = True

        self.mutation_list = [self._mutate_add, self._mutate_remove]
        self.crossover_list = [self._crossover_swap]


    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        
        if rng.uniform() < self.mutation_rate_rate:
            self.mutation_rate = self.mutation_rate * rng.uniform(0.5, 2)
            self.mutation_rate = min(self.mutation_rate, 2)
            self.mutation_rate = max(self.mutation_rate, 1/len(self.mask))
        
        return rng.choice(self.mutation_list)(rng)
    
    def crossover(self, other, rng=None):
        rng = np.random.default_rng(rng)
        
        if rng.uniform() < self.crossover_rate_rate:
            self.crossover_rate = self.crossover_rate * rng.uniform(0.5, 2)
            self.crossover_rate = min(self.crossover_rate, .6)
            self.crossover_rate = max(self.crossover_rate, 1/len(self.mask))
        
        return rng.choice(self.crossover_list)(other, rng)


    # def _mutate_add(self, rng=None):
    #     rng = np.random.default_rng(rng)

    #     add_mask = rng.choice([True, False], size=self.mask.shape, p=[self.mutation_rate,1-self.mutation_rate])
    #     self.mask = np.logical_or(self.mask, add_mask)
    #     return True

    # def _mutate_remove(self, rng=None):
    #     rng = np.random.default_rng(rng)

    #     add_mask = rng.choice([False, True], size=self.mask.shape, p=[self.mutation_rate,1-self.mutation_rate])
    #     self.mask = np.logical_and(self.mask, add_mask)
    #     return True

    def _mutate_add(self, rng=None):
        rng = np.random.default_rng(rng)

        num_pos = np.sum(self.mask)
        num_neg = len(self.mask) - num_pos

        if num_neg == 0:
            return False

        to_add = int(self.mutation_rate * num_pos)
        to_add = max(to_add, 1)

        p = to_add / num_neg
        p = min(p, 1)

        add_mask = rng.choice([True, False], size=self.mask.shape, p=[p,1-p])
        if sum(np.logical_or(self.mask, add_mask)) == 0:
            pass
        self.mask = np.logical_or(self.mask, add_mask)
        return True

    def _mutate_remove(self, rng=None):
        rng = np.random.default_rng(rng)

        num_pos = np.sum(self.mask)
        if num_pos == 1:
            return False

        num_neg = len(self.mask) - num_pos

        to_remove = int(self.mutation_rate * num_pos)
        to_remove = max(to_remove, 1)

        p = to_remove / num_pos
        p = min(p, .5)

        remove_mask = rng.choice([True, False], size=self.mask.shape, p=[p,1-p])
        self.mask = np.logical_and(self.mask, remove_mask)


        if sum(self.mask) == 0:
            index = rng.choice(len(self.mask))
            self.mask[index] = True

        return True

    def _crossover_swap(self, ss2, rng=None):
        rng = np.random.default_rng(rng)
        mask = rng.choice([True, False], size=self.mask.shape, p=[self.crossover_rate,1-self.crossover_rate])

        self.mask = np.where(mask, self.mask, ss2.mask)
    
    def export_pipeline(self):
        return MaskSelector(mask=self.mask)
    

    def unique_id(self):
        mask_idexes = np.where(self.mask)[0]
        id_str = ','.join([str(i) for i in mask_idexes])
        return id_str
    

class GeneticFeatureSelectorNode(SklearnIndividualGenerator):
    def __init__(self,                     
                    n_features,
                    start_p=0.2,
                    mutation_rate = 0.5,
                    crossover_rate = 0.5,
                    mutation_rate_rate = 0,
                    crossover_rate_rate = 0,
                    ):
        
        self.n_features = n_features
        self.start_p = start_p
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate_rate = mutation_rate_rate
        self.crossover_rate_rate = crossover_rate_rate


    def generate(self, rng=None) -> SklearnIndividual:
        return GeneticFeatureSelectorIndividual(   mask=self.n_features,
                                                    start_p=self.start_p,
                                                    mutation_rate=self.mutation_rate,
                                                    crossover_rate=self.crossover_rate,
                                                    mutation_rate_rate=self.mutation_rate_rate,
                                                    crossover_rate_rate=self.crossover_rate_rate,
                                                    rng=rng
                                                )