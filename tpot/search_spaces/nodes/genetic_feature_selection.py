"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Gabriel Ketron (gabriel.ketron@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

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
from numpy import iterable
import tpot
import numpy as np
import sklearn
import sklearn.datasets
import numpy as np

import pandas as pd
import os, os.path
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin

from ..base import SklearnIndividual, SearchSpace

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
                    mutation_rate_rate = 0,
                    crossover_rate_rate = 0,
                    rng=None,
                ):

        self.start_p = start_p
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate_rate = mutation_rate_rate
        self.crossover_rate_rate = crossover_rate_rate



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
    
    def export_pipeline(self,  **kwargs):
        return MaskSelector(mask=self.mask)
    

    def unique_id(self):
        mask_idexes = np.where(self.mask)[0]
        id_str = ','.join([str(i) for i in mask_idexes])
        return id_str
    

class GeneticFeatureSelectorNode(SearchSpace):
    def __init__(self,                     
                    n_features,
                    start_p=0.2,
                    mutation_rate = 0.1,
                    crossover_rate = 0.1,
                    mutation_rate_rate = 0, # These are still experimental but seem to help. Theory is that it takes slower steps as it gets closer to the optimal solution.
                    crossover_rate_rate = 0,# Otherwise is mutation_rate is too small, it takes forever, and if its too large, it never converges.
                    ):
        """
        A node that generates a GeneticFeatureSelectorIndividual. Uses genetic algorithm to select novel subsets of features.

        Parameters
        ----------
        n_features : int
            Number of features in the dataset.
        start_p : float
            Probability of selecting a given feature for the initial subset of features.
        mutation_rate : float
            Probability of adding/removing a feature from the subset of features.
        crossover_rate : float
            Probability of swapping a feature between two subsets of features.
        mutation_rate_rate : float
            Probability of changing the mutation rate. (experimental)
        crossover_rate_rate : float
            Probability of changing the crossover rate. (experimental)
        
        """
        
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