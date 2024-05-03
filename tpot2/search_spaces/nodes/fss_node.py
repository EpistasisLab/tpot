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

from ...builtin_modules.feature_set_selector import FeatureSetSelector

class FSSIndividual(SklearnIndividual):
    def __init__(   self,
                    subsets,
                    rng=None,
                ):

        subsets = subsets
        rng = np.random.default_rng(rng)

        if isinstance(subsets, str):
            df = pd.read_csv(subsets,header=None,index_col=0)
            df['features'] = df.apply(lambda x: list([x[c] for c in df.columns]),axis=1)
            self.subset_dict = {}
            for row in df.index:
                self.subset_dict[row] = df.loc[row]['features']
        elif isinstance(subsets, dict):
            self.subset_dict = subsets
        elif isinstance(subsets, list) or isinstance(subsets, np.ndarray):
            self.subset_dict = {str(i):subsets[i] for i in range(len(subsets))}
        elif isinstance(subsets, int):
            self.subset_dict = {"{0}".format(i):i for i in range(subsets)}
        else:
            raise ValueError("Subsets must be a string, dictionary, list, int, or numpy array")

        self.names_list = list(self.subset_dict.keys())


        self.selected_subset_name = rng.choice(self.names_list)
        self.sel_subset = self.subset_dict[self.selected_subset_name]


    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        self.selected_subset_name = rng.choice(self.names_list)
        self.sel_subset = self.subset_dict[self.selected_subset_name]
        
    
    def crossover(self, other, rng=None):
        self.selected_subset_name = other.selected_subset_name
        self.sel_subset = other.sel_subset

    def export_pipeline(self):
        return FeatureSetSelector(sel_subset=self.sel_subset, name=self.selected_subset_name)
    

    def unique_id(self):
        id_str = "FeatureSetSelector({0})".format(self.selected_subset_name)
        return id_str
    

class FSSNode(SklearnIndividualGenerator):
    def __init__(self,                     
                    subsets,
                    rng=None,
                ):
        
        self.subsets = subsets
        self.rng = rng

    def generate(self, rng=None) -> SklearnIndividual:
        return FSSIndividual(   
            subsets=self.subsets,
            rng=rng,
            )