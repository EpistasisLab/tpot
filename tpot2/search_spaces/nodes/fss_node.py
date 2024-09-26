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

from ..base import SklearnIndividual, SearchSpace

from ...builtin_modules.feature_set_selector import FeatureSetSelector

class FSSIndividual(SklearnIndividual):
    def __init__(   self,
                    subsets,
                    rng=None,
                ):
        
        """
        An individual for representing a specific FeatureSetSelector. 
        The FeatureSetSelector selects a feature list of list of predefined feature subsets.

        This instance will select one set initially. Mutation and crossover can swap the selected subset with another.

        Parameters
        ----------
        subsets : str or list, default=None
            Sets the subsets that the FeatureSetSeletor will select from if set as an option in one of the configuration dictionaries. 
            Features are defined by column names if using a Pandas data frame, or ints corresponding to indexes if using numpy arrays.
            - str : If a string, it is assumed to be a path to a csv file with the subsets. 
                The first column is assumed to be the name of the subset and the remaining columns are the features in the subset.
            - list or np.ndarray : If a list or np.ndarray, it is assumed to be a list of subsets (i.e a list of lists).
            - dict : A dictionary where keys are the names of the subsets and the values are the list of features.
            - int : If an int, it is assumed to be the number of subsets to generate. Each subset will contain one feature.
            - None : If None, each column will be treated as a subset. One column will be selected per subset.
        rng : int, np.random.Generator, optional
            The random number generator. The default is None.
            Only used to select the first subset.

        Returns
        -------
        None    
        """

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
        #get list of names not including the current one
        names = [name for name in self.names_list if name != self.selected_subset_name]
        self.selected_subset_name = rng.choice(names)
        self.sel_subset = self.subset_dict[self.selected_subset_name]
        
    
    def crossover(self, other, rng=None):
        self.selected_subset_name = other.selected_subset_name
        self.sel_subset = other.sel_subset

    def export_pipeline(self, **kwargs):
        return FeatureSetSelector(sel_subset=self.sel_subset, name=self.selected_subset_name)
    

    def unique_id(self):
        id_str = "FeatureSetSelector({0})".format(self.selected_subset_name)
        return id_str
    

class FSSNode(SearchSpace):
    def __init__(self,                     
                    subsets,
                ):
        """
        A search space for a FeatureSetSelector. 
        The FeatureSetSelector selects a feature list of list of predefined feature subsets.

        Parameters
        ----------
        subsets : str or list, default=None
            Sets the subsets that the FeatureSetSeletor will select from if set as an option in one of the configuration dictionaries. 
            Features are defined by column names if using a Pandas data frame, or ints corresponding to indexes if using numpy arrays.
            - str : If a string, it is assumed to be a path to a csv file with the subsets. 
                The first column is assumed to be the name of the subset and the remaining columns are the features in the subset.
            - list or np.ndarray : If a list or np.ndarray, it is assumed to be a list of subsets (i.e a list of lists).
            - dict : A dictionary where keys are the names of the subsets and the values are the list of features.
            - int : If an int, it is assumed to be the number of subsets to generate. Each subset will contain one feature.
            - None : If None, each column will be treated as a subset. One column will be selected per subset.

        Returns
        -------
        None    
        
        """
        
        self.subsets = subsets

    def generate(self, rng=None) -> SklearnIndividual:
        return FSSIndividual(   
            subsets=self.subsets,
            rng=rng,
            )