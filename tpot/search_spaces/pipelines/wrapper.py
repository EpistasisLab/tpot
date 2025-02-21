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

import numpy as np
import pandas as pd
import sklearn
from tpot import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SearchSpace
from ConfigSpace import ConfigurationSpace
from ..tuple_index import TupleIndex


class WrapperPipelineIndividual(SklearnIndividual):
    def __init__(
            self, 
            method: type, 
            space: ConfigurationSpace,
            estimator_search_space: SearchSpace, 
            hyperparameter_parser: callable = None,
            wrapped_param_name: str = None,
            rng=None) -> None:
        super().__init__()

        self.method = method
        self.space = space
        self.estimator_search_space = estimator_search_space
        self.hyperparameters_parser = hyperparameter_parser
        self.wrapped_param_name = wrapped_param_name

        rng = np.random.default_rng(rng)
        self.node = self.estimator_search_space.generate(rng)
        
        if isinstance(space, dict):
            self.hyperparameters = space
        else:
            rng = np.random.default_rng(rng)
            self.space.seed(rng.integers(0, 2**32))
            self.hyperparameters = dict(self.space.sample_configuration())

    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        if rng.choice([True, False]):
            return self._mutate_hyperparameters(rng)
        else:
            return self._mutate_node(rng)
    
    def _mutate_hyperparameters(self, rng=None):
        if isinstance(self.space, dict): 
            return False
        rng = np.random.default_rng(rng)
        self.space.seed(rng.integers(0, 2**32))
        self.hyperparameters = dict(self.space.sample_configuration())
        return True
    
    def _mutate_node(self, rng=None):
        return self.node.mutate(rng)

    def crossover(self, other, rng=None):
        rng = np.random.default_rng(rng)
        if rng.choice([True, False]):
            return self._crossover_hyperparameters(other, rng)
        else:
            self.node.crossover(other.estimator_search_space, rng)
    

    def _crossover_hyperparameters(self, other, rng=None):
        if isinstance(self.space, dict):
            return False
        
        rng = np.random.default_rng(rng)
        if self.method != other.method:
            return False

        #loop through hyperparameters, randomly swap items in self.hyperparameters with items in other.hyperparameters
        for hyperparameter in self.space:
            if rng.choice([True, False]):
                if hyperparameter in other.hyperparameters:
                    self.hyperparameters[hyperparameter] = other.hyperparameters[hyperparameter]

        return True


    def export_pipeline(self, **kwargs):
        
        if self.hyperparameters_parser is not None:
            final_params = self.hyperparameters_parser(self.hyperparameters)
        else:
            final_params = self.hyperparameters

        est = self.node.export_pipeline(**kwargs)
        wrapped_est = self.method(est, **final_params)
        return wrapped_est
    


    def unique_id(self):
        #return a dictionary of the method and the hyperparameters
        method_str = self.method.__name__
        params = list(self.hyperparameters.keys())
        params = sorted(params)

        id_str = f"{method_str}({', '.join([f'{param}={self.hyperparameters[param]}' for param in params])})"
        
        return TupleIndex(("WrapperPipeline", id_str, self.node.unique_id()))
    

class WrapperPipeline(SearchSpace):
    def __init__(
            self, 
            method: type, 
            space: ConfigurationSpace,
            estimator_search_space: SearchSpace,
            hyperparameter_parser: callable = None, 
            wrapped_param_name: str = None
            ) -> None:
        
        """
        This search space is for wrapping a sklearn estimator with a method that takes another estimator and hyperparameters as arguments.
        For example, this can be used with sklearn.ensemble.BaggingClassifier or sklearn.ensemble.AdaBoostClassifier.
        
        """


        self.estimator_search_space = estimator_search_space
        self.method = method
        self.space = space
        self.hyperparameter_parser=hyperparameter_parser
        self.wrapped_param_name = wrapped_param_name

    def generate(self, rng=None):
        rng = np.random.default_rng(rng)
        return WrapperPipelineIndividual(method=self.method, space=self.space, estimator_search_space=self.estimator_search_space, hyperparameter_parser=self.hyperparameter_parser, wrapped_param_name=self.wrapped_param_name,  rng=rng)