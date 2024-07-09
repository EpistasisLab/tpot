
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SklearnIndividualGenerator
from ConfigSpace import ConfigurationSpace
from ..tuple_index import TupleIndex

NONE_SPECIAL_STRING = "<NONE>"
TRUE_SPECIAL_STRING = "<TRUE>"
FALSE_SPECIAL_STRING = "<FALSE>"


class WrapperPipelineIndividual(SklearnIndividual):
    def __init__(
            self, 
            method: type, 
            space: ConfigurationSpace,
            estimator_search_space: SklearnIndividualGenerator, 
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

        self.check_hyperparameters_for_None()

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
        self.check_hyperparameters_for_None()
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

        self.check_hyperparameters_for_None()

        return True

    def check_hyperparameters_for_None(self):
        for key, value in self.hyperparameters.items():
            #if string
            if isinstance(value, str):
                if value == NONE_SPECIAL_STRING:
                    self.hyperparameters[key] = None
                elif value == TRUE_SPECIAL_STRING:
                    self.hyperparameters[key] = True
                elif value == FALSE_SPECIAL_STRING:
                    self.hyperparameters[key] = False


    def export_pipeline(self):
        
        if self.hyperparameters_parser is not None:
            final_params = self.hyperparameters_parser(self.hyperparameters)
        else:
            final_params = self.hyperparameters

        est = self.node.export_pipeline()
        wrapped_est = self.method(est, **final_params)
        return wrapped_est
    


    def unique_id(self):
        #return a dictionary of the method and the hyperparameters
        method_str = self.method.__name__
        params = list(self.hyperparameters.keys())
        params = sorted(params)

        id_str = f"{method_str}({', '.join([f'{param}={self.hyperparameters[param]}' for param in params])})"
        
        return TupleIndex(("WrapperPipeline", id_str, self.node.unique_id()))
    

class WrapperPipeline(SklearnIndividualGenerator):
    def __init__(
            self, 
            method: type, 
            space: ConfigurationSpace,
            estimator_search_space: SklearnIndividualGenerator,
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
        return WrapperPipelineIndividual(method=self.method, space=self.space, estimator_search_space=self.estimator_search_space, hyperparameter_parser=self.hyperparameter_parser, wrapped_param_name=self.wrapped_param_name,  rng=rng)