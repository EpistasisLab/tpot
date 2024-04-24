
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SklearnIndividualGenerator
from ConfigSpace import ConfigurationSpace


class WrapperPipelineIndividual(SklearnIndividual):
    def __init__(
            self, 
            method: type, 
            space: ConfigurationSpace,
            estimator_search_space: SklearnIndividualGenerator, 
            hyperparameter_parser: callable = None,
            rng=None) -> None:



        super().__init__()
        
        self.estimator_search_space = estimator_search_space
        self.node = self.estimator_search_space.generate(rng)


        self.method = method
        self.space = space
        rng = np.random.default_rng(rng)
        self.space.seed(rng.integers(0, 2**32))
        self.hyperparameters = dict(self.space.sample_configuration())

        self.hyperparameters_parser = hyperparameter_parser
        

    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        if rng.choice([True, False]):
            return self._mutate_hyperparameters(rng)
        else:
            return self._mutate_node(rng)
    
    def _mutate_hyperparameters(self, rng=None):
        rng = np.random.default_rng(rng)
        self.space.seed(rng.integers(0, 2**32))
        self.hyperparameters = dict(self.space.sample_configuration())
        return True
    
    def _mutate_node(self, rng=None):
        return self.node.mutate(rng)

    def crossover(self, other, rng=None):
        return self.node.crossover(other.node, rng)
    
    def export_pipeline(self):
        
        if self.hyperparameters_parser is not None:
            final_params = self.hyperparameters_parser(self.hyperparameters)
        else:
            final_params = self.hyperparameters

        est = self.node.export_pipeline()
        wrapped_est = self.method(est, **final_params)
        return wrapped_est
    
    def unique_id(self):

        if self.hyperparameters_parser is not None:
            final_params = self.hyperparameters_parser(self.hyperparameters)
        else:
            final_params = self.hyperparameters

        return (self.method, str(tuple(sorted(list(final_params.items())))) ,self.node.unique_id())
    

class WrapperPipeline(SklearnIndividualGenerator):
    def __init__(
            self, 
            method: type, 
            space: ConfigurationSpace,
            estimator_search_space: SklearnIndividualGenerator, 
            ) -> None:
        
        """
        This search space is for wrapping a sklearn estimator with a method that takes another estimator and hyperparameters as arguments.
        For example, this can be used with sklearn.ensemble.BaggingClassifier or sklearn.ensemble.AdaBoostClassifier.
        
        """


        self.estimator_search_space = estimator_search_space
        self.method = method
        self.space = space

    def generate(self, rng=None):
        return WrapperPipelineIndividual(method=self.method, space=self.space, estimator_search_space=self.estimator_search_space, rng=rng)