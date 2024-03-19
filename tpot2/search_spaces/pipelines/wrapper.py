
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SklearnIndividualGenerator
from ConfigSpace import ConfigurationSpace


class WrapperPipelineIndividual(SklearnIndividual):
    def __init__(self, 
                 nodegen: SklearnIndividualGenerator, 
                 method: type, 
                space: ConfigurationSpace,
                 rng=None) -> None:



        super().__init__()
        
        self.nodegen = nodegen
        self.node = np.random.default_rng(rng).choice(self.nodegen).generate()


        self.method = method
        self.space = space
        rng = np.random.default_rng(rng)
        self.space.seed(rng.integers(0, 2**32))
        self.hyperparameters = self.space.sample_configuration().get_dictionary()

        
        

    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        if rng.choice([True, False]):
            return self._mutate_hyperparameters(rng)
        else:
            return self._mutate_node(rng)
    
    def _mutate_hyperparameters(self, rng=None):
        rng = np.random.default_rng(rng)
        self.space.seed(rng.integers(0, 2**32))
        self.hyperparameters = self.space.sample_configuration().get_dictionary()
        return True
    
    def _mutate_node(self, rng=None):
        return self.node.mutate(rng)

    def crossover(self, other, rng=None):
        return self.node.crossover(other.node, rng)
    
    def export_pipeline(self):

        est = self.node.export_pipeline()
        wrapped_est = self.method(est, **self.hyperparameters)
        return wrapped_est
    
    
    def unique_id(self):
        return self.node.unique_id()
    

class WrapperPipeline(SklearnIndividualGenerator):
    def __init__(self, nodegen: SklearnIndividualGenerator, 
                 method: type, 
                space: ConfigurationSpace,
                ) -> None:
        
        """
        This search space is for wrapping a sklearn estimator with a method that takes another estimator and hyperparameters as arguments.
        For example, this can be used with sklearn.ensemble.BaggingClassifier or sklearn.ensemble.AdaBoostClassifier.
        
        """


        self.nodegen = nodegen
        self.method = method
        self.space = space

    def generate(self, rng=None):
        return WrapperPipelineIndividual(self.nodegen, self.method, self.space, rng)