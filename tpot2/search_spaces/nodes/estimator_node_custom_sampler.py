# try https://automl.github.io/ConfigSpace/main/api/hyperparameters.html
import tpot2
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SklearnIndividualGenerator
from ConfigSpace import ConfigurationSpace

class EstimatorNodeCustomIndividual(SklearnIndividual):
    def __init__(self, method: type, 
                        sample_func : callable, 
                        rng=None) -> None:
        super().__init__()
        self.method = method
        self.sample_func = sample_func
        
        self.hyperparameters = self.sample_func(rng)

    def mutate(self, rng=None):    
        rng = np.random.default_rng(rng)
        self.hyperparameters = self.sample_func(rng)
        return True

    def crossover(self, other, rng=None):
        rng = np.random.default_rng(rng)
        if self.method != other.method:
            return False

        #loop through hyperparameters, randomly swap items in self.hyperparameters with items in other.hyperparameters
        for hyperparameter in self.space:
            if rng.choice([True, False]):
                if hyperparameter in other.hyperparameters:
                    self.hyperparameters[hyperparameter] = other.hyperparameters[hyperparameter]

    def export_pipeline(self, **kwargs):
        return self.method(**self.hyperparameters)
    
    def unique_id(self):
        #return a dictionary of the method and the hyperparameters
        return (self.method, self.hyperparameters)

class EstimatorNodeCustom(SklearnIndividualGenerator):
    def __init__(self,  method : type, 
                        sample_func: callable):
        self.method = method
        self.sample_func = sample_func

    def generate(self, rng=None):
        return EstimatorNodeCustomIndividual(self.method, self.sample_func)