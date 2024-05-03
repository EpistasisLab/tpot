# try https://automl.github.io/ConfigSpace/main/api/hyperparameters.html
import tpot2
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SklearnIndividualGenerator

class EstimatorNodeIndividual(SklearnIndividual):
    def __init__(self, method, space ) -> None:
        super().__init__()
        self.method = method
        self.space = space #a dictionary. keys are hyperparameters, values are the space of the hyperparameter. If list, then hyperparameter is categorical. If tuple, then hyperparameter is continuous. If single value, then hyperparameter is fixed.
        
        self._mutate_hyperparameters()

    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        return self._mutate_hyperparameters(rng)
    
    def _mutate_hyperparameters(self, rng=None):
        rng = np.random.default_rng(rng)
        self.hyperparameters = {}
        #sample new hyperparameters from the space
        for hyperparameter in self.space:
            hyperparameter_space = self.space[hyperparameter]
            if isinstance(hyperparameter_space, list):
                hp = rng.choice(hyperparameter_space)
            elif isinstance(hyperparameter_space, tuple):
                hp = rng.uniform(hyperparameter_space[0], hyperparameter_space[1])
            else:
                hp = hyperparameter_space

            self.hyperparameters[hyperparameter] = hp
            
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
        method_str = self.method.__name__
        params = list(self.hyperparameters.keys())
        params = sorted(params)

        id_str = f"{method_str}({', '.join([f'{param}={self.hyperparameters[param]}' for param in params])})"
        
        return id_str

class EstimatorNode(SklearnIndividualGenerator):
    def __init__(self, method, space):
        self.method = method
        self.space = space

    def generate(self, rng=None):
        return EstimatorNodeIndividual(self.method, self.space)