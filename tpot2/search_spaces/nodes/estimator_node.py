# try https://automl.github.io/ConfigSpace/main/api/hyperparameters.html

import numpy as np
from ..base import SklearnIndividual, SklearnIndividualGenerator
from ConfigSpace import ConfigurationSpace
from typing import final

NONE_SPECIAL_STRING = "<NONE>"
TRUE_SPECIAL_STRING = "<TRUE>"
FALSE_SPECIAL_STRING = "<FALSE>"


def default_hyperparameter_parser(params:dict) -> dict:
    return params


class EstimatorNodeIndividual(SklearnIndividual):
    """
    Note that ConfigurationSpace does not support None as a parameter. Instead, use the special string "<NONE>". TPOT will automatically replace instances of this string with the Python None. 

    Parameters
    ----------
    method : type
        The class of the estimator to be used

    space : ConfigurationSpace|dict
        The hyperparameter space to be used. If a dict is passed, hyperparameters are fixed and not learned.
    
    """
    def __init__(self, method: type, 
                        space: ConfigurationSpace|dict, #TODO If a dict is passed, hyperparameters are fixed and not learned. Is this confusing? Should we make a second node type?
                        hyperparameter_parser: callable = None,
                        rng=None) -> None:
        super().__init__()
        self.method = method
        self.space = space
        
        if hyperparameter_parser is None:
            self.hyperparameter_parser = default_hyperparameter_parser
        else:
            self.hyperparameter_parser = hyperparameter_parser
        
        if isinstance(space, dict):
            self.hyperparameters = space
        else:
            rng = np.random.default_rng(rng)
            self.space.seed(rng.integers(0, 2**32))
            self.hyperparameters = dict(self.space.sample_configuration())

        self.check_hyperparameters_for_None()

    def mutate(self, rng=None):
        if isinstance(self.space, dict): 
            return False
        
        rng = np.random.default_rng(rng)
        self.space.seed(rng.integers(0, 2**32))
        self.hyperparameters = dict(self.space.sample_configuration())

        self.check_hyperparameters_for_None()
        return True

    def crossover(self, other, rng=None):
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

    @final #this method should not be overridden, instead override hyperparameter_parser
    def export_pipeline(self, **kwargs):
        return self.method(**self.hyperparameter_parser(self.hyperparameters))
    
    def unique_id(self):
        #return a dictionary of the method and the hyperparameters
        method_str = self.method.__name__
        params = list(self.hyperparameters.keys())
        params = sorted(params)

        id_str = f"{method_str}({', '.join([f'{param}={self.hyperparameters[param]}' for param in params])})"
        
        return id_str

class EstimatorNode(SklearnIndividualGenerator):
    def __init__(self, method, space, hyperparameter_parser=default_hyperparameter_parser):
        self.method = method
        self.space = space
        self.hyperparameter_parser = hyperparameter_parser

    def generate(self, rng=None):
        return EstimatorNodeIndividual(self.method, self.space, hyperparameter_parser=self.hyperparameter_parser, rng=rng)