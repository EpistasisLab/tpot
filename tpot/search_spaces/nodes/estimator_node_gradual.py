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
# try https://automl.github.io/ConfigSpace/main/api/hyperparameters.html

import numpy as np
from tpot.search_spaces.base import SklearnIndividual, SearchSpace
from ConfigSpace import ConfigurationSpace
from typing import final
import ConfigSpace

def default_hyperparameter_parser(params:dict) -> dict:
    return params


# NOTE: This is not the default, currently experimental
class EstimatorNodeIndividual_gradual(SklearnIndividual):
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

    def mutate(self, rng=None):
        if isinstance(self.space, dict): 
            return False
        self.hyperparameters = gradual_hyperparameter_update(params=self.hyperparameters, configspace=self.space, rng=rng)
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

        return True


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

def gradual_hyperparameter_update(params:dict, configspace:ConfigurationSpace, rng=None):
    rng = np.random.default_rng(rng)
    configspace.seed(rng.integers(0, 2**32))
    new_params = dict(configspace.sample_configuration())
    for param in list(new_params.keys()):
        #if parameter is float, multiply by normal distribution
        if param not in params:
            continue
        try:
            if issubclass(type(configspace[param]), ConfigSpace.hyperparameters.hyperparameter.FloatHyperparameter):
                
                if configspace[param].log:
                    new_params[param] = params[param] * rng.lognormal(0, 1)
                else:
                    new_params[param] = params[param] + rng.normal(0, .1)* (configspace[param].upper-configspace[param].lower)
                    # if check if above or below min and cap
                if new_params[param] < configspace[param].lower:
                    new_params[param] = configspace[param].lower
                elif new_params[param] > configspace[param].upper:
                    new_params[param] = configspace[param].upper
            #if parameter is integer, add normal distribution
            elif issubclass(type(configspace[param]), ConfigSpace.hyperparameters.hyperparameter.IntegerHyperparameter):
                new_params[param] = params[param] * rng.normal(0, 1)
                # if check if above or below min and cap
                if new_params[param] < configspace[param].lower:
                    new_params[param] = configspace[param].lower
                elif new_params[param] > configspace[param].upper:
                    new_params[param] = configspace[param].upper
                new_params[param] = int(new_params[param])
            # TODO : add support for categorical hyperparameters
            else:
                new_params[param] = params[param]
        except:
            pass

    return new_params

class EstimatorNode_gradual(SearchSpace):
    def __init__(self, method, space, hyperparameter_parser=default_hyperparameter_parser):
        self.method = method
        self.space = space
        self.hyperparameter_parser = hyperparameter_parser

    def generate(self, rng=None):
        return EstimatorNodeIndividual_gradual(self.method, self.space, hyperparameter_parser=self.hyperparameter_parser, rng=rng)