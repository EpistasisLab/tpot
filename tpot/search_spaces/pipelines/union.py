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
import tpot
import numpy as np
import pandas as pd
import sklearn
from tpot import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SearchSpace
from ..tuple_index import TupleIndex

class UnionPipelineIndividual(SklearnIndividual):
    """
    Takes in a list of search spaces. each space is a list of SearchSpaces.
    Will produce a FeatureUnion pipeline. Each step in the pipeline will correspond to the the search space provided in the same index.
    The resulting pipeline will be a FeatureUnion of the steps in the pipeline.
    
    """

    def __init__(self, search_spaces : List[SearchSpace], rng=None) -> None:
        super().__init__()
        self.search_spaces = search_spaces

        self.pipeline = []
        for space in self.search_spaces:
            self.pipeline.append(space.generate(rng))
    
    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        step = rng.choice(self.pipeline)
        return step.mutate(rng)
     

    def crossover(self, other, rng=None):
        #swap a random step in the pipeline with the corresponding step in the other pipeline
        rng = np.random.default_rng(rng)

        cx_funcs = [self._crossover_node, self._crossover_swap_node]
        rng.shuffle(cx_funcs)
        for cx_func in cx_funcs:
            if cx_func(other, rng):
                return True

        return False
    
    def _crossover_swap_node(self, other, rng):
        rng = np.random.default_rng(rng)
        idx = rng.integers(1,len(self.pipeline))

        self.pipeline[idx], other.pipeline[idx] = other.pipeline[idx], self.pipeline[idx]
        return True

    def _crossover_node(self, other, rng):
        rng = np.random.default_rng(rng)
        
        crossover_success = False
        for idx in range(len(self.pipeline)):
            if rng.random() < 0.5:
                if self.pipeline[idx].crossover(other.pipeline[idx], rng):
                    crossover_success = True
                
        return crossover_success
    
    def export_pipeline(self, **kwargs):
        return sklearn.pipeline.make_union(*[step.export_pipeline(**kwargs) for step in self.pipeline])
    
    def unique_id(self):
        l = [step.unique_id() for step in self.pipeline]
        l = ["FeatureUnion"] + l
        return TupleIndex(tuple(l))


class UnionPipeline(SearchSpace):
    def __init__(self, search_spaces : List[SearchSpace] ) -> None:
        """
        Takes in a list of search spaces. will produce a pipeline of Sequential length. Each step in the pipeline will correspond to the the search space provided in the same index.
        """
        
        self.search_spaces = search_spaces

    def generate(self, rng=None):
        rng = np.random.default_rng(rng)
        return UnionPipelineIndividual(self.search_spaces, rng=rng)