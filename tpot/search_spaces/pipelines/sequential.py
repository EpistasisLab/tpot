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

class SequentialPipelineIndividual(SklearnIndividual):
    # takes in a list of search spaces. each space is a list of SearchSpaces.
    # will produce a pipeline of Sequential length. Each step in the pipeline will correspond to the the search space provided in the same index.

    def __init__(self, search_spaces : List[SearchSpace], rng=None) -> None:
        super().__init__()
        self.search_spaces = search_spaces
        self.pipeline = []

        for space in self.search_spaces:
            self.pipeline.append(space.generate(rng))

        self.pipeline = np.array(self.pipeline)
        
    #TODO, mutate all steps or just one?
    def mutate(self, rng=None):
        # mutated = False
        # for step in self.pipeline:
        #     if rng.random() < 0.5:
        #         if step.mutate(rng):
        #             mutated = True
        # return mutated
        rng = np.random.default_rng(rng)
        step = rng.choice(self.pipeline)
        return step.mutate(rng)
     

    def crossover(self, other, rng=None):
        #swap a random step in the pipeline with the corresponding step in the other pipeline
        if len(self.pipeline) != len(other.pipeline):
            return False

        rng = np.random.default_rng(rng)
        cx_funcs = [self._crossover_swap_multiple_nodes, self._crossover_swap_segment, self._crossover_node]

        rng.shuffle(cx_funcs)
        for cx_func in cx_funcs:
            if cx_func(other, rng):
                return True
            
        return False

    def _crossover_swap_node(self, other, rng):
        if len(self.pipeline) != len(other.pipeline):
            return False
        
        
        rng = np.random.default_rng(rng)
        idx = rng.integers(1,len(self.pipeline))

        self.pipeline[idx], other.pipeline[idx] = other.pipeline[idx], self.pipeline[idx]
        return True
    
    def _crossover_swap_multiple_nodes(self, other, rng):

        if len(self.pipeline) != len(other.pipeline):
            return False
        
        if len(self.pipeline) < 2:
            return False
    
        rng = np.random.default_rng(rng)

        max_steps = int(min(len(self.pipeline), len(other.pipeline))/2)
        max_steps = max(max_steps, 1)
        
        if max_steps == 1:
            n_steps_to_swap = 1
        else:
            n_steps_to_swap = rng.integers(1, max_steps)

        indexes_to_swap = rng.choice(len(other.pipeline), n_steps_to_swap, replace=False)

        for idx in indexes_to_swap:
            self.pipeline[idx], other.pipeline[idx] = other.pipeline[idx], self.pipeline[idx]
        
        
        return True

    def _crossover_swap_segment(self, other, rng):
        if len(self.pipeline) != len(other.pipeline):
            return False
        
        if len(self.pipeline) < 2:
            return False
        
        rng = np.random.default_rng(rng)
        idx = rng.integers(1,len(self.pipeline))

        left = rng.choice([True, False])
        if left:
            self.pipeline[:idx], other.pipeline[:idx] = other.pipeline[:idx], self.pipeline[:idx]
        else:
            self.pipeline[idx:], other.pipeline[idx:] = other.pipeline[idx:], self.pipeline[idx:]

        return True
    
    def _crossover_node(self, other, rng):
        rng = np.random.default_rng(rng)
        
        # crossover_success = False
        # for idx in range(len(self.pipeline)):
        #     if rng.random() < 0.5:
        #         if self.pipeline[idx].crossover(other.pipeline[idx], rng):
        #             crossover_success = True
                
        # return crossover_success


        crossover_success = False
        for idx in range(len(self.pipeline)):
            if rng.random() < 0.5:
                if self.pipeline[idx].crossover(other.pipeline[idx], rng):
                    crossover_success = True
                
        return crossover_success
    
    def export_pipeline(self, memory=None, **kwargs):
        return sklearn.pipeline.make_pipeline(*[step.export_pipeline(memory=memory, **kwargs) for step in self.pipeline], memory=memory)
    
    def unique_id(self):
        l = [step.unique_id() for step in self.pipeline]
        l = ["SequentialPipeline"] + l
        return TupleIndex(tuple(l))
    



class SequentialPipeline(SearchSpace):
    def __init__(self, search_spaces : List[SearchSpace] ) -> None:
        """
        Takes in a list of search spaces. will produce a pipeline of Sequential length. Each step in the pipeline will correspond to the the search space provided in the same index.
        """
        
        self.search_spaces = search_spaces

    def generate(self, rng=None):
        rng = np.random.default_rng(rng)
        return SequentialPipelineIndividual(self.search_spaces, rng=rng)