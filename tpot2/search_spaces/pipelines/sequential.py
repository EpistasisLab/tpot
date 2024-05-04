import tpot2
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SklearnIndividualGenerator
from ..tuple_index import TupleIndex

class SequentialPipelineIndividual(SklearnIndividual):
    # takes in a list of search spaces. each space is a list of SklearnIndividualGenerators.
    # will produce a pipeline of Sequential length. Each step in the pipeline will correspond to the the search space provided in the same index.

    def __init__(self, search_spaces : List[SklearnIndividualGenerator], rng=None) -> None:
        super().__init__()
        self.search_spaces = search_spaces
        self.pipeline = []

        for space in self.search_spaces:
            self.pipeline.append(space.generate(rng))

        self.pipeline = np.array(self.pipeline)
        
    #TODO, mutate all steps or just one?
    def mutate(self, rng=None):
        rng = np.random.default_rng()
        step = rng.choice(self.pipeline)
        return step.mutate(rng)
     

    def _crossover(self, other, rng=None):
        #swap a random step in the pipeline with the corresponding step in the other pipeline
        if len(self.pipeline) != len(other.pipeline):
            return False
        
        if len(self.pipeline) < 2:
            return False

        rng = np.random.default_rng()
        cx_funcs = [self._crossover_swap_random_steps, self._crossover_swap_segment, self._crossover_inner_step]

        rng.shuffle(cx_funcs)
        for cx_func in cx_funcs:
            if cx_func(other, rng):
                return True
            
        return False

    def _crossover_swap_step(self, other, rng):
        if len(self.pipeline) != len(other.pipeline):
            return False
        
        if len(self.pipeline) < 2:
            return False
        
        rng = np.random.default_rng()
        idx = rng.integers(1,len(self.pipeline))

        self.pipeline[idx], other.pipeline[idx] = other.pipeline[idx], self.pipeline[idx]
        return True
    
    def _crossover_swap_random_steps(self, other, rng):
        rng = np.random.default_rng()
        #selet steps idxs with probability 0.5
        idxs = rng.random(len(self.pipeline)) < 0.5
        #swap steps
        self.pipeline[idxs], other.pipeline[idxs] = other.pipeline[idxs], self.pipeline[idxs]

        return True

    def _crossover_swap_segment(self, other, rng):
        if len(self.pipeline) != len(other.pipeline):
            return False
        
        if len(self.pipeline) < 2:
            return False
        
        rng = np.random.default_rng()
        idx = rng.integers(1,len(self.pipeline))

        left = rng.choice([True, False])
        if left:
            self.pipeline[:idx], other.pipeline[:idx] = other.pipeline[:idx], self.pipeline[:idx]
        else:
            self.pipeline[idx:], other.pipeline[idx:] = other.pipeline[idx:], self.pipeline[idx:]

        return True
    
    def _crossover_inner_step(self, other, rng):
        rng = np.random.default_rng()
        
        crossover_success = False
        for idx in range(len(self.pipeline)):
            if rng.random() < 0.5:
                if self.pipeline[idx].crossover(other.pipeline[idx], rng):
                    crossover_success = True
                
        return crossover_success
    
    def export_pipeline(self):
        return sklearn.pipeline.make_pipeline(*[step.export_pipeline() for step in self.pipeline])
    
    def unique_id(self):
        l = [step.unique_id() for step in self.pipeline]
        l = ["SequentialPipeline"] + l
        return TupleIndex(tuple(l))


class SequentialPipeline(SklearnIndividualGenerator):
    def __init__(self, search_spaces : List[SklearnIndividualGenerator] ) -> None:
        """
        Takes in a list of search spaces. will produce a pipeline of Sequential length. Each step in the pipeline will correspond to the the search space provided in the same index.
        """
        
        self.search_spaces = search_spaces

    def generate(self, rng=None):
        return SequentialPipelineIndividual(self.search_spaces)