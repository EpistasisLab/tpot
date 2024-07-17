import tpot2
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SklearnIndividualGenerator

import copy
from ..tuple_index import TupleIndex

class DynamicLinearPipelineIndividual(SklearnIndividual):
    # takes in a single search space.
    # will produce a pipeline of variable length. Each step in the pipeline will be pulled from the search space provided.

    def __init__(self, search_space : SklearnIndividualGenerator, max_length: int , rng=None) -> None:
        super().__init__()

        rng = np.random.default_rng(rng)

        self.search_space = search_space
        self.min_length = 1
        self.max_length = max_length

        self.pipeline = self._generate_pipeline(rng)

    def _generate_pipeline(self, rng=None):
        rng = np.random.default_rng(rng)
        pipeline = []
        length = rng.integers(self.min_length, self.max_length)
        length = min(length, 3)
        
        for _ in range(length):
            pipeline.append(self.search_space.generate(rng))
        return pipeline
    

    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        options = []
        if len(self.pipeline) > self.min_length:
            options.append(self._mutate_remove_node)
        if len(self.pipeline) < self.max_length:
            options.append(self._mutate_add_node)
        options.append(self._mutate_step)

        return rng.choice(options)(rng)
    
    def _mutate_add_node(self, rng=None):
        rng = np.random.default_rng(rng)
        new_node = self.search_space.generate(rng)
        idx = rng.integers(len(self.pipeline))
        self.pipeline.insert(idx, new_node)

    def _mutate_remove_node(self, rng=None):
        rng = np.random.default_rng(rng)
        idx = rng.integers(len(self.pipeline))
        self.pipeline.pop(idx)

    def _mutate_step(self, rng=None):
        #choose a random step in the pipeline and mutate it
        rng = np.random.default_rng(rng)
        step = rng.choice(self.pipeline)
        return step.mutate(rng)
    

    def crossover(self, other, rng=None):
        #swap a random step in the pipeline with the corresponding step in the other pipeline

        rng = np.random.default_rng(rng)
        cx_funcs = [self._crossover_swap_multiple_nodes, self._crossover_node]

        rng.shuffle(cx_funcs)
        for cx_func in cx_funcs:
            if cx_func(other, rng):
                return True
            
        return False
    
    def _crossover_swap_multiple_nodes(self, other, rng):
        rng = np.random.default_rng(rng)

        max_steps = int(min(len(self.pipeline), len(other.pipeline))/2)
        max_steps = max(max_steps, 1)
        
        if max_steps == 1:
            n_steps_to_swap = 1
        else:
            n_steps_to_swap = rng.integers(1, max_steps)

        other_indexes_to_take = rng.choice(len(other.pipeline), n_steps_to_swap, replace=False)
        self_indexes_to_replace = rng.choice(len(self.pipeline), n_steps_to_swap, replace=False)

        # self.pipeline[self_indexes_to_replace], other.pipeline[other_indexes_to_take] = other.pipeline[other_indexes_to_take], self.pipeline[self_indexes_to_replace]
        
        for self_idx, other_idx in zip(self_indexes_to_replace, other_indexes_to_take):
            self.pipeline[self_idx], other.pipeline[other_idx] = other.pipeline[other_idx], self.pipeline[self_idx]
        
        return True

    def _crossover_swap_node(self, other, rng):
        if len(self.pipeline) != len(other.pipeline):
            return False
        
        if len(self.pipeline) < 2:
            return False
        
        rng = np.random.default_rng(rng)
        idx = rng.integers(1,len(self.pipeline))

        self.pipeline[idx], other.pipeline[idx] = other.pipeline[idx], self.pipeline[idx]
        return True

    def _crossover_node(self, other, rng):
        rng = np.random.default_rng(rng)
        
        pipeline1_indexes= list(range(len(self.pipeline)))
        pipeline2_indexes= list(range(len(other.pipeline)))

        rng.shuffle(pipeline1_indexes)
        rng.shuffle(pipeline2_indexes)

        crossover_success = False
        for idx1, idx2 in zip(pipeline1_indexes, pipeline2_indexes):
                if self.pipeline[idx1].crossover(other.pipeline[idx2], rng):
                    crossover_success = True
        return crossover_success
    
    def export_pipeline(self):
        return sklearn.pipeline.make_pipeline(*[step.export_pipeline() for step in self.pipeline])
    
    def unique_id(self):
        l = [step.unique_id() for step in self.pipeline]
        l = ["DynamicLinearPipeline"] + l
        return TupleIndex(tuple(l))
    

class DynamicLinearPipeline(SklearnIndividualGenerator):
    def __init__(self, search_space : SklearnIndividualGenerator, max_length: int ) -> None:
        self.search_space = search_space
        self.max_length = max_length

    """
    Takes in a single search space. Will produce a linear pipeline of variable length. Each step in the pipeline will be pulled from the search space provided.

    
    """

    def generate(self, rng=None):
        return DynamicLinearPipelineIndividual(self.search_space, self.max_length, rng=rng)   