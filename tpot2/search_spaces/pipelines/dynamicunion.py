import tpot2
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SklearnIndividualGenerator
from ..tuple_index import TupleIndex

class DynamicUnionPipelineIndividual(SklearnIndividual):
    """
    Takes in one search space.
    Will produce a FeatureUnion of up to max_estimators number of steps.
    The output of the FeatureUnion will the all of the steps concatenated together.
    
    """

    def __init__(self, search_space : SklearnIndividualGenerator, max_estimators=None, rng=None) -> None:
        super().__init__()
        self.search_space = search_space
        
        if max_estimators is None:
            self.max_estimators = np.inf
        else:
            self.max_estimators = max_estimators

        self.pipeline = []
        
        if self.max_estimators == np.inf:
            init_max = 3
        else:
            init_max = self.max_estimators

        rng = np.random.default_rng(rng)

        for _ in range(rng.integers(1, init_max)):
            self.pipeline.append(self.search_space.generate(rng))
    
    def mutate(self, rng=None):
        rng = np.random.default_rng()
        mutation_funcs = [self._mutate_add_step, self._mutate_remove_step, self._mutate_replace_step, self._mutate_inner_step]
        rng.shuffle(mutation_funcs)
        for mutation_func in mutation_funcs:
            if mutation_func(rng):
                return True
    
    def _mutate_add_step(self, rng):
        rng = np.random.default_rng()
        if len(self.pipeline) < self.max_estimators:
            self.pipeline.append(self.search_space.generate(rng))
            return True
        return False
    
    def _mutate_remove_step(self, rng):
        rng = np.random.default_rng()
        if len(self.pipeline) > 1:
            self.pipeline.pop(rng.integers(0, len(self.pipeline)))
            return True
        return False

    def _mutate_replace_step(self, rng):
        rng = np.random.default_rng()
        idx = rng.integers(0, len(self.pipeline))
        self.pipeline[idx] = self.search_space.generate(rng)
        return True
    
    def _mutate_inner_step(self, rng):
        rng = np.random.default_rng()
        indexes = rng.random(len(self.pipeline)) < 0.5
        indexes = np.where(indexes)[0]
        mutated = False
        if len(indexes) > 0:
            for idx in indexes:
                if self.pipeline[idx].mutate(rng):
                    mutated = True
        else:
            mutated = self.pipeline[rng.integers(0, len(self.pipeline))].mutate(rng)

        return mutated


    def crossover(self, other, rng=None):
        rng = np.random.default_rng()

        cx_funcs = [self._crossover_swap_random_steps, self._crossover_inner_step]
        rng.shuffle(cx_funcs)
        for cx_func in cx_funcs:
            if cx_func(other, rng):
                return True

        return False
    
    def _crossover_swap_step(self, other, rng):
        rng = np.random.default_rng()
        idx = rng.integers(1,len(self.pipeline))
        idx2 = rng.integers(1,len(other.pipeline))

        self.pipeline[idx], other.pipeline[idx2] = other.pipeline[idx2], self.pipeline[idx]
        # self.pipeline[idx] = other.pipeline[idx2]
        return True
    
    def _crossover_swap_random_steps(self, other, rng):
        rng = np.random.default_rng()

        max_steps = int(min(len(self.pipeline), len(other.pipeline))/2)
        max_steps = max(max_steps, 1)
        
        n_steps_to_swap = rng.integers(1, max_steps)

        other_indexes_to_take = rng.choice(len(other.pipeline), n_steps_to_swap, replace=False)
        self_indexes_to_replace = rng.choice(len(self.pipeline), n_steps_to_swap, replace=False)

        self.pipeline[self_indexes_to_replace], other.pipeline[other_indexes_to_take] = other.pipeline[other_indexes_to_take], self.pipeline[self_indexes_to_replace]
        return True
        


    def _crossover_inner_step(self, other, rng):
        rng = np.random.default_rng()
        
        #randomly select pairs of steps to crossover
        indexes = list(range(1, len(self.pipeline)))
        other_indexes = list(range(1, len(other.pipeline)))
        #shuffle
        rng.shuffle(indexes)
        rng.shuffle(other_indexes)

        crossover_success = False
        for idx, other_idx in zip(indexes, other_indexes):
            if self.pipeline[idx].crossover(other.pipeline[other_idx], rng):
                crossover_success = True
                
        return crossover_success
    
    def export_pipeline(self):
        return sklearn.pipeline.make_pipeline(*[step.export_pipeline() for step in self.pipeline])
    
    def unique_id(self):
        l = [step.unique_id() for step in self.pipeline]
        # if all items are strings, then sort them
        if all([isinstance(x, str) for x in l]):
            l.sort()
        l = ["FeatureUnion"] + l
        return TupleIndex(tuple(l))


class DynamicUnionPipeline(SklearnIndividualGenerator):
    def __init__(self, search_spaces : List[SklearnIndividualGenerator] ) -> None:
        """
        Takes in a list of search spaces. will produce a pipeline of Sequential length. Each step in the pipeline will correspond to the the search space provided in the same index.
        """
        
        self.search_spaces = search_spaces

    def generate(self, rng=None):
        return DynamicUnionPipelineIndividual(self.search_spaces)