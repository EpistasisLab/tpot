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

    def __init__(self, search_space : SklearnIndividualGenerator, max_estimators=None, allow_repeats=False, rng=None) -> None:
        super().__init__()
        self.search_space = search_space
        
        if max_estimators is None:
            self.max_estimators = np.inf
        else:
            self.max_estimators = max_estimators

        self.allow_repeats = allow_repeats

        self.union_dict = {}
        
        if self.max_estimators == np.inf:
            init_max = 3
        else:
            init_max = self.max_estimators

        rng = np.random.default_rng(rng)

        for _ in range(rng.integers(1, init_max)):
            self._mutate_add_step(rng)
            
    
    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        mutation_funcs = [self._mutate_add_step, self._mutate_remove_step, self._mutate_replace_step, self._mutate_note]
        rng.shuffle(mutation_funcs)
        for mutation_func in mutation_funcs:
            if mutation_func(rng):
                return True
    
    def _mutate_add_step(self, rng):
        rng = np.random.default_rng(rng)
        max_attempts = 10
        if len(self.union_dict) < self.max_estimators:
            for _ in range(max_attempts):
                new_step = self.search_space.generate(rng)
                if new_step.unique_id() not in self.union_dict:
                    self.union_dict[new_step.unique_id()] = new_step
                    return True
        return False
    
    def _mutate_remove_step(self, rng):
        rng = np.random.default_rng(rng)
        if len(self.union_dict) > 1:
            self.union_dict.pop( rng.choice(list(self.union_dict.keys())))  
            return True
        return False

    def _mutate_replace_step(self, rng):
        rng = np.random.default_rng(rng)        
        changed = self._mutate_remove_step(rng) or self._mutate_add_step(rng)
        return changed
    
    #TODO mutate one step or multiple?
    def _mutate_note(self, rng):
        rng = np.random.default_rng(rng)
        changed = False
        values = list(self.union_dict.values())
        for step in values:
            if rng.random() < 0.5:
                changed = step.mutate(rng) or changed
        
        self.union_dict = {step.unique_id(): step for step in values}

        return changed


    def crossover(self, other, rng=None):
        rng = np.random.default_rng(rng)

        cx_funcs = [self._crossover_swap_multiple_nodes, self._crossover_node]
        rng.shuffle(cx_funcs)
        for cx_func in cx_funcs:
            if cx_func(other, rng):
                return True

        return False

            
    def _crossover_swap_multiple_nodes(self, other, rng):
        rng = np.random.default_rng(rng)
        self_values = list(self.union_dict.values())
        other_values = list(other.union_dict.values())

        rng.shuffle(self_values)
        rng.shuffle(other_values)

        self_idx = rng.integers(0,len(self_values))
        other_idx = rng.integers(0,len(other_values))

        #Note that this is not one-point-crossover since the sequence doesn't matter. this is just a quick way to swap multiple random items
        self_values[:self_idx], other_values[:other_idx] = other_values[:other_idx], self_values[:self_idx]
        
        self.union_dict = {step.unique_id(): step for step in self_values}
        other.union_dict = {step.unique_id(): step for step in other_values}

        return True


    def _crossover_node(self, other, rng):
        rng = np.random.default_rng(rng)
        
        changed = False
        self_values = list(self.union_dict.values())
        other_values = list(other.union_dict.values())

        rng.shuffle(self_values)
        rng.shuffle(other_values)

        for self_step, other_step in zip(self_values, other_values):
            if rng.random() < 0.5:
                changed = self_step.crossover(other_step, rng) or changed

        self.union_dict = {step.unique_id(): step for step in self_values}
        other.union_dict = {step.unique_id(): step for step in other_values}

        return changed

    def export_pipeline(self):
        values = list(self.union_dict.values())
        return sklearn.pipeline.make_union(*[step.export_pipeline() for step in values])
    
    def unique_id(self):
        values = list(self.union_dict.values())
        l = [step.unique_id() for step in values]
        # if all items are strings, then sort them
        if all([isinstance(x, str) for x in l]):
            l.sort()
        l = ["FeatureUnion"] + l
        return TupleIndex(frozenset(l))

class DynamicUnionPipeline(SklearnIndividualGenerator):
    def __init__(self, search_spaces : List[SklearnIndividualGenerator],max_estimators=None, allow_repeats=False ) -> None:
        """
        Takes in a list of search spaces. will produce a pipeline of Sequential length. Each step in the pipeline will correspond to the the search space provided in the same index.
        """
        
        self.search_spaces = search_spaces
        self.max_estimators = max_estimators
        self.allow_repeats = allow_repeats

    def generate(self, rng=None):
        return DynamicUnionPipelineIndividual(self.search_spaces, max_estimators=self.max_estimators, allow_repeats=self.allow_repeats, rng=rng)