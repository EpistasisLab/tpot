import tpot2
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SklearnIndividualGenerator
from ..tuple_index import TupleIndex

class UnionPipelineIndividual(SklearnIndividual):
    """
    Takes in a list of search spaces. each space is a list of SklearnIndividualGenerators.
    Will produce a FeatureUnion pipeline. Each step in the pipeline will correspond to the the search space provided in the same index.
    The resulting pipeline will be a FeatureUnion of the steps in the pipeline.
    
    """

    def __init__(self, search_spaces : List[SklearnIndividualGenerator], rng=None) -> None:
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
    
    def export_pipeline(self):
        return sklearn.pipeline.make_union(*[step.export_pipeline() for step in self.pipeline])
    
    def unique_id(self):
        l = [step.unique_id() for step in self.pipeline]
        l = ["FeatureUnion"] + l
        return TupleIndex(tuple(l))


class UnionPipeline(SklearnIndividualGenerator):
    def __init__(self, search_spaces : List[SklearnIndividualGenerator] ) -> None:
        """
        Takes in a list of search spaces. will produce a pipeline of Sequential length. Each step in the pipeline will correspond to the the search space provided in the same index.
        """
        
        self.search_spaces = search_spaces

    def generate(self, rng=None):
        return UnionPipelineIndividual(self.search_spaces, rng=rng)