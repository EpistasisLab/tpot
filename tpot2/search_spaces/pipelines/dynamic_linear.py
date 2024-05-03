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

    def __init__(self, search_space : SklearnIndividualGenerator, min_length: int, max_length: int ) -> None:
        super().__init__()

        rng = np.random.default_rng()

        self.search_space = search_space
        self.min_length = min_length
        self.max_length = max_length

        self.pipeline = self._generate_pipeline(rng)

    def _generate_pipeline(self, rng=None):
        rng = np.random.default_rng()
        pipeline = []
        length = rng.integers(self.min_length, self.max_length)
        for _ in range(length):
            pipeline.append(self.search_space.generate(rng))
        return pipeline
    

    def mutate(self, rng=None):
        rng = np.random.default_rng()
        options = []
        if len(self.pipeline) > self.min_length:
            options.append(self._mutate_remove_node)
        if len(self.pipeline) < self.max_length:
            options.append(self._mutate_add_node)
        options.append(self._mutate_step)

        return rng.choice(options)(rng)
    
    def _mutate_add_node(self, rng=None):
        rng = np.random.default_rng()
        new_node = self.search_space.generate(rng)
        idx = rng.integers(len(self.pipeline))
        self.pipeline.insert(idx, new_node)

    def _mutate_remove_node(self, rng=None):
        rng = np.random.default_rng()
        idx = rng.integers(len(self.pipeline))
        self.pipeline.pop(idx)

    def _mutate_step(self, rng=None):
        #choose a random step in the pipeline and mutate it
        rng = np.random.default_rng()
        step = rng.choice(self.pipeline)
        return step.mutate(rng)
    

    def crossover(self, other, rng=None):
        rng = np.random.default_rng()

        if len(self.pipeline) < 2 or len(other.pipeline) < 2:
            return False

        idx = rng.integers(1,len(self.pipeline))
        idx2 = rng.integers(1,len(other.pipeline))
        self.pipeline[idx:] = copy.deepcopy(other.pipeline[idx2:])
        
        return True
    
    def export_pipeline(self, **graph_pipeline_args):
        return [step.export_pipeline(**graph_pipeline_args) for step in self.pipeline]
    
    def unique_id(self):
        l = [step.unique_id() for step in self.pipeline]
        l = ["DynamicLinearPipeline"] + l
        return TupleIndex(tuple(l))
    

class DynamicLinearPipeline(SklearnIndividualGenerator):
    def __init__(self, search_space : SklearnIndividualGenerator, min_length: int, max_length: int ) -> None:
        self.search_space = search_space
        self.min_length = min_length
        self.max_length = max_length

    """
    Takes in a single search space. Will produce a linear pipeline of variable length. Each step in the pipeline will be pulled from the search space provided.

    
    """

    def generate(self, rng=None):
        return DynamicLinearPipelineIndividual(self.search_space, self.min_length, self.max_length)   