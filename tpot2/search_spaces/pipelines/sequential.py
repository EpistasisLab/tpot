import tpot2
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SklearnIndividualGenerator

class SequentialPipelineIndividual(SklearnIndividual):
    # takes in a list of search spaces. each space is a list of SklearnIndividualGenerators.
    # will produce a pipeline of Sequential length. Each step in the pipeline will correspond to the the search space provided in the same index.

    def __init__(self, search_spaces : List[SklearnIndividualGenerator] ) -> None:
        super().__init__()
        self.search_spaces = search_spaces
        self.pipeline = self._generate_pipeline()

    def _generate_pipeline(self, rng=None):
        pipeline = []
        for space in self.search_spaces:
            pipeline.append(space.generate(rng))
        return pipeline
    
    def mutate(self, rng=None):
        rng = np.random.default_rng()
        step = rng.choice(self.pipeline)
        return step.mutate(rng)
     

    def crossover(self, other, rng=None):
        #swap a random step in the pipeline with the corresponding step in the other pipeline

        if len(self.pipeline) != len(other.pipeline):
            return False
        
        if len(self.pipeline) < 2:
            return False
        
        rng = np.random.default_rng()
        idx = rng.integers(1,len(self.pipeline))

        self.pipeline[idx], other.pipeline[idx] = other.pipeline[idx], self.pipeline[idx]
        return True
    
    def export_pipeline(self):
        return sklearn.pipeline.make_pipeline(*[step.export_pipeline() for step in self.pipeline])
    
    def unique_id(self):
        return self


class SequentialPipeline(SklearnIndividualGenerator):
    def __init__(self, search_spaces : List[SklearnIndividualGenerator] ) -> None:
        """
        Takes in a list of search spaces. will produce a pipeline of Sequential length. Each step in the pipeline will correspond to the the search space provided in the same index.
        """
        
        self.search_spaces = search_spaces

    def generate(self, rng=None):
        return SequentialPipelineIndividual(self.search_spaces)