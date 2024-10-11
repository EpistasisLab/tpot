import tpot2
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SearchSpace

class ChoicePipelineIndividual(SklearnIndividual):
    def __init__(self, search_spaces : List[SearchSpace], rng=None) -> None:
        super().__init__()
        rng = np.random.default_rng(rng)
        self.search_spaces = search_spaces
        self.node = rng.choice(self.search_spaces).generate(rng=rng)
        

    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        if rng.choice([True, False]):
            return self._mutate_select_new_node(rng)
        else:
            return self._mutate_node(rng)
    
    def _mutate_select_new_node(self, rng=None):
        rng = np.random.default_rng(rng)
        self.node = rng.choice(self.search_spaces).generate(rng=rng)
        return True
    
    def _mutate_node(self, rng=None):
        return self.node.mutate(rng)

    def crossover(self, other, rng=None):
        return self.node.crossover(other.node, rng)
    
    def export_pipeline(self, **kwargs):
        return self.node.export_pipeline(**kwargs)
    
    def unique_id(self):
        return self.node.unique_id()
    

class ChoicePipeline(SearchSpace):
    def __init__(self, search_spaces : List[SearchSpace] ) -> None:
        self.search_spaces = search_spaces

    """
    Takes in a list of search spaces. Will select one node from the search space.

    """

    def generate(self, rng=None):
        rng = np.random.default_rng(rng)
        return ChoicePipelineIndividual(self.search_spaces, rng=rng)