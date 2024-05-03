import tpot2
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from ..base import SklearnIndividual, SklearnIndividualGenerator
import networkx as nx
import copy
import matplotlib.pyplot as plt

from .graph import GraphPipelineIndividual, GraphPipeline


from ..graph_utils import *

class TreePipelineIndividual(GraphPipelineIndividual):
    def __init__(self,  
                        **kwargs) -> None:
        super().__init__(**kwargs)

        self.crossover_methods_list = [self._crossover_swap_branch, self._crossover_swap_node, self._crossover_nodes]
        self.mutate_methods_list = [self._mutate_insert_leaf, self._mutate_insert_inner_node, self._mutate_remove_node, self._mutate_node]
        self.merge_duplicated_nodes_toggle = False
 
    

class TreePipeline(SklearnIndividualGenerator):
    def __init__(self, root_search_space : SklearnIndividualGenerator, 
                        leaf_search_space : SklearnIndividualGenerator = None, 
                        inner_search_space : SklearnIndividualGenerator =None, 
                        min_size: int = 2, 
                        max_size: int = 10,
                        crossover_same_depth=False) -> None:
        
        """
        Generates a pipeline of variable length. Pipeline will have a tree structure similar to TPOT1.

        """
        
        self.search_space = root_search_space
        self.leaf_search_space = leaf_search_space
        self.inner_search_space = inner_search_space
        self.min_size = min_size
        self.max_size = max_size
        self.crossover_same_depth = crossover_same_depth

    def generate(self, rng=None):
        return TreePipelineIndividual(self.search_space, self.leaf_search_space, self.inner_search_space, self.min_size, self.max_size, self.crossover_same_depth, rng=rng) 