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
import networkx as nx
import copy
import matplotlib.pyplot as plt

from .graph import GraphPipelineIndividual


from ..graph_utils import *

class TreePipelineIndividual(GraphPipelineIndividual):
    def __init__(self,  
                        **kwargs) -> None:
        super().__init__(**kwargs)

        self.crossover_methods_list = [self._crossover_swap_branch, self._crossover_swap_node, self._crossover_nodes]
        self.mutate_methods_list = [self._mutate_insert_leaf, self._mutate_insert_inner_node, self._mutate_remove_node, self._mutate_node]
        self.merge_duplicated_nodes_toggle = False
 
    

class TreePipeline(SearchSpace):
    def __init__(self, root_search_space : SearchSpace, 
                        leaf_search_space : SearchSpace = None, 
                        inner_search_space : SearchSpace =None, 
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
        rng = np.random.default_rng(rng)
        return TreePipelineIndividual(self.search_space, self.leaf_search_space, self.inner_search_space, self.min_size, self.max_size, self.crossover_same_depth, rng=rng) 