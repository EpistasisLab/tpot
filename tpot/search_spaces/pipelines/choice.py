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