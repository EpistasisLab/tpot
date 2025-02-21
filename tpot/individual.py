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
from abc import abstractmethod
import types
import numpy as np
import copy
import copy
import typing


class BaseIndividual:


    def __init__(self) -> None:
        self.mutation_list = []
        self.crossover_list = []

    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        mutation_list_copy = self.mutation_list.copy()
        rng.shuffle(mutation_list_copy)
        for func in mutation_list_copy:
            if func():
                return True
        return False

    def crossover(self, ind2, rng=None):
        rng = np.random.default_rng(rng)
        crossover_list_copy = self.crossover_list.copy()
        rng.shuffle(crossover_list_copy)
        for func in crossover_list_copy:
            if func(ind2):
                return True
        return False

    # a guided change of an individual when given an objective function
    def optimize(self, objective_function, rng=None , steps=5):
        rng = np.random.default_rng(rng)
        for _ in range(steps):
            self.mutate(rng=rng)

    #Return a hashable unique to this individual setup
    #For use when evaluating whether or not an individual is 'the same' and another individual
    def unique_id(self):
        return self


    #TODO https://www.pythontutorial.net/python-oop/python-__hash__/
    #python hashing and __eq__ functions look into
    #whether or not this would be a better way of doing things

    # #TODO: use this instead of unique_id()?
    # #unique_id() and __repr__ could have different levels of specificity.
    # def __repr__(self) -> str:
    #     pass

    # def __hash__(self) -> int:
    #     pass

    # def __eq__(self, other):
    #     self.unique_id() == other.unique_id()
