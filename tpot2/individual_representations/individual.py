from abc import abstractmethod
import types
import numpy as np
import random
import copy
import copy
import typing


class BaseIndividual:


    def __init__(self) -> None:


        self.mutation_list = [] 
        self.crossover_list = []

    
    def mutate(self,):
        mutation_list_copy = self.mutation_list.copy()
        random.shuffle(mutation_list_copy)
        for func in mutation_list_copy:
            if func():
                return True
        return False

    def crossover(self, ind2):
        crossover_list_copy = self.crossover_list.copy()
        random.shuffle(crossover_list_copy)
        for func in crossover_list_copy:
            if func(ind2):
                return True
        return False

    # a guided change of an individual when given an objective function
    def optimize(self, objective_function, steps=5):
        for _ in range(steps):
            self.mutate()

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
