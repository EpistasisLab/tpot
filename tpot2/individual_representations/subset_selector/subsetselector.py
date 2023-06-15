from numpy import iterable
import tpot2
import random
from .. import BaseIndividual

class SubsetSelector(BaseIndividual):
    def __init__(   self,
                    values,
                    initial_set = None,
                    k=1, #step size for shuffling
                ):

        if isinstance(values, int):
            self.values = set(range(0,values))
        else:
            self.values = set(values)


        if initial_set is None:
            self.subsets = set(random.choices(values, k=k))
        else:
            self.subsets = set(initial_set)

        self.k = k

        self.mutation_list = [self._mutate_add, self._mutate_remove]
        self.crossover_list = [self._crossover_swap]
        
    def _mutate_add(self,):
        not_included = list(self.values.difference(self.subsets))
        if len(not_included) > 1:
            self.subsets.update(random.sample(not_included, k=min(self.k, len(not_included))))
            return True
        else:
            return False

    def _mutate_remove(self,):
        if len(self.subsets) > 1:
            self.subsets = self.subsets - set(random.sample(list(self.subsets), k=min(self.k, len(self.subsets)-1) ))

    def _crossover_swap(self, ss2):
        diffs = self.subsets.symmetric_difference(ss2.subsets)

        if len(diffs) == 0:
            return False
        for v in diffs:
            self.subsets.discard(v)
            ss2.subsets.discard(v)
            random.choice([self.subsets, ss2.subsets]).add(v)
        
        return True
