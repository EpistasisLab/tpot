from numpy import iterable
import tpot2
import numpy as np
from .. import BaseIndividual

class SubsetSelector(BaseIndividual):
    def __init__(   self,
                    values,
                    rng_=None,
                    initial_set = None,
                    k=1, #step size for shuffling
                ):

        rng = np.random.default_rng(rng_)

        if isinstance(values, int):
            self.values = set(range(0,values))
        else:
            self.values = set(values)


        if initial_set is None:
            self.subsets = set(rng.choices(values, k=k))
        else:
            self.subsets = set(initial_set)

        self.k = k

        self.mutation_list = [self._mutate_add, self._mutate_remove]
        self.crossover_list = [self._crossover_swap]

    def _mutate_add(self, rng_=None):
        rng = np.random.default_rng(rng_)
        not_included = list(self.values.difference(self.subsets))
        if len(not_included) > 1:
            self.subsets.update(rng.choice(not_included, k=min(self.k, len(not_included))))
            return True
        else:
            return False

    def _mutate_remove(self, rng_=None):
        rng = np.random.default_rng(rng_)
        if len(self.subsets) > 1:
            self.subsets = self.subsets - set(rng.choice(list(self.subsets), k=min(self.k, len(self.subsets)-1) ))

    def _crossover_swap(self, ss2, rng_=None):
        rng = np.random.default_rng(rng_)
        diffs = self.subsets.symmetric_difference(ss2.subsets)

        if len(diffs) == 0:
            return False
        for v in diffs:
            self.subsets.discard(v)
            ss2.subsets.discard(v)
            rng.choice([self.subsets, ss2.subsets]).add(v)

        return True
