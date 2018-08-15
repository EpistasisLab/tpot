"""Tests that ensure the dask-based fit matches.

https://github.com/DEAP/deap/issues/75
"""
import unittest

import nose
from sklearn.datasets import make_classification

from tpot import TPOTClassifier

try:
    import dask
    import dask_ml.utils
except ImportError:
    raise nose.SkipTest()



class TestDaskMatches(unittest.TestCase):
    def test_dask_matches(self):
        a = TPOTClassifier(
            generations=2,
            population_size=5,
            cv=3,
            random_state=0,
            use_dask=False,
        )
        b = TPOTClassifier(
            generations=2,
            population_size=5,
            cv=3,
            random_state=0,
            use_dask=True,
        )
        X, y = make_classification(random_state=0)
        a.fit(X, y)
        b.fit(X, y)
