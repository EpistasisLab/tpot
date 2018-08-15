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
        for n_jobs in [1, -1]:
            X, y = make_classification(random_state=0)
            a = TPOTClassifier(
                generations=2,
                population_size=5,
                cv=3,
                random_state=0,
                n_jobs=-1,
                use_dask=False,
            )
            b = TPOTClassifier(
                generations=2,
                population_size=5,
                cv=3,
                random_state=0,
                n_jobs=-1,
                use_dask=True,
            )
            b.fit(X, y)
            a.fit(X, y)
            self.assertEqual(a.evaluated_individuals_,
                             b.evaluated_individuals_)
            self.assertEqual(a.pareto_front_fitted_pipelines_.keys(),
                             b.pareto_front_fitted_pipelines_.keys())
            self.assertEqual(a.score(X, y), b.score(X, y))
