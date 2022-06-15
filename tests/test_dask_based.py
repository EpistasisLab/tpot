"""Tests that ensure the dask-based fit matches.

https://github.com/DEAP/deap/issues/75
"""
import unittest
import nose
from sklearn.datasets import make_classification

from tpot import TPOTClassifier

try:
    import dask  # noqa
    import dask_ml  # noqa
except ImportError:
    raise nose.SkipTest()


class TestDaskMatches(unittest.TestCase):

    def test_dask_matches(self):
        with dask.config.set(scheduler='single-threaded'):
            for n_jobs in [-1]:
                X, y = make_classification(random_state=42)
                a = TPOTClassifier(
                    generations=0,
                    population_size=5,
                    cv=3,
                    random_state=42,
                    n_jobs=n_jobs,
                    use_dask=False,
                    verbosity=3
                )
                b = TPOTClassifier(
                    generations=0,
                    population_size=5,
                    cv=3,
                    random_state=42,
                    n_jobs=n_jobs,
                    use_dask=True,
                    verbosity=3
                )
                a.fit(X, y)
                b.fit(X, y)

                self.assertEqual(a.score(X, y), b.score(X, y))
                self.assertEqual(a.pareto_front_fitted_pipelines_.keys(),
                                 b.pareto_front_fitted_pipelines_.keys())
                self.assertEqual(a.evaluated_individuals_,
                                 b.evaluated_individuals_)
