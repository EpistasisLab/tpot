# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

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

from __future__ import print_function
from functools import wraps
import warnings
from sklearn.datasets import make_classification, make_regression
from .export_utils import expr_to_tree, generate_pipeline_code
from deap import creator

NUM_TESTS = 10

# generate a small data set for a new pipeline, in order to check if the pipeline
# has unsuppported combinations in params
pretest_X, pretest_y = make_classification(n_samples=50, n_features=10, random_state=42)
pretest_X_reg, pretest_y_reg = make_regression(n_samples=50, n_features=10, random_state=42)


def _pre_test(func):
    """Check if the wrapped function works with a pretest data set.

    Reruns the wrapped function until it generates a good pipeline, for a max of
    NUM_TESTS times.

    Parameters
    ----------
    func: function
        The decorated function.

    Returns
    -------
    check_pipeline: function
        A wrapper function around the func parameter
    """
    @wraps(func)
    def check_pipeline(self, *args, **kwargs):
        bad_pipeline = True
        num_test = 0  # number of tests

        # a pool for workable pipeline
        while bad_pipeline and num_test < NUM_TESTS:
            # clone individual before each func call so it is not altered for
            # the possible next cycle loop
            args = [self._toolbox.clone(arg) if isinstance(arg, creator.Individual) else arg for arg in args]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')

                    expr = func(self, *args, **kwargs)
                    # mutation operator returns tuple (ind,); crossover operator
                    # returns tuple of (ind1, ind2)
                    expr_tuple = expr if isinstance(expr, tuple) else (expr,)

                    for expr_test in expr_tuple:
                        pipeline_code = generate_pipeline_code(
                            expr_to_tree(expr_test, self._pset),
                            self.operators
                        )
                        sklearn_pipeline = eval(pipeline_code, self.operators_context)

                        if self.classification:
                            sklearn_pipeline.fit(pretest_X, pretest_y)
                        else:
                            sklearn_pipeline.fit(pretest_X_reg, pretest_y_reg)
                        bad_pipeline = False
            except BaseException as e:
                message = '_pre_test decorator: {fname}: num_test={n} {e}'.format(
                    n=num_test,
                    fname=func.__name__,
                    e=e
                )
                # Use the pbar output stream if it's active
                self._update_pbar(pbar_num=0, pbar_msg=message)
            finally:
                num_test += 1

        return expr

    return check_pipeline
