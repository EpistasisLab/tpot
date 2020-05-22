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

from tpot import TPOTClassifier, TPOTRegressor
from tpot.base import TPOTBase
from tpot.driver import float_range
from tpot.gp_types import Output_Array
from tpot.gp_deap import mutNodeReplacement, _wrapped_cross_val_score, pick_two_individuals_eligible_for_crossover, cxOnePoint, varOr, initialize_stats_dict
from tpot.metrics import balanced_accuracy, SCORERS
from tpot.operator_utils import TPOTOperatorClassFactory, set_sample_weight, source_decode

from tpot.config.classifier import classifier_config_dict
from tpot.config.classifier_light import classifier_config_dict_light
from tpot.config.regressor_light import regressor_config_dict_light
from tpot.config.classifier_mdr import tpot_mdr_classifier_config_dict
from tpot.config.regressor_mdr import tpot_mdr_regressor_config_dict
from tpot.config.regressor_sparse import regressor_config_sparse
from tpot.config.classifier_sparse import classifier_config_sparse
from tpot.config.classifier_nn import classifier_config_nn

import numpy as np
import pandas as pd
from scipy import sparse
import inspect
import random
import warnings
from multiprocessing import cpu_count
import os
import sys
from re import search
from datetime import datetime
from time import sleep
from tempfile import mkdtemp
from shutil import rmtree
import platform

from sklearn.datasets import load_digits, load_boston, make_classification, make_regression
from sklearn import model_selection
from joblib import Memory
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
try:
    from sklearn.feature_selection._base import SelectorMixin
except ImportError:
    from sklearn.feature_selection.base import SelectorMixin
from deap import creator, gp
from deap.tools import ParetoFront
from nose.tools import nottest, assert_raises, assert_not_equal, assert_greater_equal, assert_equal, assert_in
from driver_tests import captured_output

train_test_split = nottest(model_selection.train_test_split)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

# Ensure we can use `with closing(...) as ... :` syntax
if getattr(StringIO, '__exit__', False) and \
   getattr(StringIO, '__enter__', False):
    def closing(arg):
        return arg
else:
    from contextlib import closing

# Set up the digits data set for testing
digits_data = load_digits()
training_features, testing_features, training_target, testing_target = \
    train_test_split(digits_data.data.astype(np.float64), digits_data.target.astype(np.float64), random_state=42)

# Set up test data with missing value
features_with_nan = np.copy(training_features)
features_with_nan[0][0] = float('nan')

# Set up the Boston data set for testing
boston_data = load_boston()
training_features_r, testing_features_r, training_target_r, testing_target_r = \
    train_test_split(boston_data.data, boston_data.target, random_state=42)

# Set up a small test dataset

pretest_X, pretest_y = make_classification(
                                            n_samples=100,
                                            n_features=10,
                                            random_state=42)

pretest_X_reg, pretest_y_reg = make_regression(
                                            n_samples=100,
                                            n_features=10,
                                            random_state=42
                                            )

# Set up artificial groups
n_datapoints = pretest_y.shape[0]
groups = np.random.randint(low=0, high=n_datapoints//2, size=n_datapoints)

# Set up custom cv split generators
custom_cvs = [
    model_selection.LeaveOneGroupOut().split(X=pretest_X, y=pretest_y, groups=groups),
    model_selection.LeavePGroupsOut(2).split(X=pretest_X, y=pretest_y, groups=groups),
    model_selection.RepeatedStratifiedKFold(),
    5,
    10,
    ]

# Set up pandas DataFrame for testing
input_data = pd.read_csv(
    'tests/tests.csv',
    sep=',',
    dtype=np.float64,
)
pd_features = input_data.drop('class', axis=1)
pd_target = input_data['class']

# Set up the sparse matrix for testing
sparse_features = sparse.csr_matrix(training_features)
sparse_target = training_target

np.random.seed(42)
random.seed(42)

test_operator_key = 'sklearn.feature_selection.SelectPercentile'
TPOTSelectPercentile, TPOTSelectPercentile_args = TPOTOperatorClassFactory(
    test_operator_key,
    classifier_config_dict[test_operator_key]
)

tpot_obj = TPOTClassifier()
tpot_obj._fit_init()

def test_init_custom_parameters():
    """Assert that the TPOT instantiator stores the TPOT variables properly."""
    tpot_obj = TPOTClassifier(
        population_size=500,
        generations=1000,
        offspring_size=2000,
        mutation_rate=0.05,
        crossover_rate=0.9,
        scoring='accuracy',
        cv=10,
        verbosity=1,
        random_state=42,
        disable_update_check=True,
        warm_start=True,
        log_file=None
    )

    assert tpot_obj.population_size == 500
    assert tpot_obj.generations == 1000
    assert tpot_obj.offspring_size == 2000
    assert tpot_obj.mutation_rate == 0.05
    assert tpot_obj.crossover_rate == 0.9
    assert tpot_obj.scoring_function == 'accuracy'
    assert tpot_obj.cv == 10
    assert tpot_obj.max_time_mins is None
    assert tpot_obj.warm_start is True
    assert tpot_obj.verbosity == 1
    assert tpot_obj.log_file == None

    tpot_obj._fit_init()

    assert tpot_obj._pop == []
    assert tpot_obj._pareto_front == None
    assert tpot_obj._last_optimized_pareto_front == None
    assert tpot_obj._last_optimized_pareto_front_n_gens == 0
    assert tpot_obj._optimized_pipeline == None
    assert tpot_obj._optimized_pipeline_score == None
    assert tpot_obj.fitted_pipeline_ == None
    assert tpot_obj._exported_pipeline_text == []
    assert tpot_obj.log_file == sys.stdout

def test_init_custom_progress_file():
    """ Assert that TPOT has right file handler to save progress. """
    file_name = "progress.txt"
    file_handle = open(file_name, "w")
    tpot_obj = TPOTClassifier(log_file=file_handle)
    assert tpot_obj.log_file == file_handle

def test_init_default_scoring():
    """Assert that TPOT intitializes with the correct default scoring function."""
    tpot_obj = TPOTRegressor()
    assert tpot_obj.scoring_function == 'neg_mean_squared_error'

    tpot_obj = TPOTClassifier()
    assert tpot_obj.scoring_function == 'accuracy'


def test_init_default_scoring_2():
    """Assert that TPOT rasies ValueError with a invalid sklearn metric function."""
    tpot_obj = TPOTClassifier(scoring=balanced_accuracy)
    assert_raises(ValueError, tpot_obj._fit_init)


def test_init_default_scoring_3():
    """Assert that TPOT intitializes with a valid _BaseScorer."""
    with warnings.catch_warnings(record=True) as w:
        tpot_obj = TPOTClassifier(scoring=make_scorer(balanced_accuracy))
        tpot_obj._fit_init()
    assert len(w) == 0 # deap 1.2.2 warning message made this unit test failed
    assert tpot_obj.scoring_function._score_func == balanced_accuracy


def test_init_default_scoring_4():
    """Assert that TPOT intitializes with a valid scorer."""
    def my_scorer(clf, X, y):
        return 0.9

    with warnings.catch_warnings(record=True) as w:
        tpot_obj = TPOTClassifier(scoring=my_scorer)
        tpot_obj._fit_init()
    assert len(w) == 0 # deap 1.2.2 warning message made this unit test failed
    assert tpot_obj.scoring_function == my_scorer


def test_init_default_scoring_5():
    """Assert that TPOT rasies ValueError with a invalid sklearn metric function roc_auc_score."""
    tpot_obj = TPOTClassifier(scoring=roc_auc_score)
    assert_raises(ValueError, tpot_obj._fit_init)


def test_init_default_scoring_6():
    """Assert that TPOT rasies ValueError with a invalid sklearn metric function from __main__."""
    def my_scorer(y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    tpot_obj = TPOTClassifier(scoring=my_scorer)
    assert_raises(ValueError, tpot_obj._fit_init)


def test_init_default_scoring_7():
    """Assert that TPOT rasies ValueError with a valid sklearn metric function from __main__."""
    def my_scorer(estimator, X, y):
        return make_scorer(balanced_accuracy)

    tpot_obj = TPOTClassifier(scoring=my_scorer)
    tpot_obj._fit_init()


def test_invalid_score_warning():
    """Assert that the TPOT intitializes raises a ValueError when the scoring metrics is not available in SCORERS."""
    # Mis-spelled scorer
    tpot_obj = TPOTClassifier(scoring='balanced_accuray')
    assert_raises(ValueError, tpot_obj._fit_init)
    # Correctly spelled
    tpot_obj = TPOTClassifier(scoring='balanced_accuracy')


def test_invalid_dataset_warning():
    """Assert that the TPOT fit function raises a ValueError when dataset is not in right format."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0
    )
    tpot_obj._fit_init()
    # common mistake in target
    bad_training_target = training_target.reshape((1, len(training_target)))
    assert_raises(ValueError, tpot_obj.fit, training_features, bad_training_target)


def test_invalid_subsample_ratio_warning():
    """Assert that the TPOT intitializes raises a ValueError when subsample ratio is not in the range (0.0, 1.0]."""
    # Invalid ratio
    tpot_obj = TPOTClassifier(subsample=0.0)
    assert_raises(ValueError, tpot_obj._fit_init)
    # Valid ratio
    TPOTClassifier(subsample=0.1)


def test_invalid_mut_rate_plus_xo_rate():
    """Assert that the TPOT intitializes raises a ValueError when the sum of crossover and mutation probabilities is large than 1."""
    # Invalid ratio
    tpot_obj = TPOTClassifier(mutation_rate=0.8, crossover_rate=0.8)
    assert_raises(ValueError, tpot_obj._fit_init)
    # Valid ratio
    TPOTClassifier(mutation_rate=0.8, crossover_rate=0.1)


def test_init_max_time_mins():
    """Assert that the TPOT init stores max run time and sets generations to 1000000."""
    tpot_obj = TPOTClassifier(max_time_mins=30, generations=None)
    tpot_obj._fit_init()
    assert tpot_obj.generations == 1000000
    assert tpot_obj.max_time_mins == 30


def test_init_max_time_mins_and_generations():
    """Assert that the TPOT init stores max run time but keeps the generations at the user-supplied value."""
    tpot_obj = TPOTClassifier(max_time_mins=30, generations=1000)
    tpot_obj._fit_init()
    assert tpot_obj.generations == 1000
    assert tpot_obj.max_time_mins == 30


def test_init_n_jobs():
    """Assert that the TPOT init stores current number of processes."""
    tpot_obj = TPOTClassifier(n_jobs=2)
    assert tpot_obj.n_jobs == 2
    tpot_obj._fit_init()
    assert tpot_obj._n_jobs == 2

    tpot_obj = TPOTClassifier(n_jobs=-1)
    assert tpot_obj.n_jobs == -1
    tpot_obj._fit_init()
    assert tpot_obj._n_jobs == cpu_count()


def test_init_n_jobs_2():
    """Assert that the TPOT init assign right"""
    tpot_obj = TPOTClassifier(n_jobs=-2)
    assert tpot_obj.n_jobs == -2

    tpot_obj._fit_init()
    assert tpot_obj._n_jobs == cpu_count() - 1


def test_init_n_jobs_3():
    """Assert that the TPOT init rasies ValueError if n_jobs=0."""
    tpot_obj = TPOTClassifier(n_jobs=0)
    assert tpot_obj.n_jobs == 0

    assert_raises(ValueError, tpot_obj._fit_init)


def test_timeout():
    """Assert that _wrapped_cross_val_score return Timeout in a time limit."""
    tpot_obj = TPOTRegressor(scoring='neg_mean_squared_error')
    tpot_obj._fit_init()
    # a complex pipeline for the test
    pipeline_string = (
        "ExtraTreesRegressor("
        "GradientBoostingRegressor(input_matrix, GradientBoostingRegressor__alpha=0.8,"
        "GradientBoostingRegressor__learning_rate=0.1,GradientBoostingRegressor__loss=huber,"
        "GradientBoostingRegressor__max_depth=5, GradientBoostingRegressor__max_features=0.5,"
        "GradientBoostingRegressor__min_samples_leaf=5, GradientBoostingRegressor__min_samples_split=5,"
        "GradientBoostingRegressor__n_estimators=100, GradientBoostingRegressor__subsample=0.25),"
        "ExtraTreesRegressor__bootstrap=True, ExtraTreesRegressor__max_features=0.5,"
        "ExtraTreesRegressor__min_samples_leaf=5, ExtraTreesRegressor__min_samples_split=5, "
        "ExtraTreesRegressor__n_estimators=100)"
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    # test _wrapped_cross_val_score with cv=20 so that it is impossible to finish in 1 second
    cv = model_selection.KFold(n_splits=20).split(
        X=training_features_r,
        y=training_target_r,
    )
    cv = model_selection._split.check_cv(cv, training_target_r, classifier=False)
    return_value = _wrapped_cross_val_score(tpot_obj.fitted_pipeline_,
                                            training_features_r,
                                            training_target_r,
                                            cv=cv,
                                            scoring_function='neg_mean_squared_error',
                                            sample_weight=None,
                                            groups=None,
                                            timeout=1)
    assert return_value == "Timeout"

def test_custom_cv_test_generator():
    """Check that custom cv generators processed correctly.
    """
    def check_custom_cv(_cv):
        tpot_obj = TPOTClassifier(
            random_state=42,
            population_size=1,
            offspring_size=2,
            generations=1,
            verbosity=0,
            cv=_cv,
            n_jobs=-1  # ensure pickling / parallelization
        )
        tpot_obj.fit(pretest_X, pretest_y)

    for cv in custom_cvs:
        yield check_custom_cv, cv

def test_invalid_pipeline():
    """Assert that _wrapped_cross_val_score return -float(\'inf\') with a invalid_pipeline"""

    # a invalid pipeline
    # Dual or primal formulation. Dual formulation is only implemented for l2 penalty.
    pipeline_string = (
        'LogisticRegression(input_matrix,  LogisticRegression__C=10.0, '
        'LogisticRegression__dual=True, LogisticRegression__penalty=l1)'
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)

    cv = model_selection.KFold().split(
        X=training_features,
        y=training_target,
    )
    cv = model_selection._split.check_cv(
        cv,
        training_target,
        classifier=False  # choice of classifier vs. regressor here is arbitrary
    )
    return_value = _wrapped_cross_val_score(tpot_obj.fitted_pipeline_,
                                            training_features,
                                            training_target,
                                            cv=cv,
                                            scoring_function='accuracy',
                                            sample_weight=None,
                                            groups=None,
                                            timeout=300)
    assert return_value == -float('inf')


def test_balanced_accuracy():
    """Assert that the balanced_accuracy in TPOT returns correct accuracy."""
    y_true = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4])
    y_pred1 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4])
    y_pred2 = np.array([3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4])
    accuracy_score1 = balanced_accuracy(y_true, y_pred1)
    accuracy_score2 = balanced_accuracy(y_true, y_pred2)
    assert np.allclose(accuracy_score1, 1.0)
    assert np.allclose(accuracy_score2, 0.833333333333333)


def test_get_params():
    """Assert that get_params returns the exact dictionary of parameters used by TPOT."""
    kwargs = {
        'population_size': 500,
        'generations': 1000,
        'config_dict': 'TPOT light',
        'offspring_size': 2000,
        'verbosity': 1
    }

    tpot_obj = TPOTClassifier(**kwargs)
    # Get default parameters of TPOT and merge with our specified parameters
    initializer = inspect.getargspec(TPOTBase.__init__)
    default_kwargs = dict(zip(initializer.args[1:], initializer.defaults))
    default_kwargs.update(kwargs)

    assert tpot_obj.get_params()['config_dict'] == 'TPOT light'
    assert tpot_obj.get_params() == default_kwargs


def test_set_params():
    """Assert that set_params returns a reference to the TPOT instance."""

    assert tpot_obj.set_params() is tpot_obj


def test_set_params_2():
    """Assert that set_params updates TPOT's instance variables."""
    tpot_obj = TPOTClassifier(generations=2)
    tpot_obj.set_params(generations=3)

    assert tpot_obj.generations == 3


def test_TPOTBase():
    """Assert that TPOTBase class raises RuntimeError when using it directly."""
    assert_raises(RuntimeError, TPOTBase)


def test_conf_dict():
    """Assert that TPOT uses the pre-configured dictionary of operators when config_dict is 'TPOT light' or 'TPOT MDR'."""
    tpot_obj = TPOTClassifier(config_dict='TPOT light')
    tpot_obj._fit_init()
    assert tpot_obj._config_dict == classifier_config_dict_light

    tpot_obj = TPOTClassifier(config_dict='TPOT MDR')
    tpot_obj._fit_init()
    assert tpot_obj._config_dict == tpot_mdr_classifier_config_dict

    tpot_obj = TPOTClassifier(config_dict='TPOT sparse')
    tpot_obj._fit_init()
    assert tpot_obj._config_dict == classifier_config_sparse

    tpot_obj = TPOTRegressor(config_dict='TPOT light')
    tpot_obj._fit_init()
    assert tpot_obj._config_dict == regressor_config_dict_light

    tpot_obj = TPOTRegressor(config_dict='TPOT MDR')
    tpot_obj._fit_init()
    assert tpot_obj._config_dict == tpot_mdr_regressor_config_dict

    tpot_obj = TPOTRegressor(config_dict='TPOT sparse')
    tpot_obj._fit_init()
    assert tpot_obj._config_dict == regressor_config_sparse


def test_conf_dict_2():
    """Assert that TPOT uses a custom dictionary of operators when config_dict is Python dictionary."""
    tpot_obj = TPOTClassifier(config_dict=tpot_mdr_classifier_config_dict)
    assert tpot_obj.config_dict == tpot_mdr_classifier_config_dict


def test_conf_dict_3():
    """Assert that TPOT uses a custom dictionary of operators when config_dict is the path of Python dictionary."""
    tpot_obj = TPOTRegressor(config_dict='tests/test_config.py')
    tpot_obj._fit_init()
    tested_config_dict = {
        'sklearn.naive_bayes.GaussianNB': {
        },

        'sklearn.naive_bayes.BernoulliNB': {
            'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
            'fit_prior': [True, False]
        },

        'sklearn.naive_bayes.MultinomialNB': {
            'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
            'fit_prior': [True, False]
        }
    }

    assert isinstance(tpot_obj.config_dict, str)
    assert isinstance(tpot_obj._config_dict, dict)
    assert tpot_obj._config_dict == tested_config_dict


def test_read_config_file():
    """Assert that _read_config_file rasies FileNotFoundError with a wrong path."""
    tpot_obj = TPOTRegressor()
    # typo for "tests/test_config.py"
    assert_raises(ValueError, tpot_obj._read_config_file, "tests/test_confg.py")


def test_read_config_file_2():
    """Assert that _read_config_file rasies ValueError with wrong dictionary format"""
    tpot_obj = TPOTRegressor()
    assert_raises(ValueError, tpot_obj._read_config_file, "tests/test_config.py.bad")


def test_read_config_file_3():
    """Assert that _read_config_file rasies ValueError without a dictionary named 'tpot_config'."""
    tpot_obj = TPOTRegressor()
    assert_raises(ValueError, tpot_obj._setup_config, "tpot/config/regressor_sparse.py")


def test_random_ind():
    """Assert that the TPOTClassifier can generate the same pipeline with same random seed."""
    tpot_obj = TPOTClassifier(random_state=43)
    tpot_obj._fit_init()
    pipeline1 = str(tpot_obj._toolbox.individual())
    tpot_obj = TPOTClassifier(random_state=43)
    tpot_obj._fit_init()
    pipeline2 = str(tpot_obj._toolbox.individual())
    assert pipeline1 == pipeline2


def test_random_ind_2():
    """Assert that the TPOTRegressor can generate the same pipeline with same random seed."""
    tpot_obj = TPOTRegressor(random_state=43)
    tpot_obj._fit_init()
    pipeline1 = str(tpot_obj._toolbox.individual())
    tpot_obj = TPOTRegressor(random_state=43)
    tpot_obj._fit_init()
    pipeline2 = str(tpot_obj._toolbox.individual())

    assert pipeline1 == pipeline2


def test_score():
    """Assert that the TPOT score function raises a RuntimeError when no optimized pipeline exists."""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()
    assert_raises(RuntimeError, tpot_obj.score, testing_features, testing_target)


def test_score_2():
    """Assert that the TPOTClassifier score function outputs a known score for a fixed pipeline."""
    tpot_obj = TPOTClassifier(random_state=34)
    tpot_obj._fit_init()
    known_score = 0.977777777778  # Assumes use of the TPOT accuracy function

    # Create a pipeline with a known score
    pipeline_string = (
        'KNeighborsClassifier('
        'input_matrix, '
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')'
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj.fitted_pipeline_.fit(training_features, training_target)
    # Get score from TPOT
    score = tpot_obj.score(testing_features, testing_target)

    assert np.allclose(known_score, score)


def test_score_3():
    """Assert that the TPOTRegressor score function outputs a known score for a fixed pipeline."""
    tpot_obj = TPOTRegressor(scoring='neg_mean_squared_error', random_state=72)
    tpot_obj._fit_init()
    known_score = -11.708199875921563

    # Reify pipeline with known score
    pipeline_string = (
        "ExtraTreesRegressor("
        "GradientBoostingRegressor(input_matrix, GradientBoostingRegressor__alpha=0.8,"
        "GradientBoostingRegressor__learning_rate=0.1,GradientBoostingRegressor__loss=huber,"
        "GradientBoostingRegressor__max_depth=5, GradientBoostingRegressor__max_features=0.5,"
        "GradientBoostingRegressor__min_samples_leaf=5, GradientBoostingRegressor__min_samples_split=5,"
        "GradientBoostingRegressor__n_estimators=100, GradientBoostingRegressor__subsample=0.25),"
        "ExtraTreesRegressor__bootstrap=True, ExtraTreesRegressor__max_features=0.5,"
        "ExtraTreesRegressor__min_samples_leaf=5, ExtraTreesRegressor__min_samples_split=5, "
        "ExtraTreesRegressor__n_estimators=100)"
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj.fitted_pipeline_.fit(training_features_r, training_target_r)

    # Get score from TPOT
    score = tpot_obj.score(testing_features_r, testing_target_r)
    # On some non-amd64 systems such as arm64, a resulting score of
    # 0.8207525232725118 was observed, so we need to add a tolerance there
    if platform.machine() != 'amd64':
        assert np.allclose(known_score, score, rtol=0.03)
    else:
        assert np.allclose(known_score, score)


def test_sample_weight_func():
    """Assert that the TPOTRegressor score function outputs a known score for a fixed pipeline with sample weights."""
    tpot_obj = TPOTRegressor(scoring='neg_mean_squared_error')
    tpot_obj._fit_init()
    # Reify pipeline with known scor
    pipeline_string = (
        "ExtraTreesRegressor("
        "GradientBoostingRegressor(input_matrix, GradientBoostingRegressor__alpha=0.8,"
        "GradientBoostingRegressor__learning_rate=0.1,GradientBoostingRegressor__loss=huber,"
        "GradientBoostingRegressor__max_depth=5, GradientBoostingRegressor__max_features=0.5,"
        "GradientBoostingRegressor__min_samples_leaf=5, GradientBoostingRegressor__min_samples_split=5,"
        "GradientBoostingRegressor__n_estimators=100, GradientBoostingRegressor__subsample=0.25),"
        "ExtraTreesRegressor__bootstrap=True, ExtraTreesRegressor__max_features=0.5,"
        "ExtraTreesRegressor__min_samples_leaf=5, ExtraTreesRegressor__min_samples_split=5, "
        "ExtraTreesRegressor__n_estimators=100)"
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj.fitted_pipeline_.fit(training_features_r, training_target_r)

    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)

    # make up a sample weight
    training_target_r_weight = np.array(range(1, len(training_target_r)+1))
    training_target_r_weight_dict = set_sample_weight(tpot_obj.fitted_pipeline_.steps, training_target_r_weight)

    np.random.seed(42)
    cv_score1 = model_selection.cross_val_score(tpot_obj.fitted_pipeline_, training_features_r, training_target_r, cv=3, scoring='neg_mean_squared_error')

    np.random.seed(42)
    cv_score2 = model_selection.cross_val_score(tpot_obj.fitted_pipeline_, training_features_r, training_target_r, cv=3, scoring='neg_mean_squared_error')

    np.random.seed(42)
    cv_score_weight = model_selection.cross_val_score(tpot_obj.fitted_pipeline_, training_features_r, training_target_r, cv=3, scoring='neg_mean_squared_error', fit_params=training_target_r_weight_dict)

    np.random.seed(42)
    tpot_obj.fitted_pipeline_.fit(training_features_r, training_target_r, **training_target_r_weight_dict)
    # Get score from TPOT
    known_score = -11.586816877933911
    score = tpot_obj.score(testing_features_r, testing_target_r)


    assert np.allclose(cv_score1, cv_score2)
    assert not np.allclose(cv_score1, cv_score_weight)
    # On some non-amd64 systems such as arm64, a resulting score of
    # 0.8207525232725118 was observed, so we need to add a tolerance there
    if platform.machine() != 'amd64':
        assert np.allclose(known_score, score, rtol=0.01)
    else:
        assert np.allclose(known_score, score)


def test_template_1():
    """Assert that TPOT template option generates pipeline when each step is a type of operator."""

    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0,
        template='Selector-Transformer-Classifier'
    )
    tpot_obj._fit_init()
    pop = tpot_obj._toolbox.population(n=10)
    for deap_pipeline in pop:
        operator_count = tpot_obj._operator_count(deap_pipeline)
        sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
        assert operator_count == 3
        assert issubclass(sklearn_pipeline.steps[0][1].__class__, SelectorMixin)
        assert issubclass(sklearn_pipeline.steps[1][1].__class__, TransformerMixin)
        assert issubclass(sklearn_pipeline.steps[2][1].__class__, ClassifierMixin)


def test_template_2():
    """Assert that TPOT template option generates pipeline when each step is operator type with a duplicate main type."""

    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0,
        template='Selector-Selector-Transformer-Classifier'
    )
    tpot_obj._fit_init()
    pop = tpot_obj._toolbox.population(n=10)
    for deap_pipeline in pop:
        operator_count = tpot_obj._operator_count(deap_pipeline)
        sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
        assert operator_count == 4
        assert issubclass(sklearn_pipeline.steps[0][1].__class__, SelectorMixin)
        assert issubclass(sklearn_pipeline.steps[1][1].__class__, SelectorMixin)
        assert issubclass(sklearn_pipeline.steps[2][1].__class__, TransformerMixin)
        assert issubclass(sklearn_pipeline.steps[3][1].__class__, ClassifierMixin)


def test_template_3():
    """Assert that TPOT template option generates pipeline when one of steps is a specific operator."""

    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0,
        template='SelectPercentile-Transformer-Classifier'
    )
    tpot_obj._fit_init()
    pop = tpot_obj._toolbox.population(n=10)
    for deap_pipeline in pop:
        operator_count = tpot_obj._operator_count(deap_pipeline)
        sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
        assert operator_count == 3
        assert sklearn_pipeline.steps[0][0] == 'SelectPercentile'.lower()
        assert issubclass(sklearn_pipeline.steps[0][1].__class__, SelectorMixin)
        assert issubclass(sklearn_pipeline.steps[1][1].__class__, TransformerMixin)
        assert issubclass(sklearn_pipeline.steps[2][1].__class__, ClassifierMixin)


def test_template_4():
    """Assert that TPOT template option generates pipeline when one of steps is a specific operator."""

    tpot_obj = TPOTClassifier(
        population_size=5,
        generations=2,
        random_state=42,
        verbosity=0,
        config_dict = 'TPOT light',
        template='SelectPercentile-Transformer-Classifier'
    )
    tpot_obj.fit(pretest_X, pretest_y)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)

    sklearn_pipeline = tpot_obj.fitted_pipeline_
    operator_count = tpot_obj._operator_count(tpot_obj._optimized_pipeline)
    assert operator_count == 3
    assert sklearn_pipeline.steps[0][0] == 'SelectPercentile'.lower()
    assert issubclass(sklearn_pipeline.steps[0][1].__class__, SelectorMixin)
    assert issubclass(sklearn_pipeline.steps[1][1].__class__, TransformerMixin)
    assert issubclass(sklearn_pipeline.steps[2][1].__class__, ClassifierMixin)


def test_template_5():
    """Assert that TPOT rasie ValueError when template parameter is invalid."""

    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0,
        template='SelectPercentile-Transformer-Classifie' # a typ in Classifier
    )
    assert_raises(ValueError, tpot_obj._fit_init)


def test_fit_GroupKFold():
    """Assert that TPOT properly handles the group parameter when using GroupKFold."""
    # This check tests if the darker digits images would generalize to the lighter ones.
    means = np.mean(training_features, axis=1)
    groups = means >= np.median(means)

    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=2,
        offspring_size=4,
        generations=1,
        verbosity=0,
        config_dict='TPOT light',
        cv=model_selection.GroupKFold(n_splits=2),
    )
    tpot_obj.fit(training_features, training_target, groups=groups)

    assert_greater_equal(tpot_obj.score(testing_features, testing_target), 0.97)


def test_predict():
    """Assert that the TPOT predict function raises a RuntimeError when no optimized pipeline exists."""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()
    assert_raises(RuntimeError, tpot_obj.predict, testing_features)


def test_predict_2():
    """Assert that the TPOT predict function returns a numpy matrix of shape (num_testing_rows,)."""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()
    pipeline_string = (
        'DecisionTreeClassifier('
        'input_matrix, '
        'DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, '
        'DecisionTreeClassifier__min_samples_split=5'
        ')'
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj.fitted_pipeline_.fit(training_features, training_target)
    result = tpot_obj.predict(testing_features)

    assert result.shape == (testing_features.shape[0],)


def test_predict_3():
    """Assert that the TPOT predict function works on dataset with nan"""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()

    pipeline_string = (
        'DecisionTreeClassifier('
        'input_matrix, '
        'DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, '
        'DecisionTreeClassifier__min_samples_split=5'
        ')'
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj.fitted_pipeline_.fit(training_features, training_target)
    result = tpot_obj.predict(features_with_nan)

    assert result.shape == (features_with_nan.shape[0],)


def test_predict_proba():
    """Assert that the TPOT predict_proba function returns a numpy matrix of shape (num_testing_rows, num_testing_target)."""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()

    pipeline_string = (
        'DecisionTreeClassifier('
        'input_matrix, '
        'DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, '
        'DecisionTreeClassifier__min_samples_split=5)'
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj.fitted_pipeline_.fit(training_features, training_target)

    result = tpot_obj.predict_proba(testing_features)
    num_labels = np.amax(testing_target) + 1

    assert result.shape == (testing_features.shape[0], num_labels)


def test_predict_proba_2():
    """Assert that the TPOT predict_proba function returns a numpy matrix filled with probabilities (float)."""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()
    pipeline_string = (
        'DecisionTreeClassifier('
        'input_matrix, '
        'DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, '
        'DecisionTreeClassifier__min_samples_split=5)'
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj.fitted_pipeline_.fit(training_features, training_target)

    result = tpot_obj.predict_proba(testing_features)
    rows, columns = result.shape

    for i in range(rows):
        for j in range(columns):
            float_range(result[i][j])


def test_predict_proba_3():
    """Assert that the TPOT predict_proba function raises a RuntimeError when no optimized pipeline exists."""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()

    assert_raises(RuntimeError, tpot_obj.predict_proba, testing_features)


def test_predict_proba_4():
    """Assert that the TPOT predict_proba function raises a RuntimeError when the optimized pipeline do not have the predict_proba() function"""
    tpot_obj = TPOTRegressor()
    tpot_obj._fit_init()
    pipeline_string = (
        "ExtraTreesRegressor(input_matrix, "
        "ExtraTreesRegressor__bootstrap=True, ExtraTreesRegressor__max_features=0.5,"
        "ExtraTreesRegressor__min_samples_leaf=5, ExtraTreesRegressor__min_samples_split=5, "
        "ExtraTreesRegressor__n_estimators=100)"
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj.fitted_pipeline_.fit(training_features_r, training_target_r)

    assert_raises(RuntimeError, tpot_obj.predict_proba, testing_features)


def test_predict_proba_5():
    """Assert that the TPOT predict_proba function works on dataset with nan."""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()
    pipeline_string = (
        'DecisionTreeClassifier('
        'input_matrix, '
        'DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, '
        'DecisionTreeClassifier__min_samples_split=5)'
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj.fitted_pipeline_.fit(training_features, training_target)

    result = tpot_obj.predict_proba(features_with_nan)
    num_labels = np.amax(training_target) + 1

    assert result.shape == (features_with_nan.shape[0], num_labels)


def test_warm_start():
    """Assert that the TPOT warm_start flag stores the pop and pareto_front from the first run."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light',
        warm_start=True)
    tpot_obj.fit(pretest_X, pretest_y)

    assert tpot_obj._pop is not None
    assert tpot_obj._pareto_front is not None

    first_pop = tpot_obj._pop
    tpot_obj.random_state = 21
    tpot_obj.fit(pretest_X, pretest_y)

    assert tpot_obj._pop == first_pop



def test_fit():
    """Assert that the TPOT fit function provides an optimized pipeline."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0
    )
    tpot_obj.fit(pretest_X, pretest_y)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)


def test_fit_2():
    """Assert that the TPOT fit function provides an optimized pipeline when config_dict is 'TPOT light'."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj.fit(training_features, training_target)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)


def test_fit_3():
    """Assert that the TPOT fit function provides an optimized pipeline with subsample of 0.8."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        subsample=0.8,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj.fit(training_features, training_target)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)


def test_fit_4():
    """Assert that the TPOT fit function provides an optimized pipeline with max_time_mins of 2 second."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=2,
        generations=None,
        verbosity=0,
        max_time_mins=2/60.,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()
    assert tpot_obj.generations == 1000000

    # reset generations to 20 just in case that the failed test may take too much time
    tpot_obj.generations = 20

    tpot_obj.fit(training_features, training_target)
    assert tpot_obj._pop == []
    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)


def test_fit_5():
    """Assert that the TPOT fit function provides an optimized pipeline with max_time_mins of 2 second with warm_start=True."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=2,
        generations=None,
        verbosity=0,
        max_time_mins=3/60.,
        config_dict='TPOT light',
        warm_start=True
    )
    tpot_obj._fit_init()
    assert tpot_obj.generations == 1000000

    # reset generations to 20 just in case that the failed test may take too much time
    tpot_obj.generations = 20

    tpot_obj.fit(training_features, training_target)
    assert tpot_obj._pop != []
    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)
    # rerun it
    tpot_obj.fit(training_features, training_target)
    assert tpot_obj._pop != []



def test_fit_6():
    """Assert that the TPOT fit function provides an optimized pipeline with pandas DataFrame"""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0
    )

    tpot_obj.fit(pd_features, pd_target)

    assert isinstance(pd_features, pd.DataFrame)
    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)


def test_fit_7():
    """Assert that the TPOT fit function provides an optimized pipeline."""
    tpot_obj = TPOTRegressor(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0
    )
    tpot_obj.fit(pretest_X_reg, pretest_y_reg)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)


def test_memory():
    """Assert that the TPOT fit function runs normally with memory=\'auto\'."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        config_dict='TPOT light',
        memory='auto',
        verbosity=0
    )
    tpot_obj.fit(training_features, training_target)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)
    assert tpot_obj.memory is not None
    assert tpot_obj._memory is None
    assert tpot_obj._cachedir is not None
    assert not os.path.isdir(tpot_obj._cachedir)


def test_memory_2():
    """Assert that the TPOT _setup_memory function runs normally with a valid path."""
    cachedir = mkdtemp()
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        config_dict='TPOT light',
        memory=cachedir,
        verbosity=0
    )
    tpot_obj._setup_memory()
    rmtree(cachedir)

    assert tpot_obj._cachedir == cachedir
    assert isinstance(tpot_obj._memory, Memory)


def test_memory_3():
    """Assert that the TPOT fit function does not clean up caching directory when memory is a valid path."""
    cachedir = mkdtemp()
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        config_dict='TPOT light',
        memory=cachedir,
        verbosity=0
    )
    tpot_obj.fit(training_features, training_target)

    assert tpot_obj._cachedir == cachedir
    assert os.path.isdir(tpot_obj._cachedir)
    assert isinstance(tpot_obj._memory, Memory)
    # clean up
    rmtree(cachedir)
    tpot_obj._memory = None


def test_memory_4():
    """Assert that the TPOT _setup_memory function create a directory which does not exist."""
    cachedir = mkdtemp()
    dir = cachedir + '/test'
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        config_dict='TPOT light',
        memory=dir,
        verbosity=0
    )
    tpot_obj._setup_memory()
    assert os.path.isdir(dir)
    rmtree(cachedir)




def test_memory_5():
    """Assert that the TPOT _setup_memory function runs normally with a Memory object."""
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        config_dict='TPOT light',
        memory=memory,
        verbosity=0
    )

    tpot_obj._setup_memory()
    rmtree(cachedir)
    assert tpot_obj.memory == memory
    assert tpot_obj._memory == memory
    # clean up
    tpot_obj._memory = None
    memory = None


def test_memory_6():
    """Assert that the TPOT _setup_memory function rasies ValueError with a invalid object."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        config_dict='TPOT light',
        memory=str,
        verbosity=0
    )

    assert_raises(ValueError, tpot_obj._setup_memory)


def test_check_periodic_pipeline():
    """Assert that the _check_periodic_pipeline exports periodic pipeline."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj.fit(training_features, training_target)
    with closing(StringIO()) as our_file:
        tpot_obj.log_file = our_file
        tpot_obj.verbosity = 3
        tpot_obj._last_pipeline_write = datetime.now()
        sleep(0.11)
        tpot_obj._output_best_pipeline_period_seconds = 0.1
        tmpdir = mkdtemp() + '/'
        tpot_obj.periodic_checkpoint_folder = tmpdir
        tpot_obj._check_periodic_pipeline(1)
        our_file.seek(0)
        assert_in('Saving periodic pipeline from pareto front', our_file.read())
        # clean up
        rmtree(tmpdir)


def test_check_periodic_pipeline_2():
    """Assert that the _check_periodic_pipeline rasie StopIteration if self._last_optimized_pareto_front_n_gens >= self.early_stop."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj.fit(training_features, training_target)
    tpot_obj.early_stop = 3
    # will pass
    tpot_obj._check_periodic_pipeline(1)
    tpot_obj._last_optimized_pareto_front_n_gens = 3
    assert_raises(StopIteration, tpot_obj._check_periodic_pipeline, 1)


def test_save_periodic_pipeline():
    """Assert that the _save_periodic_pipeline does not export periodic pipeline if exception happened"""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj.fit(training_features, training_target)
    with closing(StringIO()) as our_file:
        tpot_obj.log_file = our_file
        tpot_obj.verbosity = 3
        tpot_obj._last_pipeline_write = datetime.now()
        sleep(0.11)
        tpot_obj._output_best_pipeline_period_seconds = 0.1
        tmpdir = mkdtemp() + '/'
        tpot_obj.periodic_checkpoint_folder = tmpdir
        # reset _pareto_front to rasie exception
        tpot_obj._pareto_front = None

        tpot_obj._save_periodic_pipeline(1)
        our_file.seek(0)

        assert_in('Failed saving periodic pipeline, exception', our_file.read())
        #clean up
        rmtree(tmpdir)


def test_save_periodic_pipeline_2():
    """Assert that _save_periodic_pipeline creates the checkpoint folder and exports to it if it didn't exist"""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj.fit(training_features, training_target)
    with closing(StringIO()) as our_file:
        tpot_obj.log_file = our_file
        tpot_obj.verbosity = 3
        tpot_obj._last_pipeline_write = datetime.now()
        sleep(0.11)
        tpot_obj._output_best_pipeline_period_seconds = 0.1
        tmpdir = mkdtemp() + '_test/'
        tpot_obj.periodic_checkpoint_folder = tmpdir
        tpot_obj._save_periodic_pipeline(1)
        our_file.seek(0)

        msg = our_file.read()

        assert_in('Saving periodic pipeline from pareto front to {}'.format(tmpdir), msg)
        assert_in('Created new folder to save periodic pipeline: {}'.format(tmpdir), msg)

        #clean up
        rmtree(tmpdir)


def test_check_periodic_pipeline_3():
    """Assert that the _save_periodic_pipeline does not export periodic pipeline if the pipeline has been saved before."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj.fit(training_features, training_target)
    with closing(StringIO()) as our_file:
        tpot_obj.log_file = our_file
        tpot_obj.verbosity = 3
        tpot_obj._exported_pipeline_text = []
        tpot_obj._last_pipeline_write = datetime.now()
        sleep(0.11)
        tpot_obj._output_best_pipeline_period_seconds = 0
        tmpdir = mkdtemp() + '/'
        tpot_obj.periodic_checkpoint_folder = tmpdir
        # export once before
        tpot_obj._save_periodic_pipeline(1)

        tpot_obj._save_periodic_pipeline(2)

        our_file.seek(0)
        assert_in('Periodic pipeline was not saved, probably saved before...', our_file.read())
    #clean up
    rmtree(tmpdir)


def test_fit_predict():
    """Assert that the TPOT fit_predict function provides an optimized pipeline and correct output."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    result = tpot_obj.fit_predict(training_features, training_target)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)
    assert result.shape == (training_features.shape[0],)


def test_update_top_pipeline():
    """Assert that the TPOT _update_top_pipeline updated an optimized pipeline."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj.fit(training_features, training_target)
    tpot_obj._optimized_pipeline = None
    tpot_obj.fitted_pipeline_ = None
    tpot_obj._update_top_pipeline()

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)


def test_update_top_pipeline_2():
    """Assert that the TPOT _update_top_pipeline raises RuntimeError when self._pareto_front is empty."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj.fit(training_features, training_target)

    def pareto_eq(ind1, ind2):
        return np.allclose(ind1.fitness.values, ind2.fitness.values)

    tpot_obj._pareto_front = ParetoFront(similar=pareto_eq)

    assert_raises(RuntimeError, tpot_obj._update_top_pipeline)


def test_update_top_pipeline_3():
    """Assert that the TPOT _update_top_pipeline raises RuntimeError when self._optimized_pipeline is not updated."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj.fit(training_features, training_target)
    tpot_obj._optimized_pipeline = None
    # reset the fitness score to -float('inf')
    for pipeline_scores in reversed(tpot_obj._pareto_front.keys):
        pipeline_scores.wvalues = (5000., -float('inf'))

    assert_raises(RuntimeError, tpot_obj._update_top_pipeline)


def test_summary_of_best_pipeline():
    """Assert that the TPOT _update_top_pipeline raises RuntimeError when self._optimized_pipeline is not updated."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()

    assert_raises(RuntimeError, tpot_obj._summary_of_best_pipeline, features=training_features, target=training_target)


def test_evaluated_individuals_():
    """Assert that evaluated_individuals_ stores current pipelines and their CV scores."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=2,
        offspring_size=4,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj.fit(training_features, training_target)
    assert isinstance(tpot_obj.evaluated_individuals_, dict)
    for pipeline_string in sorted(tpot_obj.evaluated_individuals_.keys()):
        deap_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
        sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
        operator_count = tpot_obj._operator_count(deap_pipeline)

        try:
            cv_scores = model_selection.cross_val_score(sklearn_pipeline, training_features, training_target, cv=5, scoring='accuracy', verbose=0)
            mean_cv_scores = np.mean(cv_scores)
        except Exception:
            mean_cv_scores = -float('inf')
        assert np.allclose(tpot_obj.evaluated_individuals_[pipeline_string]['internal_cv_score'], mean_cv_scores)
        assert np.allclose(tpot_obj.evaluated_individuals_[pipeline_string]['operator_count'], operator_count)


def test_stop_by_max_time_mins():
    """Assert that _stop_by_max_time_mins raises KeyboardInterrupt when maximum minutes have elapsed."""
    tpot_obj = TPOTClassifier(config_dict='TPOT light')
    tpot_obj._start_datetime = datetime.now()
    sleep(0.11)
    tpot_obj.max_time_mins = 0.1/60.
    assert_raises(KeyboardInterrupt, tpot_obj._stop_by_max_time_mins)


def test_update_evaluated_individuals_():
    """Assert that _update_evaluated_individuals_ raises ValueError when scoring function does not return a float."""
    tpot_obj = TPOTClassifier(config_dict='TPOT light')
    assert_raises(ValueError, tpot_obj._update_evaluated_individuals_, ['Non-Float-Score'], ['Test_Pipeline'], [1], [dict])


def test_evaluate_individuals():
    """Assert that _evaluate_individuals returns operator_counts and CV scores in correct order."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()
    def pareto_eq(ind1, ind2):
        return np.allclose(ind1.fitness.values, ind2.fitness.values)

    tpot_obj._pareto_front = ParetoFront(similar=pareto_eq)

    tpot_obj._pbar = tqdm(total=1, disable=True)
    pop = tpot_obj._toolbox.population(n=10)
    pop = tpot_obj._evaluate_individuals(pop, training_features, training_target)
    fitness_scores = [ind.fitness.values for ind in pop]

    for deap_pipeline, fitness_score in zip(pop, fitness_scores):
        operator_count = tpot_obj._operator_count(deap_pipeline)
        sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cv_scores = model_selection.cross_val_score(sklearn_pipeline,
                                                            training_features,
                                                            training_target,
                                                            cv=5,
                                                            scoring='accuracy',
                                                            verbose=0,
                                                            error_score='raise')
            mean_cv_scores = np.mean(cv_scores)
        except Exception:
            mean_cv_scores = -float('inf')

        assert isinstance(deap_pipeline, creator.Individual)
        assert np.allclose(fitness_score[0], operator_count)
        assert np.allclose(fitness_score[1], mean_cv_scores)


def test_evaluate_individuals_2():
    """Assert that _evaluate_individuals returns operator_counts and CV scores in correct order with n_jobs=2"""
    tpot_obj = TPOTClassifier(
        n_jobs=2,
        random_state=42,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()
    def pareto_eq(ind1, ind2):
        return np.allclose(ind1.fitness.values, ind2.fitness.values)

    tpot_obj._pareto_front = ParetoFront(similar=pareto_eq)

    tpot_obj._pbar = tqdm(total=1, disable=True)
    pop = tpot_obj._toolbox.population(n=10)
    pop = tpot_obj._evaluate_individuals(pop, training_features, training_target)
    fitness_scores = [ind.fitness.values for ind in pop]

    for deap_pipeline, fitness_score in zip(pop, fitness_scores):
        operator_count = tpot_obj._operator_count(deap_pipeline)
        sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cv_scores = model_selection.cross_val_score(sklearn_pipeline,
                                                            training_features,
                                                            training_target,
                                                            cv=5,
                                                            scoring='accuracy',
                                                            verbose=0,
                                                            error_score='raise')
            mean_cv_scores = np.mean(cv_scores)
        except Exception:
            mean_cv_scores = -float('inf')

        assert isinstance(deap_pipeline, creator.Individual)
        assert np.allclose(fitness_score[0], operator_count)
        assert np.allclose(fitness_score[1], mean_cv_scores)


def test_update_pbar():
    """Assert that _update_pbar updates self._pbar with printing correct warning message."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()
    # reset verbosity = 3 for checking pbar message
    tpot_obj.verbosity = 3
    with closing(StringIO()) as our_file:
        tpot_obj.log_file=our_file
        tpot_obj._pbar = tqdm(total=10, disable=False, file=our_file)
        tpot_obj._update_pbar(pbar_num=2, pbar_msg="Test Warning Message")
        our_file.seek(0)
        assert_in("Test Warning Message", our_file.read())
        assert_equal(tpot_obj._pbar.n, 2)


def test_update_val():
    """Assert _update_val updates result score in list and prints timeout message."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()
    # reset verbosity = 3 for checking pbar message
    tpot_obj.verbosity = 3
    with closing(StringIO()) as our_file:
        tpot_obj.log_file=our_file
        tpot_obj._pbar = tqdm(total=10, disable=False, file=our_file)
        result_score_list = []
        result_score_list = tpot_obj._update_val(0.9999, result_score_list)
        assert_equal(result_score_list, [0.9999])
        # check "Timeout"
        result_score_list = tpot_obj._update_val("Timeout", result_score_list)
        our_file.seek(0)
        assert_in("Skipped pipeline #2 due to time out.", our_file.read())
        assert_equal(result_score_list, [0.9999, -float('inf')])


def test_preprocess_individuals():
    """Assert _preprocess_individuals preprocess DEAP individuals including one evaluated individual"""
    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0
    )
    tpot_obj._fit_init()

    pipeline_string_1 = (
        'LogisticRegression(PolynomialFeatures'
        '(input_matrix, PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=False, '
        'PolynomialFeatures__interaction_only=False), LogisticRegression__C=10.0, '
        'LogisticRegression__dual=False, LogisticRegression__penalty=l2)'
    )

    # a normal pipeline
    pipeline_string_2 = (
        'DecisionTreeClassifier('
        'input_matrix, '
        'DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, '
        'DecisionTreeClassifier__min_samples_split=5)'
    )

    individuals = []
    individuals.append(creator.Individual.from_string(pipeline_string_1, tpot_obj._pset))
    individuals.append(creator.Individual.from_string(pipeline_string_2, tpot_obj._pset))

    # set pipeline 2 has been evaluated before
    tpot_obj.evaluated_individuals_[pipeline_string_2] = (1, 0.99999)

    # reset verbosity = 3 for checking pbar message
    tpot_obj.verbosity = 3
    with closing(StringIO()) as our_file:
        tpot_obj.log_file=our_file
        tpot_obj._pbar = tqdm(total=2, disable=False, file=our_file)
        operator_counts, eval_individuals_str, sklearn_pipeline_list, _ = \
                                tpot_obj._preprocess_individuals(individuals)
        our_file.seek(0)
        assert_in("Pipeline encountered that has previously been evaluated", our_file.read())
        assert_in(pipeline_string_1, eval_individuals_str)
        assert_equal(operator_counts[pipeline_string_1], 2)
        assert_equal(len(sklearn_pipeline_list), 1)


def test_preprocess_individuals_2():
    """Assert _preprocess_individuals preprocess DEAP individuals with one invalid pipeline"""
    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0
    )
    tpot_obj._fit_init()

    # pipeline with two PolynomialFeatures operator
    pipeline_string_1 = (
        'LogisticRegression(PolynomialFeatures'
        '(PolynomialFeatures(input_matrix, PolynomialFeatures__degree=2, '
        'PolynomialFeatures__include_bias=False, PolynomialFeatures__interaction_only=False), '
        'PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=False, '
        'PolynomialFeatures__interaction_only=False), LogisticRegression__C=10.0, '
        'LogisticRegression__dual=False, LogisticRegression__penalty=l2)'
    )

    # a normal pipeline
    pipeline_string_2 = (
        'DecisionTreeClassifier('
        'input_matrix, '
        'DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, '
        'DecisionTreeClassifier__min_samples_split=5)'
    )

    individuals = []
    individuals.append(creator.Individual.from_string(pipeline_string_1, tpot_obj._pset))
    individuals.append(creator.Individual.from_string(pipeline_string_2, tpot_obj._pset))

    # reset verbosity = 3 for checking pbar message
    tpot_obj.verbosity = 3
    with closing(StringIO()) as our_file:
        tpot_obj.log_file=our_file
        tpot_obj._pbar = tqdm(total=3, disable=False, file=our_file)
        operator_counts, eval_individuals_str, sklearn_pipeline_list, _ = \
                                tpot_obj._preprocess_individuals(individuals)
        our_file.seek(0)

        assert_in("Invalid pipeline encountered. Skipping its evaluation.", our_file.read())
        assert_in(pipeline_string_2, eval_individuals_str)
        assert_equal(operator_counts[pipeline_string_2], 1)
        assert_equal(len(sklearn_pipeline_list), 1)


def test_preprocess_individuals_3():
    """Assert _preprocess_individuals updatas self._pbar.total when max_time_mins is not None"""
    tpot_obj = TPOTClassifier(
        population_size=2,
        offspring_size=4,
        random_state=42,
        max_time_mins=5,
        verbosity=0
    )
    tpot_obj._fit_init()

    pipeline_string_1 = (
        'LogisticRegression(PolynomialFeatures'
        '(input_matrix, PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=False, '
        'PolynomialFeatures__interaction_only=False), LogisticRegression__C=10.0, '
        'LogisticRegression__dual=False, LogisticRegression__penalty=l2)'
    )

    # a normal pipeline
    pipeline_string_2 = (
        'DecisionTreeClassifier('
        'input_matrix, '
        'DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, '
        'DecisionTreeClassifier__min_samples_split=5)'
    )

    individuals = []
    individuals.append(creator.Individual.from_string(pipeline_string_1, tpot_obj._pset))
    individuals.append(creator.Individual.from_string(pipeline_string_2, tpot_obj._pset))

    # reset verbosity = 3 for checking pbar message

    with closing(StringIO()) as our_file:
        tpot_obj.log_file=our_file
        tpot_obj._lambda=4
        tpot_obj._pbar = tqdm(total=2, disable=False, file=our_file)
        tpot_obj._pbar.n = 2
        _, _, _, _ = \
                                tpot_obj._preprocess_individuals(individuals)
        assert tpot_obj._pbar.total == 6

def test__init_pretest():
    """Assert that the init_pretest function produces a sample with all labels"""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()

    np.random.seed(seed=42)
    features = np.random.rand(10000,2)
    target = np.random.binomial(1,0.01,(10000,1))

    tpot_obj._init_pretest(features, target)
    assert(np.unique(tpot_obj.pretest_y).size == np.unique(target).size)

def test_check_dataset():
    """Assert that the check_dataset function returns feature and target as expected."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()

    ret_features, ret_target = tpot_obj._check_dataset(training_features, training_target)
    assert np.allclose(ret_features, training_features)
    assert np.allclose(ret_target, training_target)


def test_check_dataset_2():
    """Assert that the check_dataset function raise ValueError when sample_weight can not be converted to float array"""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()
    test_sample_weight = list(range(1, len(training_target)+1))
    _, _ = tpot_obj._check_dataset(training_features, training_target, test_sample_weight)
    test_sample_weight[0] = 'opps'

    assert_raises(ValueError, tpot_obj._check_dataset, training_features, training_target, test_sample_weight)


def test_check_dataset_3():
    """Assert that the check_dataset function raise ValueError when sample_weight has NaN"""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()
    test_sample_weight = list(range(1, len(training_target)+1))
    _, _ = tpot_obj._check_dataset(training_features, training_target, test_sample_weight)
    test_sample_weight[0] = np.nan

    assert_raises(ValueError, tpot_obj._check_dataset, training_features, training_target, test_sample_weight)


def test_check_dataset_4():
    """Assert that the check_dataset function raise ValueError when sample_weight has a length different length"""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()
    test_sample_weight = list(range(1, len(training_target)))
    assert_raises(ValueError, tpot_obj._check_dataset, training_features, training_target, test_sample_weight)


def test_check_dataset_5():
    """Assert that the check_dataset function returns feature and target as expected."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()
    ret_features = tpot_obj._check_dataset(training_features, target=None)
    assert np.allclose(ret_features, training_features)


def test_imputer():
    """Assert that the TPOT fit function will not raise a ValueError in a dataset where NaNs are present."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )

    tpot_obj.fit(features_with_nan, training_target)


def test_imputer_2():
    """Assert that the TPOT predict function will not raise a ValueError in a dataset where NaNs are present."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )

    tpot_obj.fit(training_features, training_target)
    assert_equal(tpot_obj._fitted_imputer, None)
    tpot_obj.predict(features_with_nan)
    assert_not_equal(tpot_obj._fitted_imputer, None)



def test_imputer_3():
    """Assert that the TPOT _impute_values function returns a feature matrix with imputed NaN values."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=2,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()

    with captured_output() as (out, err):
        imputed_features = tpot_obj._impute_values(features_with_nan)
        assert_in("Imputing missing values in feature set", out.getvalue())

    assert_not_equal(imputed_features[0][0], float('nan'))


def test_imputer_4():
    """Assert that the TPOT score function will not raise a ValueError in a dataset where NaNs are present."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )

    tpot_obj.fit(training_features, training_target)
    assert_equal(tpot_obj._fitted_imputer, None)
    tpot_obj.score(features_with_nan, training_target)
    assert_not_equal(tpot_obj._fitted_imputer, None)


def test_sparse_matrix():
    """Assert that the TPOT fit function will raise a ValueError in a sparse matrix with config_dict='TPOT light'."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )

    assert_raises(ValueError, tpot_obj.fit, sparse_features, sparse_target)


def test_sparse_matrix_2():
    """Assert that the TPOT fit function will raise a ValueError in a sparse matrix with config_dict=None."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict=None
    )

    assert_raises(ValueError, tpot_obj.fit, sparse_features, sparse_target)


def test_sparse_matrix_3():
    """Assert that the TPOT fit function will raise a ValueError in a sparse matrix with config_dict='TPOT MDR'."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT MDR'
    )

    assert_raises(ValueError, tpot_obj.fit, sparse_features, sparse_target)


def test_sparse_matrix_4():
    """Assert that the TPOT fit function will not raise a ValueError in a sparse matrix with config_dict='TPOT sparse'."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT sparse'
    )

    tpot_obj.fit(sparse_features, sparse_target)


def test_sparse_matrix_5():
    """Assert that the TPOT fit function will not raise a ValueError in a sparse matrix with a customized config dictionary."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='tests/test_config_sparse.py'
    )

    tpot_obj.fit(sparse_features, sparse_target)


def test_source_decode():
    """Assert that the source_decode can decode operator source and import operator class."""
    import_str, op_str, op_obj = source_decode("sklearn.linear_model.LogisticRegression")
    from sklearn.linear_model import LogisticRegression
    assert import_str == "sklearn.linear_model"
    assert op_str == "LogisticRegression"
    assert op_obj == LogisticRegression


def test_source_decode_2():
    """Assert that the source_decode return None when sourcecode is not available."""
    import_str, op_str, op_obj = source_decode("sklearn.linear_model.LogisticReg")
    from sklearn.linear_model import LogisticRegression
    assert import_str == "sklearn.linear_model"
    assert op_str == "LogisticReg"
    assert op_obj is None


def test_source_decode_3():
    """Assert that the source_decode raise ImportError when sourcecode is not available and verbose=3."""
    assert_raises(ImportError, source_decode, "sklearn.linear_model.LogisticReg", 3)


def test_tpot_operator_factory_class():
    """Assert that the TPOT operators class factory."""
    test_config_dict = {
        'sklearn.svm.LinearSVC': {
            'penalty': ["l1", "l2"],
            'loss': ["hinge", "squared_hinge"],
            'dual': [True, False],
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
        },

        'sklearn.linear_model.LogisticRegression': {
            'penalty': ["l1", "l2"],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'dual': [True, False]
        },

        'sklearn.preprocessing.Binarizer': {
            'threshold': np.arange(0.0, 1.01, 0.05)
        }
    }

    tpot_operator_list = []
    tpot_argument_list = []

    for key in sorted(test_config_dict.keys()):
        op, args = TPOTOperatorClassFactory(key, test_config_dict[key])
        tpot_operator_list.append(op)
        tpot_argument_list += args

    assert len(tpot_operator_list) == 3
    assert len(tpot_argument_list) == 9
    assert tpot_operator_list[0].root is True
    assert tpot_operator_list[1].root is False
    assert tpot_operator_list[2].type() == "Classifier"
    assert tpot_argument_list[1].values == [True, False]


def test_PolynomialFeatures_exception():
    """Assert that TPOT allows only one PolynomialFeatures operator in a pipeline."""

    tpot_obj._pbar = tqdm(total=1, disable=True)
    def pareto_eq(ind1, ind2):
        return np.allclose(ind1.fitness.values, ind2.fitness.values)

    tpot_obj._pareto_front = ParetoFront(similar=pareto_eq)
    # pipeline with one PolynomialFeatures operator
    pipeline_string_1 = (
        'LogisticRegression(PolynomialFeatures'
        '(input_matrix, PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=False, '
        'PolynomialFeatures__interaction_only=False), LogisticRegression__C=10.0, '
        'LogisticRegression__dual=False, LogisticRegression__penalty=l2)'
    )

    # pipeline with two PolynomialFeatures operator
    pipeline_string_2 = (
        'LogisticRegression(PolynomialFeatures'
        '(PolynomialFeatures(input_matrix, PolynomialFeatures__degree=2, '
        'PolynomialFeatures__include_bias=False, PolynomialFeatures__interaction_only=False), '
        'PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=False, '
        'PolynomialFeatures__interaction_only=False), LogisticRegression__C=10.0, '
        'LogisticRegression__dual=False, LogisticRegression__penalty=l2)'
    )

    # make a list for _evaluate_individuals
    pipelines = []
    pipelines.append(creator.Individual.from_string(pipeline_string_1, tpot_obj._pset))
    pipelines.append(creator.Individual.from_string(pipeline_string_2, tpot_obj._pset))

    for pipeline in pipelines:
        initialize_stats_dict(pipeline)

    pop = tpot_obj._evaluate_individuals(pipelines, pretest_X, pretest_y)
    fitness_scores = [ind.fitness.values for ind in pop]
    assert fitness_scores[0][0] == 2
    assert fitness_scores[1][0] == 5000.0


def test_pick_two_individuals_eligible_for_crossover():
    """Assert that pick_two_individuals_eligible_for_crossover() picks the correct pair of nodes to perform crossover with"""

    ind1 = creator.Individual.from_string(
        'BernoulliNB(input_matrix, BernoulliNB__alpha=1.0, BernoulliNB__fit_prior=True)',
        tpot_obj._pset
    )
    ind2 = creator.Individual.from_string(
        'BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=True)',
        tpot_obj._pset
    )
    ind3 = creator.Individual.from_string(
        'GaussianNB(input_matrix)',
        tpot_obj._pset
    )

    pick1, pick2 = pick_two_individuals_eligible_for_crossover([ind1, ind2, ind3])
    assert ((str(pick1) == str(ind1) and str(pick2) == str(ind2)) or
             str(pick1) == str(ind2) and str(pick2) == str(ind1))

    ind4 = creator.Individual.from_string(
        'KNeighborsClassifier('
        'BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=True),'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')',
        tpot_obj._pset
    )

    # Eventhough ind4 does not have the same primitive at the root, the tree shares a primitive with ind1
    pick1, pick2 = pick_two_individuals_eligible_for_crossover([ind1, ind3, ind4])
    assert ((str(pick1) == str(ind1) and str(pick2) == str(ind4)) or
             str(pick1) == str(ind4) and str(pick2) == str(ind1))


def test_pick_two_individuals_eligible_for_crossover_bad():
    """Assert that pick_two_individuals_eligible_for_crossover() returns the right output when no pair is eligible"""

    ind1 = creator.Individual.from_string(
        'BernoulliNB(input_matrix, BernoulliNB__alpha=1.0, BernoulliNB__fit_prior=True)',
        tpot_obj._pset
    )
    ind2 = creator.Individual.from_string(
        'BernoulliNB(input_matrix, BernoulliNB__alpha=1.0, BernoulliNB__fit_prior=True)',
        tpot_obj._pset
    )
    ind3 = creator.Individual.from_string(
        'GaussianNB(input_matrix)',
        tpot_obj._pset
    )

    # Ind1 and ind2 are not a pair because they are the same, ind3 shares no primitive
    pick1, pick2 = pick_two_individuals_eligible_for_crossover([ind1, ind2, ind3])
    assert pick1 is None and pick2 is None

    # You can not do crossover with a population of only 1.
    pick1, pick2 = pick_two_individuals_eligible_for_crossover([ind1])
    assert pick1 is None and pick2 is None

    # You can not do crossover with a population of 0.
    pick1, pick2 = pick_two_individuals_eligible_for_crossover([])
    assert pick1 is None and pick2 is None


def test_mate_operator():
    """Assert that self._mate_operator returns offsprings as expected."""

    ind1 = creator.Individual.from_string(
        'KNeighborsClassifier('
        'BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=False),'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')',
        tpot_obj._pset
    )
    ind2 = creator.Individual.from_string(
        'KNeighborsClassifier('
        'BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=True),'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=2, '
        'KNeighborsClassifier__weights=uniform'
        ')',
        tpot_obj._pset
    )

    # Initialize stats
    initialize_stats_dict(ind1)
    initialize_stats_dict(ind2)


    # set as evaluated pipelines in tpot_obj.evaluated_individuals_
    tpot_obj.evaluated_individuals_[str(ind1)] = (2, 0.99)
    tpot_obj.evaluated_individuals_[str(ind2)] = (2, 0.99)

    offspring1, _ = tpot_obj._mate_operator(ind1, ind2)
    expected_offspring1 = (
        'KNeighborsClassifier('
        'BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=False), '
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=2, '
        'KNeighborsClassifier__weights=uniform'
        ')'
    )
    expected_offspring1_alt = (
        'KNeighborsClassifier('
        'BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=True), '
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')'
    )
    assert str(offspring1) in [expected_offspring1, expected_offspring1_alt]


def test_cxOnePoint():
    """Assert that cxOnePoint() returns the correct type of node between two fixed pipelines."""

    ind1 = creator.Individual.from_string(
        'KNeighborsClassifier('
        'BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=False),'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')',
        tpot_obj._pset
    )
    ind2 = creator.Individual.from_string(
        'KNeighborsClassifier('
        'BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=True),'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=2, '
        'KNeighborsClassifier__weights=uniform'
        ')',
        tpot_obj._pset
    )
    ind1[0].ret = Output_Array
    ind2[0].ret = Output_Array
    ind1_copy, ind2_copy = tpot_obj._toolbox.clone(ind1),tpot_obj._toolbox.clone(ind2)
    offspring1, offspring2 = cxOnePoint(ind1_copy, ind2_copy)

    assert offspring1[0].ret == Output_Array
    assert offspring2[0].ret == Output_Array


def test_mutNodeReplacement():
    """Assert that mutNodeReplacement() returns the correct type of mutation node in a fixed pipeline."""

    pipeline_string = (
        'LogisticRegression(PolynomialFeatures'
        '(input_matrix, PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=False, '
        'PolynomialFeatures__interaction_only=False), LogisticRegression__C=10.0, '
        'LogisticRegression__dual=False, LogisticRegression__penalty=l2)'
    )

    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    pipeline[0].ret = Output_Array
    old_ret_type_list = [node.ret for node in pipeline]
    old_prims_list = [node for node in pipeline if node.arity != 0]

    # test 10 times
    for _ in range(10):
        mut_ind = mutNodeReplacement(tpot_obj._toolbox.clone(pipeline), pset=tpot_obj._pset)
        new_ret_type_list = [node.ret for node in mut_ind[0]]
        new_prims_list = [node for node in mut_ind[0] if node.arity != 0]

        if new_prims_list == old_prims_list:  # Terminal mutated
            assert new_ret_type_list == old_ret_type_list
        else:  # Primitive mutated
            diff_prims = [x for x in new_prims_list if x not in old_prims_list]
            diff_prims += [x for x in old_prims_list if x not in new_prims_list]
            if len(diff_prims) > 1: # Sometimes mutation randomly replaces an operator that already in the pipelines
                assert diff_prims[0].ret == diff_prims[1].ret
        assert mut_ind[0][0].ret == Output_Array


def test_mutNodeReplacement_2():
    """Assert that mutNodeReplacement() returns the correct type of mutation node in a complex pipeline."""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()
    # a pipeline with 4 operators
    pipeline_string = (
        "LogisticRegression("
        "KNeighborsClassifier(BernoulliNB(PolynomialFeatures"
        "(input_matrix, PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=False, "
        "PolynomialFeatures__interaction_only=False), BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=False), "
        "KNeighborsClassifier__n_neighbors=10, KNeighborsClassifier__p=1, KNeighborsClassifier__weights=uniform),"
        "LogisticRegression__C=10.0, LogisticRegression__dual=False, LogisticRegression__penalty=l2)"
    )

    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    pipeline[0].ret = Output_Array

    old_ret_type_list = [node.ret for node in pipeline]
    old_prims_list = [node for node in pipeline if node.arity != 0]

    # test 30 times
    for _ in range(30):
        mut_ind = mutNodeReplacement(tpot_obj._toolbox.clone(pipeline), pset=tpot_obj._pset)
        new_ret_type_list = [node.ret for node in mut_ind[0]]
        new_prims_list = [node for node in mut_ind[0] if node.arity != 0]
        if new_prims_list == old_prims_list:  # Terminal mutated
            assert new_ret_type_list == old_ret_type_list
        else:  # Primitive mutated
            Primitive_Count = 0
            for node in mut_ind[0]:
                if isinstance(node, gp.Primitive):
                    Primitive_Count += 1
            assert Primitive_Count == 4
            diff_prims = [x for x in new_prims_list if x not in old_prims_list]
            diff_prims += [x for x in old_prims_list if x not in new_prims_list]
            if len(diff_prims) > 1: # Sometimes mutation randomly replaces an operator that already in the pipelines
                assert diff_prims[0].ret == diff_prims[1].ret
        assert mut_ind[0][0].ret == Output_Array


def test_varOr():
    """Assert that varOr() applys crossover only and removes CV scores in offsprings."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()

    tpot_obj._pbar = tqdm(total=1, disable=True)
    pop = tpot_obj._toolbox.population(n=5)
    for ind in pop:
        initialize_stats_dict(ind)
        ind.fitness.values = (2, 1.0)


    offspring = varOr(pop, tpot_obj._toolbox, 5, cxpb=1.0, mutpb=0.0)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    assert len(offspring) == 5
    assert len(invalid_ind) == 5


def test_varOr_2():
    """Assert that varOr() applys mutation only and removes CV scores in offsprings."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()

    tpot_obj._pbar = tqdm(total=1, disable=True)
    pop = tpot_obj._toolbox.population(n=5)
    for ind in pop:
        initialize_stats_dict(ind)
        ind.fitness.values = (2, 1.0)


    offspring = varOr(pop, tpot_obj._toolbox, 5, cxpb=0.0, mutpb=1.0)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    assert len(offspring) == 5
    assert len(invalid_ind) == 5


def test_varOr_3():
    """Assert that varOr() applys reproduction only and does NOT remove CV scores in offsprings."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        verbosity=0,
        config_dict='TPOT light'
    )
    tpot_obj._fit_init()

    tpot_obj._pbar = tqdm(total=1, disable=True)
    pop = tpot_obj._toolbox.population(n=5)
    for ind in pop:
        ind.fitness.values = (2, 1.0)

    offspring = varOr(pop, tpot_obj._toolbox, 5, cxpb=0.0, mutpb=0.0)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    assert len(offspring) == 5
    assert len(invalid_ind) == 0


def test_operator_type():
    """Assert that TPOT operators return their type, e.g. 'Classifier', 'Preprocessor'."""
    assert TPOTSelectPercentile.type() == "Selector"


def test_gen():
    """Assert that TPOT's gen_grow_safe function returns a pipeline of expected structure."""


    pipeline = tpot_obj._gen_grow_safe(tpot_obj._pset, 1, 3)

    assert len(pipeline) > 1
    assert pipeline[0].ret == Output_Array


def test_clean_pipeline_string():
    """Assert that clean_pipeline_string correctly returns a string without parameter prefixes"""

    with_prefix = 'BernoulliNB(input_matrix, BernoulliNB__alpha=1.0, BernoulliNB__fit_prior=True)'
    without_prefix = 'BernoulliNB(input_matrix, alpha=1.0, fit_prior=True)'

    ind1 = creator.Individual.from_string(with_prefix, tpot_obj._pset)

    pretty_string = tpot_obj.clean_pipeline_string(ind1)
    assert pretty_string == without_prefix
