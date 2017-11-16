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
from tpot.operator_utils import TPOTOperatorClassFactory, set_sample_weight
from tpot.decorators import pretest_X, pretest_y

from tpot.config.classifier import classifier_config_dict
from tpot.config.classifier_light import classifier_config_dict_light
from tpot.config.regressor_light import regressor_config_dict_light
from tpot.config.classifier_mdr import tpot_mdr_classifier_config_dict
from tpot.config.regressor_mdr import tpot_mdr_regressor_config_dict
from tpot.config.regressor_sparse import regressor_config_sparse
from tpot.config.classifier_sparse import classifier_config_sparse

import numpy as np
from scipy import sparse
import inspect
import random
import warnings
from multiprocessing import cpu_count
import os
from re import search
from datetime import datetime
from time import sleep
from tempfile import mkdtemp
from shutil import rmtree

from sklearn.datasets import load_digits, load_boston
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.externals.joblib import Memory
from sklearn.metrics import make_scorer
from deap import creator, gp
from deap.tools import ParetoFront
from nose.tools import assert_raises, assert_not_equal, assert_greater_equal, assert_equal, assert_in
from driver_tests import captured_output

from tqdm import tqdm

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

# Set up the MNIST data set for testing
mnist_data = load_digits()
training_features, testing_features, training_target, testing_target = \
    train_test_split(mnist_data.data.astype(np.float64), mnist_data.target.astype(np.float64), random_state=42)

# Set up the Boston data set for testing
boston_data = load_boston()
training_features_r, testing_features_r, training_target_r, testing_target_r = \
    train_test_split(boston_data.data, boston_data.target, random_state=42)

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
        warm_start=True
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
    assert tpot_obj._optimized_pipeline is None
    assert tpot_obj.fitted_pipeline_ is None
    assert not (tpot_obj._pset is None)
    assert not (tpot_obj._toolbox is None)


def test_init_default_scoring():
    """Assert that TPOT intitializes with the correct default scoring function."""
    tpot_obj = TPOTRegressor()
    assert tpot_obj.scoring_function == 'neg_mean_squared_error'

    tpot_obj = TPOTClassifier()
    assert tpot_obj.scoring_function == 'accuracy'


def test_init_default_scoring_2():
    """Assert that TPOT intitializes with a valid customized metric function."""
    with warnings.catch_warnings(record=True) as w:
        tpot_obj = TPOTClassifier(scoring=balanced_accuracy)
    assert len(w) == 1
    assert issubclass(w[-1].category, DeprecationWarning)
    assert "This scoring type was deprecated" in str(w[-1].message)
    assert tpot_obj.scoring_function == 'balanced_accuracy'


def test_init_default_scoring_3():
    """Assert that TPOT intitializes with a valid _BaseScorer."""
    with warnings.catch_warnings(record=True) as w:
        tpot_obj = TPOTClassifier(scoring=make_scorer(balanced_accuracy))
    assert len(w) == 0
    assert tpot_obj.scoring_function == 'balanced_accuracy'


def test_init_default_scoring_4():
    """Assert that TPOT intitializes with a valid scorer."""
    def my_scorer(clf, X, y):
        return 0.9

    with warnings.catch_warnings(record=True) as w:
        tpot_obj = TPOTClassifier(scoring=my_scorer)
    assert len(w) == 0
    assert tpot_obj.scoring_function == 'my_scorer'


def test_invalid_score_warning():
    """Assert that the TPOT intitializes raises a ValueError when the scoring metrics is not available in SCORERS."""
    # Mis-spelled scorer
    assert_raises(ValueError, TPOTClassifier, scoring='balanced_accuray')
    # Correctly spelled
    TPOTClassifier(scoring='balanced_accuracy')


def test_invalid_dataset_warning():
    """Assert that the TPOT fit function raises a ValueError when dataset is not in right format."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0
    )
    # common mistake in target
    bad_training_target = training_target.reshape((1, len(training_target)))
    assert_raises(ValueError, tpot_obj.fit, training_features, bad_training_target)


def test_invalid_subsample_ratio_warning():
    """Assert that the TPOT intitializes raises a ValueError when subsample ratio is not in the range (0.0, 1.0]."""
    # Invalid ratio
    assert_raises(ValueError, TPOTClassifier, subsample=0.0)
    # Valid ratio
    TPOTClassifier(subsample=0.1)


def test_invalid_mut_rate_plus_xo_rate():
    """Assert that the TPOT intitializes raises a ValueError when the sum of crossover and mutation probabilities is large than 1."""
    # Invalid ratio
    assert_raises(ValueError, TPOTClassifier, mutation_rate=0.8, crossover_rate=0.8)
    # Valid ratio
    TPOTClassifier(mutation_rate=0.8, crossover_rate=0.1)


def test_init_max_time_mins():
    """Assert that the TPOT init stores max run time and sets generations to 1000000."""
    tpot_obj = TPOTClassifier(max_time_mins=30, generations=1000)

    assert tpot_obj.generations == 1000000
    assert tpot_obj.max_time_mins == 30


def test_init_n_jobs():
    """Assert that the TPOT init stores current number of processes."""
    tpot_obj = TPOTClassifier(n_jobs=2)
    assert tpot_obj.n_jobs == 2

    tpot_obj = TPOTClassifier(n_jobs=-1)
    assert tpot_obj.n_jobs == cpu_count()


def test_timeout():
    """Assert that _wrapped_cross_val_score return Timeout in a time limit."""
    tpot_obj = TPOTRegressor(scoring='neg_mean_squared_error')
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
    return_value = _wrapped_cross_val_score(tpot_obj.fitted_pipeline_,
                                            training_features_r,
                                            training_target_r,
                                            cv=20,
                                            scoring_function='neg_mean_squared_error',
                                            sample_weight=None,
                                            groups=None,
                                            timeout=1)
    assert return_value == "Timeout"


def test_invalid_pipeline():
    """Assert that _wrapped_cross_val_score return -float(\'inf\') with a invalid_pipeline"""
    tpot_obj = TPOTClassifier()
    # a invalid pipeline
    # Dual or primal formulation. Dual formulation is only implemented for l2 penalty.
    pipeline_string = (
        'LogisticRegression(input_matrix,  LogisticRegression__C=10.0, '
        'LogisticRegression__dual=True, LogisticRegression__penalty=l1)'
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj.fitted_pipeline_ = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    # test _wrapped_cross_val_score with cv=20 so that it is impossible to finish in 1 second
    return_value = _wrapped_cross_val_score(tpot_obj.fitted_pipeline_,
                                            training_features,
                                            training_target,
                                            cv=5,
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
    # update to dictionary instead of input string
    default_kwargs.update({'config_dict': classifier_config_dict_light})
    assert tpot_obj.get_params()['config_dict'] == default_kwargs['config_dict']
    assert tpot_obj.get_params() == default_kwargs


def test_set_params():
    """Assert that set_params returns a reference to the TPOT instance."""
    tpot_obj = TPOTClassifier()
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
    assert tpot_obj.config_dict == classifier_config_dict_light

    tpot_obj = TPOTClassifier(config_dict='TPOT MDR')
    assert tpot_obj.config_dict == tpot_mdr_classifier_config_dict

    tpot_obj = TPOTClassifier(config_dict='TPOT sparse')
    assert tpot_obj.config_dict == classifier_config_sparse

    tpot_obj = TPOTRegressor(config_dict='TPOT light')
    assert tpot_obj.config_dict == regressor_config_dict_light

    tpot_obj = TPOTRegressor(config_dict='TPOT MDR')
    assert tpot_obj.config_dict == tpot_mdr_regressor_config_dict

    tpot_obj = TPOTRegressor(config_dict='TPOT sparse')
    assert tpot_obj.config_dict == regressor_config_sparse


def test_conf_dict_2():
    """Assert that TPOT uses a custom dictionary of operators when config_dict is Python dictionary."""
    tpot_obj = TPOTClassifier(config_dict=tpot_mdr_classifier_config_dict)
    assert tpot_obj.config_dict == tpot_mdr_classifier_config_dict


def test_conf_dict_3():
    """Assert that TPOT uses a custom dictionary of operators when config_dict is the path of Python dictionary."""
    tpot_obj = TPOTRegressor(config_dict='tests/test_config.py')
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
    assert isinstance(tpot_obj.config_dict, dict)
    assert tpot_obj.config_dict == tested_config_dict


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
    pipeline1 = str(tpot_obj._toolbox.individual())
    tpot_obj = TPOTClassifier(random_state=43)
    pipeline2 = str(tpot_obj._toolbox.individual())
    assert pipeline1 == pipeline2


def test_random_ind_2():
    """Assert that the TPOTRegressor can generate the same pipeline with same random seed."""
    tpot_obj = TPOTRegressor(random_state=43)
    pipeline1 = str(tpot_obj._toolbox.individual())
    tpot_obj = TPOTRegressor(random_state=43)
    pipeline2 = str(tpot_obj._toolbox.individual())

    assert pipeline1 == pipeline2


def test_score():
    """Assert that the TPOT score function raises a RuntimeError when no optimized pipeline exists."""
    tpot_obj = TPOTClassifier()

    assert_raises(RuntimeError, tpot_obj.score, testing_features, testing_target)


def test_score_2():
    """Assert that the TPOTClassifier score function outputs a known score for a fixed pipeline."""
    tpot_obj = TPOTClassifier(random_state=34)
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
    known_score = 12.1791953611

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

    assert np.allclose(known_score, score)


def test_sample_weight_func():
    """Assert that the TPOTRegressor score function outputs a known score for a fixed pipeline with sample weights."""
    tpot_obj = TPOTRegressor(scoring='neg_mean_squared_error')

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
    cv_score1 = cross_val_score(tpot_obj.fitted_pipeline_, training_features_r, training_target_r, cv=3, scoring='neg_mean_squared_error')

    np.random.seed(42)
    cv_score2 = cross_val_score(tpot_obj.fitted_pipeline_, training_features_r, training_target_r, cv=3, scoring='neg_mean_squared_error')

    np.random.seed(42)
    cv_score_weight = cross_val_score(tpot_obj.fitted_pipeline_, training_features_r, training_target_r, cv=3, scoring='neg_mean_squared_error', fit_params=training_target_r_weight_dict)

    np.random.seed(42)
    tpot_obj.fitted_pipeline_.fit(training_features_r, training_target_r, **training_target_r_weight_dict)
    # Get score from TPOT
    known_score = 11.5790430757
    score = tpot_obj.score(testing_features_r, testing_target_r)

    assert np.allclose(cv_score1, cv_score2)
    assert not np.allclose(cv_score1, cv_score_weight)
    assert np.allclose(known_score, score)


def test_fit_GroupKFold():
    """Assert that TPOT properly handles the group parameter when using GroupKFold."""
    # This check tests if the darker MNIST images would generalize to the lighter ones.
    means = np.mean(training_features, axis=1)
    groups = means >= np.median(means)

    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=2,
        offspring_size=4,
        generations=1,
        verbosity=0,
        config_dict='TPOT light',
        cv=GroupKFold(n_splits=2),
    )
    tpot_obj.fit(training_features, training_target, groups=groups)

    assert_greater_equal(tpot_obj.score(testing_features, testing_target), 0.97)


def test_predict():
    """Assert that the TPOT predict function raises a RuntimeError when no optimized pipeline exists."""
    tpot_obj = TPOTClassifier()

    assert_raises(RuntimeError, tpot_obj.predict, testing_features)


def test_predict_2():
    """Assert that the TPOT predict function returns a numpy matrix of shape (num_testing_rows,)."""
    tpot_obj = TPOTClassifier()
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


def test_predict_proba():
    """Assert that the TPOT predict_proba function returns a numpy matrix of shape (num_testing_rows, num_testing_target)."""
    tpot_obj = TPOTClassifier()
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

    assert_raises(RuntimeError, tpot_obj.predict_proba, testing_features)


def test_predict_proba_4():
    """Assert that the TPOT predict_proba function raises a RuntimeError when the optimized pipeline do not have the predict_proba() function"""
    tpot_obj = TPOTRegressor()
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
        generations=1,
        verbosity=0,
        max_time_mins=2/60.,
        config_dict='TPOT light'
    )
    assert tpot_obj.generations == 1000000

    # reset generations to 20 just in case that the failed test may take too much time
    tpot_obj.generations == 20

    tpot_obj.fit(training_features, training_target)

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
    """Assert that the TPOT _setup_memory function rasies ValueError with a invalid path."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        config_dict='TPOT light',
        memory="./fake_temp_dir",
        verbosity=0
    )

    assert_raises(ValueError, tpot_obj._setup_memory)


def test_memory_5():
    """Assert that the TPOT _setup_memory function runs normally with a Memory object."""
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)
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
        tpot_obj._file = our_file
        tpot_obj.verbosity = 3
        tpot_obj._last_pipeline_write = datetime.now()
        sleep(0.11)
        tpot_obj._output_best_pipeline_period_seconds = 0.1
        tpot_obj.periodic_checkpoint_folder = './'
        tpot_obj._check_periodic_pipeline()
        our_file.seek(0)

        assert_in('Saving best periodic pipeline to ./pipeline', our_file.read())
        # clean up
        for f in os.listdir('./'):
            if search('pipeline_', f):
                os.remove(os.path.join('./', f))


def test_check_periodic_pipeline_2():
    """Assert that the _check_periodic_pipeline does not export periodic pipeline if the pipeline has been saved before."""
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
        tpot_obj._file = our_file
        tpot_obj.verbosity = 3
        tpot_obj._last_pipeline_write = datetime.now()
        sleep(0.11)
        tpot_obj._output_best_pipeline_period_seconds = 0.1
        tpot_obj.periodic_checkpoint_folder = './'
        # export once before
        tpot_obj.export('./pipeline_test.py')
        tpot_obj._check_periodic_pipeline()
        our_file.seek(0)

        assert_in('Periodic pipeline was not saved, probably saved before...', our_file.read())
        # clean up
        for f in os.listdir('./'):
            if search('pipeline_', f):
                os.remove(os.path.join('./', f))


def test_check_periodic_pipeline_3():
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
    tpot_obj._check_periodic_pipeline()
    tpot_obj._last_optimized_pareto_front_n_gens = 3
    assert_raises(StopIteration, tpot_obj._check_periodic_pipeline)


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
        tpot_obj._file = our_file
        tpot_obj.verbosity = 3
        tpot_obj._last_pipeline_write = datetime.now()
        sleep(0.11)
        tpot_obj._output_best_pipeline_period_seconds = 0.1
        tpot_obj.periodic_checkpoint_folder = './'
        # reset _optimized_pipeline to rasie exception
        tpot_obj._optimized_pipeline = None

        tpot_obj._save_periodic_pipeline()
        our_file.seek(0)

        assert_in('Failed saving periodic pipeline, exception', our_file.read())
        # clean up
        for f in os.listdir('./'):
            if search('pipeline_', f):
                os.remove(os.path.join('./', f))

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
        tpot_obj._file = our_file
        tpot_obj.verbosity = 3
        tpot_obj._last_pipeline_write = datetime.now()
        sleep(0.11)
        tpot_obj._output_best_pipeline_period_seconds = 0.1
        tpot_obj.periodic_checkpoint_folder = './tmp'
        tpot_obj._save_periodic_pipeline()
        our_file.seek(0)

        msg = our_file.read()
        expected_filepath_prefix = os.path.join('./tmp', 'pipeline_')
        assert_in('Saving best periodic pipeline to ' + expected_filepath_prefix, msg)
        assert_in('Created new folder to save periodic pipeline: ./tmp', msg)

        # clean up
        for f in os.listdir('./tmp'):
            if search('pipeline_', f):
                os.remove(os.path.join('./tmp', f))
        os.rmdir('./tmp')


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

    assert_raises(RuntimeError, tpot_obj._summary_of_best_pipeline, features=training_features, target=training_target)


def test_set_param_recursive():
    """Assert that _set_param_recursive sets \"random_state\" to 42 in all steps in a simple pipeline."""
    pipeline_string = (
        'DecisionTreeClassifier(PCA(input_matrix, PCA__iterated_power=5, PCA__svd_solver=randomized), '
        'DecisionTreeClassifier__criterion=gini, DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, DecisionTreeClassifier__min_samples_split=5)'
    )
    tpot_obj = TPOTClassifier()
    deap_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
    tpot_obj._set_param_recursive(sklearn_pipeline.steps, 'random_state', 42)
    # assert "random_state" of PCA at step 1
    assert getattr(sklearn_pipeline.steps[0][1], 'random_state') == 42
    # assert "random_state" of DecisionTreeClassifier at step 2
    assert getattr(sklearn_pipeline.steps[1][1], 'random_state') == 42


def test_set_param_recursive_2():
    """Assert that _set_param_recursive sets \"random_state\" to 42 in nested estimator in SelectFromModel."""
    pipeline_string = (
        'DecisionTreeRegressor(SelectFromModel(input_matrix, '
        'SelectFromModel__ExtraTreesRegressor__max_features=0.05, SelectFromModel__ExtraTreesRegressor__n_estimators=100, '
        'SelectFromModel__threshold=0.05), DecisionTreeRegressor__max_depth=8,'
        'DecisionTreeRegressor__min_samples_leaf=5, DecisionTreeRegressor__min_samples_split=5)'
    )
    tpot_obj = TPOTRegressor()
    deap_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
    tpot_obj._set_param_recursive(sklearn_pipeline.steps, 'random_state', 42)

    assert getattr(getattr(sklearn_pipeline.steps[0][1], 'estimator'), 'random_state') == 42
    assert getattr(sklearn_pipeline.steps[1][1], 'random_state') == 42


def test_set_param_recursive_3():
    """Assert that _set_param_recursive sets \"random_state\" to 42 in nested estimator in StackingEstimator in a complex pipeline."""
    pipeline_string = (
        'DecisionTreeClassifier(CombineDFs('
        'DecisionTreeClassifier(input_matrix, DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8, DecisionTreeClassifier__min_samples_leaf=5,'
        'DecisionTreeClassifier__min_samples_split=5),input_matrix) '
        'DecisionTreeClassifier__criterion=gini, DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, DecisionTreeClassifier__min_samples_split=5)'
    )
    tpot_obj = TPOTClassifier()
    deap_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
    tpot_obj._set_param_recursive(sklearn_pipeline.steps, 'random_state', 42)

    # StackingEstimator under the transformer_list of FeatureUnion
    assert getattr(getattr(sklearn_pipeline.steps[0][1].transformer_list[0][1], 'estimator'), 'random_state') == 42
    assert getattr(sklearn_pipeline.steps[1][1], 'random_state') == 42


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
        tpot_obj._set_param_recursive(sklearn_pipeline.steps, 'random_state', 42)
        operator_count = tpot_obj._operator_count(deap_pipeline)

        try:
            cv_scores = cross_val_score(sklearn_pipeline, training_features, training_target, cv=5, scoring='accuracy', verbose=0)
            mean_cv_scores = np.mean(cv_scores)
        except Exception as e:
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

    tpot_obj._pbar = tqdm(total=1, disable=True)
    pop = tpot_obj._toolbox.population(n=10)
    fitness_scores = tpot_obj._evaluate_individuals(pop, training_features, training_target)

    for deap_pipeline, fitness_score in zip(pop, fitness_scores):
        operator_count = tpot_obj._operator_count(deap_pipeline)
        sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
        tpot_obj._set_param_recursive(sklearn_pipeline.steps, 'random_state', 42)

        try:
            cv_scores = cross_val_score(sklearn_pipeline, training_features, training_target, cv=5, scoring='accuracy', verbose=0)
            mean_cv_scores = np.mean(cv_scores)
        except Exception as e:
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

    tpot_obj._pbar = tqdm(total=1, disable=True)
    pop = tpot_obj._toolbox.population(n=10)
    fitness_scores = tpot_obj._evaluate_individuals(pop, training_features, training_target)

    for deap_pipeline, fitness_score in zip(pop, fitness_scores):
        operator_count = tpot_obj._operator_count(deap_pipeline)
        sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
        tpot_obj._set_param_recursive(sklearn_pipeline.steps, 'random_state', 42)

        try:
            cv_scores = cross_val_score(sklearn_pipeline, training_features, training_target, cv=5, scoring='accuracy', verbose=0)
            mean_cv_scores = np.mean(cv_scores)
        except Exception as e:
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
    # reset verbosity = 3 for checking pbar message
    tpot_obj.verbosity = 3
    with closing(StringIO()) as our_file:
        tpot_obj._file=our_file
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
    # reset verbosity = 3 for checking pbar message
    tpot_obj.verbosity = 3
    with closing(StringIO()) as our_file:
        tpot_obj._file=our_file
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
        tpot_obj._file=our_file
        tpot_obj._pbar = tqdm(total=2, disable=False, file=our_file)
        operator_counts, eval_individuals_str, sklearn_pipeline_list, stats_dicts = \
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
        tpot_obj._file=our_file
        tpot_obj._pbar = tqdm(total=3, disable=False, file=our_file)
        operator_counts, eval_individuals_str, sklearn_pipeline_list, stats_dicts = \
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
        tpot_obj._file=our_file
        tpot_obj._pbar = tqdm(total=2, disable=False, file=our_file)
        tpot_obj._pbar.n = 2
        operator_counts, eval_individuals_str, sklearn_pipeline_list, stats_dicts = \
                                tpot_obj._preprocess_individuals(individuals)
        assert tpot_obj._pbar.total == 6


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
    features_with_nan = np.copy(training_features)
    features_with_nan[0][0] = float('nan')

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
    features_with_nan = np.copy(training_features)
    features_with_nan[0][0] = float('nan')

    tpot_obj.fit(features_with_nan, training_target)
    tpot_obj.predict(features_with_nan)


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
    features_with_nan = np.copy(training_features)
    features_with_nan[0][0] = float('nan')
    with captured_output() as (out, err):
        imputed_features = tpot_obj._impute_values(features_with_nan)
        assert_in("Imputing missing values in feature set", out.getvalue())

    assert_not_equal(imputed_features[0][0], float('nan'))


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
    assert tpot_operator_list[2].type() == "Classifier or Regressor"
    assert tpot_argument_list[1].values == [True, False]


def test_PolynomialFeatures_exception():
    """Assert that TPOT allows only one PolynomialFeatures operator in a pipeline."""
    tpot_obj = TPOTClassifier()
    tpot_obj._pbar = tqdm(total=1, disable=True)
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

    fitness_scores = tpot_obj._evaluate_individuals(pipelines, pretest_X, pretest_y)
    known_scores = [(2, 0.94000000000000006), (5000.0, -float('inf'))]
    assert np.allclose(known_scores, fitness_scores)


def test_pick_two_individuals_eligible_for_crossover():
    """Assert that pick_two_individuals_eligible_for_crossover() picks the correct pair of nodes to perform crossover with"""
    tpot_obj = TPOTClassifier()
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
    tpot_obj = TPOTClassifier()
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
    tpot_obj = TPOTClassifier()
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

    offspring1, offspring2 = tpot_obj._mate_operator(ind1, ind2)
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
    tpot_obj = TPOTClassifier()
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
    tpot_obj = TPOTClassifier()
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
            diff_prims = list(set(new_prims_list).symmetric_difference(old_prims_list))
            if len(diff_prims) > 1: # Sometimes mutation randomly replaces an operator that already in the pipelines
                assert diff_prims[0].ret == diff_prims[1].ret
        assert mut_ind[0][0].ret == Output_Array


def test_mutNodeReplacement_2():
    """Assert that mutNodeReplacement() returns the correct type of mutation node in a complex pipeline."""
    tpot_obj = TPOTClassifier()
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
            diff_prims = list(set(new_prims_list).symmetric_difference(old_prims_list))
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
    assert TPOTSelectPercentile.type() == "Preprocessor or Selector"


def test_gen():
    """Assert that TPOT's gen_grow_safe function returns a pipeline of expected structure."""
    tpot_obj = TPOTClassifier()

    pipeline = tpot_obj._gen_grow_safe(tpot_obj._pset, 1, 3)

    assert len(pipeline) > 1
    assert pipeline[0].ret == Output_Array


def test_clean_pipeline_string():
    """Assert that clean_pipeline_string correctly returns a string without parameter prefixes"""

    with_prefix = 'BernoulliNB(input_matrix, BernoulliNB__alpha=1.0, BernoulliNB__fit_prior=True)'
    without_prefix = 'BernoulliNB(input_matrix, alpha=1.0, fit_prior=True)'
    tpot_obj = TPOTClassifier()
    ind1 = creator.Individual.from_string(with_prefix, tpot_obj._pset)

    pretty_string = tpot_obj.clean_pipeline_string(ind1)
    assert pretty_string == without_prefix
