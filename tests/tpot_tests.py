# -*- coding: utf-8 -*-

"""Copyright 2015-Present Randal S. Olson.

This file is part of the TPOT library.

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
from tpot.gp_deap import mutNodeReplacement, _wrapped_cross_val_score
from tpot.metrics import balanced_accuracy
from tpot.operator_utils import TPOTOperatorClassFactory, set_sample_weight

from tpot.config.classifier import classifier_config_dict
from tpot.config.classifier_light import classifier_config_dict_light
from tpot.config.regressor_light import regressor_config_dict_light
from tpot.config.classifier_mdr import tpot_mdr_classifier_config_dict
from tpot.config.regressor_mdr import tpot_mdr_regressor_config_dict
from tpot.config.regressor_sparse import regressor_config_sparse
from tpot.config.classifier_sparse import classifier_config_sparse

import numpy as np
import inspect
import random
from multiprocessing import cpu_count

from sklearn.datasets import load_digits, load_boston
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from deap import creator
from deap.tools import ParetoFront
from nose.tools import assert_raises, assert_not_equal, assert_greater_equal

from tqdm import tqdm

# Set up the MNIST data set for testing
mnist_data = load_digits()
training_features, testing_features, training_target, testing_target = \
    train_test_split(mnist_data.data.astype(np.float64), mnist_data.target.astype(np.float64), random_state=42)

# Set up the Boston data set for testing
boston_data = load_boston()
training_features_r, testing_features_r, training_target_r, testing_target_r = \
    train_test_split(boston_data.data, boston_data.target, random_state=42)

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
    """Assert that TPOT intitializes with the correct customized scoring function."""
    tpot_obj = TPOTClassifier(scoring=balanced_accuracy)
    assert tpot_obj.scoring_function == 'balanced_accuracy'


def test_invaild_score_warning():
    """Assert that the TPOT intitializes raises a ValueError when the scoring metrics is not available in SCORERS."""
    # Mis-spelled scorer
    assert_raises(ValueError, TPOTClassifier, scoring='balanced_accuray')
    # Correctly spelled
    TPOTClassifier(scoring='balanced_accuracy')


def test_invaild_dataset_warning():
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


def test_invaild_subsample_ratio_warning():
    """Assert that the TPOT intitializes raises a ValueError when subsample ratio is not in the range (0.0, 1.0]."""
    # Invalid ratio
    assert_raises(ValueError, TPOTClassifier, subsample=0.0)
    # Valid ratio
    TPOTClassifier(subsample=0.1)


def test_invaild_mut_rate_plus_xo_rate():
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


def test_random_ind():
    """Assert that the TPOTClassifier can generate the same pipeline with same random seed."""
    tpot_obj = TPOTClassifier(random_state=43)
    pipeline1 = str(tpot_obj._toolbox.individual())
    tpot_obj = TPOTClassifier(random_state=43)
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


def test_predict_proba2():
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


def test_warm_start():
    """Assert that the TPOT warm_start flag stores the pop and pareto_front from the first run."""
    tpot_obj = TPOTClassifier(random_state=42, population_size=1, offspring_size=2, generations=1, verbosity=0, warm_start=True)
    tpot_obj.fit(training_features, training_target)

    assert tpot_obj._pop is not None
    assert tpot_obj._pareto_front is not None

    first_pop = tpot_obj._pop
    tpot_obj.random_state = 21
    tpot_obj.fit(training_features, training_target)

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
    tpot_obj.fit(training_features, training_target)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)


def test_fit2():
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


def test_fit3():
    """Assert that the TPOT fit function provides an optimized pipeline with subsample is 0.8."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        subsample=0.8,
        verbosity=0
    )
    tpot_obj.fit(training_features, training_target)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)


def test_update_top_pipeline():
    """Assert that the TPOT _update_top_pipeline updated an optimized pipeline"""
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
    tpot_obj._update_top_pipeline(training_features, training_target)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj.fitted_pipeline_ is None)


def test_update_top_pipeline_2():
    """Assert that the TPOT _update_top_pipeline raises RuntimeError when self._pareto_front is empty"""
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

    assert_raises(RuntimeError, tpot_obj._update_top_pipeline, features=training_features, target=training_target)


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
        assert np.allclose(tpot_obj.evaluated_individuals_[pipeline_string][1], mean_cv_scores)
        assert np.allclose(tpot_obj.evaluated_individuals_[pipeline_string][0], operator_count)


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


def test_imputer2():
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


def test_imputer3():
    """Assert that the TPOT _impute_values function returns a feature matrix with imputed NaN values."""
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

    imputed_features = tpot_obj._impute_values(features_with_nan)
    assert_not_equal(imputed_features[0][0], float('nan'))


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
    fitness_scores = tpot_obj._evaluate_individuals(pipelines, training_features, training_target)
    known_scores = [(2, 0.98068077235290885), (5000.0, -float('inf'))]
    assert np.allclose(known_scores, fitness_scores)


def test_mutNodeReplacement():
    """Assert that mutNodeReplacement() returns the correct type of mutation node in a fixed pipeline."""
    tpot_obj = TPOTClassifier()
    pipeline_string = (
        'KNeighborsClassifier(CombineDFs('
        'DecisionTreeClassifier(input_matrix, '
        'DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, '
        'DecisionTreeClassifier__min_samples_split=5'
        '), '
        'SelectPercentile('
        'input_matrix, '
        'SelectPercentile__percentile=20'
        ')'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')'
    )

    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    pipeline[0].ret = Output_Array
    old_ret_type_list = [node.ret for node in pipeline]
    old_prims_list = [node for node in pipeline if node.arity != 0]
    mut_ind = mutNodeReplacement(pipeline, pset=tpot_obj._pset)
    new_ret_type_list = [node.ret for node in mut_ind[0]]
    new_prims_list = [node for node in mut_ind[0] if node.arity != 0]

    if new_prims_list == old_prims_list:  # Terminal mutated
        assert new_ret_type_list == old_ret_type_list
    else:  # Primitive mutated
        diff_prims = list(set(new_prims_list).symmetric_difference(old_prims_list))
        assert diff_prims[0].ret == diff_prims[1].ret

    assert mut_ind[0][0].ret == Output_Array


def test_operator_type():
    """Assert that TPOT operators return their type, e.g. 'Classifier', 'Preprocessor'."""
    assert TPOTSelectPercentile.type() == "Preprocessor or Selector"


def test_gen():
    """Assert that TPOT's gen_grow_safe function returns a pipeline of expected structure."""
    tpot_obj = TPOTClassifier()

    pipeline = tpot_obj._gen_grow_safe(tpot_obj._pset, 1, 3)

    assert len(pipeline) > 1
    assert pipeline[0].ret == Output_Array
