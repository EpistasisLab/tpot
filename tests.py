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
from tpot.builtins import ZeroCount, StackingEstimator
from tpot.driver import positive_integer, float_range, _get_arg_parser, _print_args, main, _read_data_file
from tpot.export_utils import export_pipeline, generate_import_code, _indent, generate_pipeline_code, get_by_name
from tpot.gp_types import Output_Array
from tpot.gp_deap import mutNodeReplacement, _wrapped_cross_val_score
from tpot.metrics import balanced_accuracy

from tpot.operator_utils import TPOTOperatorClassFactory, set_sample_weight
from tpot.config.classifier import classifier_config_dict
from tpot.config.classifier_light import classifier_config_dict_light
from tpot.config.regressor_light import regressor_config_dict_light
from tpot.config.classifier_mdr import tpot_mdr_classifier_config_dict
from tpot.config.regressor_mdr import tpot_mdr_regressor_config_dict

import numpy as np
import inspect
import random
import subprocess
import sys
from multiprocessing import cpu_count

from sklearn.datasets import load_digits, load_boston
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from deap import creator
from tqdm import tqdm
from nose.tools import assert_raises, assert_equal, assert_not_equal
from unittest import TestCase
from contextlib import contextmanager
try:
    from StringIO import StringIO
except:
    from io import StringIO

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


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_driver():
    """Assert that the TPOT driver outputs normal result in mode mode."""
    batcmd = "python -m tpot.driver tests.csv -is , -target class -g 2 -p 2 -os 4 -cv 5 -s 45 -v 1"
    ret_stdout = subprocess.check_output(batcmd, shell=True)
    try:
        ret_val = float(ret_stdout.decode('UTF-8').split('\n')[-2].split(': ')[-1])
    except Exception:
        ret_val = -float('inf')
    assert ret_val > 0.0

def test_read_data_file():
    """Assert that _read_data_file raises ValueError when the targe column is missing."""
    # Mis-spelled target
    args_list = [
                'tests.csv',
                '-is', ',',
                '-target', 'clas' # typo for right target 'class'
                ]
    args = _get_arg_parser().parse_args(args_list)
    assert_raises(ValueError, _read_data_file, args=args)
    # Correctly spelled
    args_list = [
                'tests.csv',
                '-is', ',',
                '-target', 'class'
                ]
    args = _get_arg_parser().parse_args(args_list)
    input_data = _read_data_file(args)
    assert isinstance(input_data, np.recarray)


class ParserTest(TestCase):
    def setUp(self):
        self.parser = _get_arg_parser()

    def test_default_param(self):
        """Assert that the TPOT driver stores correct default values for all parameters."""
        args = self.parser.parse_args(['tests.csv'])
        self.assertEqual(args.CROSSOVER_RATE, 0.1)
        self.assertEqual(args.DISABLE_UPDATE_CHECK, False)
        self.assertEqual(args.GENERATIONS, 100)
        self.assertEqual(args.INPUT_FILE, 'tests.csv')
        self.assertEqual(args.INPUT_SEPARATOR, '\t')
        self.assertEqual(args.MAX_EVAL_MINS, 5)
        self.assertEqual(args.MUTATION_RATE, 0.9)
        self.assertEqual(args.NUM_CV_FOLDS, 5)
        self.assertEqual(args.NUM_JOBS, 1)
        self.assertEqual(args.OFFSPRING_SIZE, None)
        self.assertEqual(args.OUTPUT_FILE, '')
        self.assertEqual(args.POPULATION_SIZE, 100)
        self.assertEqual(args.RANDOM_STATE, None)
        self.assertEqual(args.SUBSAMPLE, 1.0)
        self.assertEqual(args.SCORING_FN, None)
        self.assertEqual(args.TARGET_NAME, 'class')
        self.assertEqual(args.TPOT_MODE, 'classification')
        self.assertEqual(args.VERBOSITY, 1)


    def test_print_args(self):
        """Assert that _print_args prints correct values for all parameters."""
        args = self.parser.parse_args(['tests.csv'])
        with captured_output() as (out, err):
            _print_args(args)
        output = out.getvalue()
        expected_output = """
TPOT settings:
CONFIG_FILE\t=\tNone
CROSSOVER_RATE\t=\t0.1
GENERATIONS\t=\t100
INPUT_FILE\t=\ttests.csv
INPUT_SEPARATOR\t=\t\t
MAX_EVAL_MINS\t=\t5
MAX_TIME_MINS\t=\tNone
MUTATION_RATE\t=\t0.9
NUM_CV_FOLDS\t=\t5
NUM_JOBS\t=\t1
OFFSPRING_SIZE\t=\t100
OUTPUT_FILE\t=\t
POPULATION_SIZE\t=\t100
RANDOM_STATE\t=\tNone
SCORING_FN\t=\taccuracy
SUBSAMPLE\t=\t1.0
TARGET_NAME\t=\tclass
TPOT_MODE\t=\tclassification
VERBOSITY\t=\t1

"""

        self.assertEqual(_sort_lines(expected_output), _sort_lines(output))

def _sort_lines(text):
    return '\n'.join(sorted(text.split('\n')))

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
    """Assert that the TPOT init stores current number of processes"""
    tpot_obj = TPOTClassifier(n_jobs=2)
    assert tpot_obj.n_jobs == 2

    tpot_obj = TPOTClassifier(n_jobs=-1)
    assert tpot_obj.n_jobs == cpu_count()


def test_timeout():
    """Assert that _wrapped_cross_val_score return Timeout in a time limit"""
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
                                            max_eval_time_mins=0.02,
                                            groups=None)
    assert return_value == "Timeout"


def test_balanced_accuracy():
    """Assert that the balanced_accuracy in TPOT returns correct accuracy."""
    y_true = np.array([1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4])
    y_pred1 = np.array([1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4])
    y_pred2 = np.array([3,3,3,3,3,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4])
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

    tpot_obj = TPOTRegressor(config_dict='TPOT light')
    assert tpot_obj.config_dict == regressor_config_dict_light

    tpot_obj = TPOTRegressor(config_dict='TPOT MDR')
    assert tpot_obj.config_dict == tpot_mdr_regressor_config_dict


def test_conf_dict_2():
    """Assert that TPOT uses a custom dictionary of operators when config_dict is Python dictionary."""
    tpot_obj = TPOTClassifier(config_dict=tpot_mdr_classifier_config_dict)
    assert tpot_obj.config_dict == tpot_mdr_classifier_config_dict


def test_conf_dict_3():
    """Assert that TPOT uses a custom dictionary of operators when config_dict is the path of Python dictionary."""
    tpot_obj = TPOTRegressor(config_dict='test_config.py')
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


def test_random_ind_2():
    """Assert that the TPOTClassifier can generate the same pipeline export with random seed of 39."""
    tpot_obj = TPOTClassifier(random_state=39)
    tpot_obj._pbar = tqdm(total=1, disable=True)
    pipeline = tpot_obj._toolbox.individual()
    expected_code = """import numpy as np

from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \\
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=65),
    DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_leaf=4, min_samples_split=18)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""

    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)


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
    """Assert that TPOT properly handles the group parameter when using GroupKFold"""
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
        cv = GroupKFold(n_splits=2),
    )
    tpot_obj.fit(training_features, training_target, groups=groups)
    assert tpot_obj.score(testing_features, testing_target) >= 0.97


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
    """Assert that the TPOT fit function provides an optimized pipeline with subsample is 0.8"""
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


def test_evaluated_individuals_():
    """Assert that evaluated_individuals_ stores corrent pipelines and their CV scores."""
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
        except:
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
        except:
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


def check_export(op, tpot_obj):
    """Assert that a TPOT operator exports as expected."""
    prng = np.random.RandomState(42)
    np.random.seed(42)

    args = []
    for type_ in op.parameter_types()[0][1:]:
        args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)
    export_string = op.export(*args)

    assert export_string.startswith(op.__name__ + "(") and export_string.endswith(")")


def test_operators():
    """Assert that the TPOT operators match the output of their sklearn counterparts."""
    tpot_obj = TPOTClassifier(random_state=42)
    for op in tpot_obj.operators:
        check_export.description = ("Assert that the TPOT {} operator exports "
                                    "as expected".format(op.__name__))
        yield check_export, op, tpot_obj


def test_export():
    """Assert that TPOT's export function throws a RuntimeError when no optimized pipeline exists."""
    tpot_obj = TPOTClassifier()
    assert_raises(RuntimeError, tpot_obj.export, "test_export.py")


def test_generate_pipeline_code():
    """Assert that generate_pipeline_code() returns the correct code given a specific pipeline."""
    tpot_obj = TPOTClassifier()
    pipeline = [
        'KNeighborsClassifier',
        [
            'CombineDFs',
            [
                'GradientBoostingClassifier',
                'input_matrix',
                38.0,
                5,
                5,
                5,
                0.05,
                0.5],
            [
                'GaussianNB',
                [
                    'ZeroCount',
                    'input_matrix'
                ]
            ]
        ],
        18,
        'uniform',
        2
    ]

    expected_code = """make_pipeline(
    make_union(
        StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=38.0, max_depth=5, max_features=5, min_samples_leaf=5, min_samples_split=0.05, n_estimators=0.5)),
        StackingEstimator(estimator=make_pipeline(
            ZeroCount(),
            GaussianNB()
        ))
    ),
    KNeighborsClassifier(n_neighbors=18, p="uniform", weights=2)
)"""
    assert expected_code == generate_pipeline_code(pipeline, tpot_obj.operators)


def test_generate_import_code():
    """Assert that generate_import_code() returns the correct set of dependancies for a given pipeline."""
    tpot_obj = TPOTClassifier()
    pipeline = creator.Individual.from_string('GaussianNB(RobustScaler(input_matrix))', tpot_obj._pset)

    expected_code = """import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
"""
    assert expected_code == generate_import_code(pipeline, tpot_obj.operators)


def test_generate_import_code_2():
    """Assert that generate_import_code() returns the correct set of dependancies and dependancies are importable."""
    tpot_obj = TPOTClassifier()
    pipeline_string = (
        'KNeighborsClassifier(CombineDFs('
        'DecisionTreeClassifier(input_matrix, DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8,DecisionTreeClassifier__min_samples_leaf=5,'
        'DecisionTreeClassifier__min_samples_split=5), ZeroCount(input_matrix))'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1,KNeighborsClassifier__weights=uniform'
    )

    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)

    import_code = generate_import_code(pipeline, tpot_obj.operators)

    expected_code = """import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator, ZeroCount
"""
    exec(import_code) # should not raise error
    assert expected_code == import_code


def test_PolynomialFeatures_exception():
    """Assert that TPOT allows only one PolynomialFeatures operator in a pipeline"""
    tpot_obj = TPOTClassifier()
    tpot_obj._pbar = tqdm(total=1, disable=True)
    # pipeline with one PolynomialFeatures operator
    pipeline_string_1 = ('LogisticRegression(PolynomialFeatures'
    '(input_matrix, PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=DEFAULT, '
    'PolynomialFeatures__interaction_only=False), LogisticRegression__C=10.0, '
    'LogisticRegression__dual=DEFAULT, LogisticRegression__penalty=DEFAULT)')

    # pipeline with two PolynomialFeatures operator
    pipeline_string_2 = ('LogisticRegression(PolynomialFeatures'
    '(PolynomialFeatures(input_matrix, PolynomialFeatures__degree=2, '
    'PolynomialFeatures__include_bias=DEFAULT, PolynomialFeatures__interaction_only=False), '
    'PolynomialFeatures__degree=2, PolynomialFeatures__include_bias=DEFAULT, '
    'PolynomialFeatures__interaction_only=False), LogisticRegression__C=10.0, '
    'LogisticRegression__dual=DEFAULT, LogisticRegression__penalty=DEFAULT)')

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


def test_export_pipeline():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed pipeline."""
    tpot_obj = TPOTClassifier()
    pipeline_string = (
        'KNeighborsClassifier(CombineDFs('
        'DecisionTreeClassifier(input_matrix, DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8,DecisionTreeClassifier__min_samples_leaf=5,'
        'DecisionTreeClassifier__min_samples_split=5),SelectPercentile(input_matrix, SelectPercentile__percentile=20))'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1,KNeighborsClassifier__weights=uniform'
    )

    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    expected_code = """import numpy as np

from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \\
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=5, min_samples_split=5)),
        SelectPercentile(score_func=f_classif, percentile=20)
    ),
    KNeighborsClassifier(n_neighbors=10, p=1, weights="uniform")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)


def test_export_pipeline_2():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed simple pipeline (only one classifier)."""
    tpot_obj = TPOTClassifier()
    pipeline_string = (
        'KNeighborsClassifier('
        'input_matrix, '
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')'
    )
    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    expected_code = """import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \\
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = KNeighborsClassifier(n_neighbors=10, p=1, weights="uniform")

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)


def test_export_pipeline_3():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed simple pipeline with a preprocessor."""
    tpot_obj = TPOTClassifier()
    pipeline_string = (
        'DecisionTreeClassifier(SelectPercentile(input_matrix, SelectPercentile__percentile=20),'
        'DecisionTreeClassifier__criterion=gini, DecisionTreeClassifier__max_depth=8,'
        'DecisionTreeClassifier__min_samples_leaf=5, DecisionTreeClassifier__min_samples_split=5)'
    )
    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)

    expected_code = """import numpy as np

from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \\
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=20),
    DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=5, min_samples_split=5)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)

def test_export_pipeline_4():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed simple pipeline with input_matrix in CombineDFs."""
    tpot_obj = TPOTClassifier()
    pipeline_string = (
        'KNeighborsClassifier(CombineDFs('
        'DecisionTreeClassifier(input_matrix, DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8,DecisionTreeClassifier__min_samples_leaf=5,'
        'DecisionTreeClassifier__min_samples_split=5),input_matrix)'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1,KNeighborsClassifier__weights=uniform'
    )

    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    expected_code = """import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \\
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=5, min_samples_split=5)),
        FunctionTransformer(copy)
    ),
    KNeighborsClassifier(n_neighbors=10, p=1, weights="uniform")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)

def test_operator_export():
    """Assert that a TPOT operator can export properly with a function as a parameter to a classifier."""
    export_string = TPOTSelectPercentile.export(5)
    assert export_string == "SelectPercentile(score_func=f_classif, percentile=5)"


def test_indent():
    """Assert that indenting a multiline string by 4 spaces prepends 4 spaces before each new line."""
    multiline_string = """test
test1
test2
test3"""

    indented_multiline_string = """    test
    test1
    test2
    test3"""

    assert indented_multiline_string == _indent(multiline_string, 4)


def test_operator_type():
    """Assert that TPOT operators return their type, e.g. 'Classifier', 'Preprocessor'."""
    assert TPOTSelectPercentile.type() == "Preprocessor or Selector"


def test_get_by_name():
    """Assert that the Operator class returns operators by name appropriately."""
    tpot_obj = TPOTClassifier()
    assert get_by_name("SelectPercentile", tpot_obj.operators).__class__ == TPOTSelectPercentile.__class__


def test_gen():
    """Assert that TPOT's gen_grow_safe function returns a pipeline of expected structure."""
    tpot_obj = TPOTClassifier()

    pipeline = tpot_obj._gen_grow_safe(tpot_obj._pset, 1, 3)

    assert len(pipeline) > 1
    assert pipeline[0].ret == Output_Array


def test_positive_integer():
    """Assert that the TPOT CLI interface's integer parsing throws an exception when n < 0."""
    assert_raises(Exception, positive_integer, '-1')


def test_positive_integer_2():
    """Assert that the TPOT CLI interface's integer parsing returns the integer value of a string encoded integer when n > 0."""
    assert 1 == positive_integer('1')


def test_positive_integer_3():
    """Assert that the TPOT CLI interface's integer parsing throws an exception when n is not an integer."""
    assert_raises(Exception, positive_integer, 'foobar')


def test_float_range():
    """Assert that the TPOT CLI interface's float range returns a float with input is in 0. - 1.0."""
    assert 0.5 == float_range('0.5')


def test_float_range_2():
    """Assert that the TPOT CLI interface's float range throws an exception when input it out of range."""
    assert_raises(Exception, float_range, '2.0')


def test_float_range_3():
    """Assert that the TPOT CLI interface's float range throws an exception when input is not a float."""
    assert_raises(Exception, float_range, 'foobar')


def test_StackingEstimator_1():
    """Assert that the StackingEstimator returns transformed X with synthetic features in classification."""
    clf = RandomForestClassifier(random_state=42)
    stack_clf = StackingEstimator(estimator=RandomForestClassifier(random_state=42))
    # fit
    clf.fit(training_features, training_target)
    stack_clf.fit(training_features, training_target)
    # get transformd X
    X_clf_transformed = stack_clf.transform(training_features)

    assert np.allclose(clf.predict(training_features), X_clf_transformed[:,0])
    assert np.allclose(clf.predict_proba(training_features), X_clf_transformed[:,1:1+len(np.unique(training_target))])


def test_StackingEstimator_2():
    """Assert that the StackingEstimator returns transformed X with a synthetic feature in regression."""
    reg = RandomForestRegressor(random_state=42)
    stack_reg = StackingEstimator(estimator=RandomForestRegressor(random_state=42))
    # fit
    reg.fit(training_features_r, training_target_r)
    stack_reg.fit(training_features_r, training_target_r)
    # get transformd X
    X_reg_transformed = stack_reg.transform(training_features_r)

    assert np.allclose(reg.predict(training_features_r), X_reg_transformed[:,0])


def test_StackingEstimator_3():
    """Assert that the StackingEstimator worked as expected in scikit-learn pipeline in classification"""
    stack_clf = StackingEstimator(estimator=RandomForestClassifier(random_state=42))
    meta_clf = LogisticRegression()
    sklearn_pipeline = make_pipeline(stack_clf, meta_clf)
    # fit in pipeline
    sklearn_pipeline.fit(training_features, training_target)
    # fit step by step
    stack_clf.fit(training_features, training_target)
    X_clf_transformed = stack_clf.transform(training_features)
    meta_clf.fit(X_clf_transformed, training_target)
    # scoring
    score = meta_clf.score(X_clf_transformed, training_target)
    pipeline_score = sklearn_pipeline.score(training_features, training_target)
    assert np.allclose(score, pipeline_score)

    # test cv score
    cv_score = np.mean(cross_val_score(sklearn_pipeline, training_features, training_target, cv=3, scoring='accuracy'))

    known_cv_score = 0.947282375315

    assert np.allclose(known_cv_score, cv_score)

def test_StackingEstimator_4():
    """Assert that the StackingEstimator worked as expected in scikit-learn pipeline in regression"""
    stack_reg = StackingEstimator(estimator=RandomForestRegressor(random_state=42))
    meta_reg = Lasso(random_state=42)
    sklearn_pipeline = make_pipeline(stack_reg, meta_reg)
    # fit in pipeline
    sklearn_pipeline.fit(training_features_r, training_target_r)
    # fit step by step
    stack_reg.fit(training_features_r, training_target_r)
    X_reg_transformed = stack_reg.transform(training_features_r)
    meta_reg.fit(X_reg_transformed, training_target_r)
    # scoring
    score = meta_reg.score(X_reg_transformed, training_target_r)
    pipeline_score = sklearn_pipeline.score(training_features_r, training_target_r)
    assert np.allclose(score, pipeline_score)

    # test cv score
    cv_score = np.mean(cross_val_score(sklearn_pipeline, training_features_r, training_target_r, cv=3, scoring='r2'))
    known_cv_score = 0.795877470354

    assert np.allclose(known_cv_score, cv_score)


def test_ZeroCount():
    """Assert that ZeroCount operator returns correct transformed X"""
    X = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19], [0, 1, 3, 4, 5], [5, 0, 0, 0, 0]])
    op = ZeroCount()
    X_transformed = op.transform(X)
    zero_col = np.array([3, 2, 1, 4])
    non_zero = np.array([2, 3, 4, 1])

    assert np.allclose(zero_col, X_transformed[:, 0])
    assert np.allclose(non_zero, X_transformed[:, 1])
