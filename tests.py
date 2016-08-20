# -*- coding: utf-8 -*-

"""
TPOT Unit Tests
"""

from tpot import TPOT
from tpot.tpot import positive_integer, float_range
from tpot.export_utils import export_pipeline, generate_import_code, _indent, generate_pipeline_code
from tpot.decorators import _gp_new_generation
from tpot.gp_types import Output_DF

from tpot.operators import Operator
from tpot.operators.selectors import TPOTSelectKBest

import numpy as np
import inspect
import random

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

from deap import creator
from tqdm import tqdm

# Set up the MNIST data set for testing
mnist_data = load_digits()
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(mnist_data.data.astype(np.float64), mnist_data.target.astype(np.float64), random_state=42)

np.random.seed(42)
random.seed(42)


def test_init():
    """Assert that the TPOT instantiator stores the TPOT variables properly"""

    def dummy_scoring_func(foo, bar):
        return

    tpot_obj = TPOT(population_size=500, generations=1000, scoring_function=dummy_scoring_func,
                    mutation_rate=0.05, crossover_rate=0.9, verbosity=1, random_state=42,
                    disable_update_check=True)

    assert tpot_obj.population_size == 500
    assert tpot_obj.generations == 1000
    assert tpot_obj.mutation_rate == 0.05
    assert tpot_obj.crossover_rate == 0.9
    assert tpot_obj.verbosity == 1
    assert tpot_obj._optimized_pipeline is None
    assert tpot_obj._fitted_pipeline is None
    assert tpot_obj.scoring_function == dummy_scoring_func
    assert tpot_obj._pset


def test_get_params():
    """Assert that get_params returns the exact dictionary of parameters used by TPOT"""
    kwargs = {
        'population_size': 500,
        'generations': 1000,
        'verbosity': 1
    }

    tpot_obj = TPOT(**kwargs)

    # Get default parameters of TPOT and merge with our specified parameters
    initializer = inspect.getargspec(TPOT.__init__)
    default_kwargs = dict(zip(initializer.args[1:], initializer.defaults))
    default_kwargs.update(kwargs)

    assert tpot_obj.get_params() == default_kwargs


def test_score():
    """Assert that the TPOT score function raises a ValueError when no optimized pipeline exists"""

    tpot_obj = TPOT()

    try:
        tpot_obj.score(testing_features, testing_classes)
        assert False  # Should be unreachable
    except ValueError:
        pass


def test_score_2():
    """Assert that the TPOT score function outputs a known score for a fixed pipeline"""

    tpot_obj = TPOT()
    tpot_obj.pbar = tqdm(total=1, disable=True)
    known_score = 0.986318199045  # Assumes use of the TPOT balanced_accuracy function

    # Reify pipeline with known score
    tpot_obj._optimized_pipeline = creator.Individual.\
        from_string('RandomForestClassifier(input_matrix)', tpot_obj._pset)
    tpot_obj._fitted_pipeline = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj._fitted_pipeline.fit(training_features, training_classes)

    # Get score from TPOT
    score = tpot_obj.score(testing_features, testing_classes)

    # http://stackoverflow.com/questions/5595425/
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    assert isclose(known_score, score)


def test_predict():
    """Assert that the TPOT predict function raises a ValueError when no optimized pipeline exists"""

    tpot_obj = TPOT()

    try:
        tpot_obj.predict(testing_features)
        assert False  # Should be unreachable
    except ValueError:
        pass


def test_predict_2():
    """Assert that the TPOT predict function returns a numpy matrix of shape (num_testing_rows,)"""

    tpot_obj = TPOT()
    tpot_obj._optimized_pipeline = creator.Individual.\
        from_string('DecisionTreeClassifier(input_matrix)', tpot_obj._pset)
    tpot_obj._fitted_pipeline = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj._fitted_pipeline.fit(training_features, training_classes)

    result = tpot_obj.predict(testing_features)

    assert result.shape == (testing_features.shape[0],)


def test_fit():
    """Assert that the TPOT fit function provides an optimized pipeline"""
    tpot_obj = TPOT(random_state=42, population_size=1, generations=1, verbosity=0)
    tpot_obj.fit(training_features, training_classes)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert tpot_obj.gp_generation == 0


def test_gp_new_generation():
    """Assert that the gp_generation count gets incremented when _gp_new_generation is called"""
    tpot_obj = TPOT()
    tpot_obj.pbar = tqdm(total=1, disable=True)

    assert(tpot_obj.gp_generation == 0)

    # Since _gp_new_generation is a decorator, and we dont want to run a full
    # fit(), decorate a dummy function and then call the dummy function.
    @_gp_new_generation
    def dummy_function(self, foo):
        pass

    dummy_function(tpot_obj, None)

    assert(tpot_obj.gp_generation == 1)


def check_export(op):
    """Assert that a TPOT operator exports as expected"""
    tpot_obj = TPOT(random_state=42)

    prng = np.random.RandomState(42)
    np.random.seed(42)

    args = []
    for type_ in op.parameter_types()[0][1:]:
        args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)

    export_string = op.export(*args)

    assert export_string.startswith(op.__name__ + "(") and export_string.endswith(")")


def test_operators():
    """Assert that the TPOT operators match the output of their sklearn counterparts"""
    for op in Operator.inheritors():
        check_export.description = ("Assert that the TPOT {} operator exports "
                                    "as expected".format(op.__name__))
        yield check_export, op


def test_export():
    """Assert that TPOT's export function throws a ValueError when no optimized pipeline exists"""
    tpot_obj = TPOT()

    try:
        tpot_obj.export("test_export.py")
        assert False  # Should be unreachable
    except ValueError:
        pass


def test_generate_pipeline_code():
    """Assert that generate_pipeline_code() returns the correct code given a specific pipeline"""
    pipeline = ['KNeighborsClassifier',
        ['CombineDFs',
            ['GradientBoostingClassifier',
                'input_matrix',
                38.0,
                0.87,
                0.5],
            ['GaussianNB',
                ['ZeroCount',
                    'input_matrix']]],
        18,
        33]

    expected_code = """make_pipeline(
    make_union(
        make_union(VotingClassifier(estimators=[('branch',
            GradientBoostingClassifier(learning_rate=1.0, max_features=1.0, min_weight_fraction_leaf=0.5, n_estimators=500)
        )]), FunctionTransformer(lambda X: X)),
        make_union(VotingClassifier(estimators=[('branch',
            make_pipeline(
                ZeroCount(),
                GaussianNB()
            )
        )]), FunctionTransformer(lambda X: X))
    ),
    KNeighborsClassifier(n_neighbors=5, weights="distance")
)"""

    assert expected_code == generate_pipeline_code(pipeline)


def test_generate_import_code():
    """Assert that generate_import_code() returns the correct set of dependancies for a given pipeline"""
    tpot_obj = TPOT()
    pipeline = creator.Individual.\
        from_string('DecisionTreeClassifier(SelectKBest(input_matrix, 7), 0.5)', tpot_obj._pset)

    expected_code = """import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \\
    train_test_split(features, tpot_data['class'], random_state=42)
"""

    assert expected_code == generate_import_code(pipeline)


def test_export_pipeline():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed pipeline"""
    tpot_obj = TPOT()
    pipeline = creator.Individual.\
        from_string("KNeighborsClassifier(CombineDFs(GradientBoostingClassifier(input_matrix, 38.0, 0.87, 0.5), RFE(input_matrix, 0.17999999999999999)), 18, 33)", tpot_obj._pset)

    expected_code = """import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \\
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    make_union(
        make_union(VotingClassifier(estimators=[('branch',
            GradientBoostingClassifier(learning_rate=1.0, max_features=1.0, min_weight_fraction_leaf=0.5, n_estimators=500)
        )]), FunctionTransformer(lambda X: X)),
        RFE(estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
          max_iter=-1, probability=False, random_state=42, shrinking=True,
          tol=0.001, verbose=False), step=0.18)
    ),
    KNeighborsClassifier(n_neighbors=5, weights="distance")
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
"""

    assert expected_code == export_pipeline(pipeline)


def test_operator_export():
    """Assert that a TPOT operator can export properly with a function as a parameter to a classifier"""
    export_string = TPOTSelectKBest().export(5)
    assert export_string == "SelectKBest(k=5, score_func=f_classif)"


def test_indent():
    """Assert that indenting a multiline string by 4 spaces prepends 4 spaces before each new line"""

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
    """Assert that TPOT operators return their type, e.g. "Classifier", "Preprocessor" """
    assert TPOTSelectKBest().type == "Selector"


def test_get_by_name():
    """Assert that the Operator class returns operators by name appropriately"""
    assert Operator.get_by_name("SelectKBest").__class__ == TPOTSelectKBest


def test_gen():
    """Assert that TPOT's gen_grow_safe function returns a pipeline of expected structure"""
    tpot_obj = TPOT()

    pipeline = tpot_obj._gen_grow_safe(tpot_obj._pset, 1, 3)

    assert len(pipeline) > 1
    assert pipeline[0].ret == Output_DF


def test_positive_integer():
    """Assert that the TPOT CLI interface's integer parsing throws an exception when n < 0"""
    try:
        positive_integer('-1')
        assert False  # Should be unreachable
    except Exception:
        pass


def test_positive_integer_2():
    """Assert that the TPOT CLI interface's integer parsing returns the integer value of a string encoded integer when n > 0"""
    assert 1 == positive_integer('1')


def test_positive_integer_3():
    """Assert that the TPOT CLI interface's integer parsing throws an exception when n is not an integer"""
    try:
        positive_integer('foobar')
        assert False  # Should be unreachable
    except Exception:
        pass


def test_float_range():
    """Assert that the TPOT CLI interface's float range returns a float with input is in 0. - 1.0"""
    assert 0.5 == float_range('0.5')


def test_float_range_2():
    """Assert that the TPOT CLI interface's float range throws an exception when input it out of range"""
    try:
        float_range('2.0')
        assert False  # Should be unreachable
    except Exception:
        pass


def test_float_range_3():
    """Assert that the TPOT CLI interface's float range throws an exception when input is not a float"""
    try:
        float_range('foobar')
        assert False  # Should be unreachable
    except Exception:
        pass
