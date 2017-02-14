# -*- coding: utf-8 -*-

"""
TPOT Unit Tests
"""

from tpot import TPOTClassifier, TPOTRegressor
from tpot.base import TPOTBase
from tpot.driver import positive_integer, float_range
from tpot.export_utils import export_pipeline, generate_import_code, _indent, generate_pipeline_code, get_by_name
from tpot.gp_types import Output_DF
from tpot.gp_deap import mutNodeReplacement

from tpot.operator_utils import TPOTOperatorClassFactory
from tpot.config_classifier import classifier_config_dict


import numpy as np
import inspect
import random
from datetime import datetime

from sklearn.datasets import load_digits, load_boston
from sklearn.model_selection import train_test_split

from deap import creator
from tqdm import tqdm

# Set up the MNIST data set for testing
mnist_data = load_digits()
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(mnist_data.data.astype(np.float64), mnist_data.target.astype(np.float64), random_state=42)

# Set up the Boston data set for testing
boston_data = load_boston()
training_features_r, testing_features_r, training_classes_r, testing_classes_r = \
    train_test_split(boston_data.data, boston_data.target, random_state=42)

np.random.seed(42)
random.seed(42)

test_operator_key = 'sklearn.feature_selection.SelectKBest'
TPOTSelectKBest,TPOTSelectKBest_args = TPOTOperatorClassFactory(test_operator_key,
                                            classifier_config_dict[test_operator_key])


def test_init_custom_parameters():
    """Assert that the TPOT instantiator stores the TPOT variables properly"""

    tpot_obj = TPOTClassifier(population_size=500, generations=1000, offspring_size=2000,
                    mutation_rate=0.05, crossover_rate=0.9,
                    scoring='accuracy', cv=10,
                    verbosity=1, random_state=42,
                    disable_update_check=True, warm_start=True)

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
    assert tpot_obj._fitted_pipeline is None
    assert not (tpot_obj._pset is None)
    assert not (tpot_obj._toolbox is None)


def test_init_default_scoring():
    """Assert that TPOT intitializes with the correct default scoring function"""

    tpot_obj = TPOTRegressor()
    assert tpot_obj.scoring_function == 'neg_mean_squared_error'


def test_init_max_time_mins():
    """Assert that the TPOT init stores max run time and sets generations to 1000000"""

    tpot_obj = TPOTClassifier(max_time_mins=30, generations=1000)

    assert tpot_obj.generations == 1000000
    assert tpot_obj.max_time_mins == 30


def test_get_params():
    """Assert that get_params returns the exact dictionary of parameters used by TPOT"""

    kwargs = {
        'population_size': 500,
        'generations': 1000,
        'offspring_size': 2000,
        'verbosity': 1,
        'operator_dict': classifier_config_dict
    }

    tpot_obj = TPOTClassifier(**kwargs)

    # Get default parameters of TPOT and merge with our specified parameters
    initializer = inspect.getargspec(TPOTBase.__init__)
    default_kwargs = dict(zip(initializer.args[1:], initializer.defaults))
    default_kwargs.update(kwargs)

    assert tpot_obj.get_params() == default_kwargs


def test_set_params():
    """Assert that set_params returns a reference to the TPOT instance"""

    tpot_obj = TPOTClassifier()
    assert tpot_obj.set_params() is tpot_obj


def test_set_params_2():
    """Assert that set_params updates TPOT's instance variables"""
    tpot_obj = TPOTClassifier(generations=2)
    tpot_obj.set_params(generations=3)

    assert tpot_obj.generations == 3

def test_random_ind():
    """Assert that the TPOTClassifier can generate the same pipeline with same random seed"""
    tpot_obj = TPOTClassifier(random_state=43)
    pipeline1 = str(tpot_obj._toolbox.individual())
    tpot_obj = TPOTClassifier(random_state=43)
    pipeline2 = str(tpot_obj._toolbox.individual())
    assert pipeline1 == pipeline2

def test_random_ind_2():
    """Assert that the TPOTClassifier can generate the same pipeline export with random seed of 45"""

    tpot_obj = TPOTClassifier(random_state=45)
    tpot_obj._pbar = tqdm(total=1, disable=True)
    pipeline = tpot_obj._toolbox.individual()
    expected_code = """import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.build_in_operators import ZeroCount

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \\
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    ZeroCount(),
    LogisticRegression(C=0.0001, dual=False, penalty="l2")
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators)

def test_score():
    """Assert that the TPOT score function raises a ValueError when no optimized pipeline exists"""

    tpot_obj = TPOTClassifier()

    try:
        tpot_obj.score(testing_features, testing_classes)
        assert False  # Should be unreachable
    except ValueError:
        pass


def test_score_2():
    """Assert that the TPOTClassifier score function outputs a known score for a ramdom pipeline"""

    tpot_obj = TPOTClassifier(random_state=43)
    tpot_obj._pbar = tqdm(total=1, disable=True)
    known_score = 0.96710588996037627  # Assumes use of the TPOT balanced_accuracy function

    # Reify pipeline with known score
    tpot_obj._optimized_pipeline = tpot_obj._toolbox.individual()
    tpot_obj._fitted_pipeline = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj._fitted_pipeline.fit(training_features, training_classes)

    # Get score from TPOT
    score = tpot_obj.score(testing_features, testing_classes)

    # http://stackoverflow.com/questions/5595425/
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    assert isclose(known_score, score)

def test_score_3():
    """Assert that the TPOTRegressor score function outputs a known score for a random pipeline"""

    tpot_obj = TPOTRegressor(scoring='neg_mean_squared_error', random_state=53)
    tpot_obj._pbar = tqdm(total=1, disable=True)
    known_score = 15.724128278216726 # Assumes use of mse

    # Reify pipeline with known score
    tpot_obj._optimized_pipeline = tpot_obj._toolbox.individual()
    tpot_obj._fitted_pipeline = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj._fitted_pipeline.fit(training_features_r, training_classes_r)

    # Get score from TPOT
    score = tpot_obj.score(testing_features_r, testing_classes_r)

    # http://stackoverflow.com/questions/5595425/
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    assert isclose(known_score, score)



def test_predict():
    """Assert that the TPOT predict function raises a ValueError when no optimized pipeline exists"""

    tpot_obj = TPOTClassifier()

    try:
        tpot_obj.predict(testing_features)
        assert False  # Should be unreachable
    except ValueError:
        pass


def test_predict_2():
    """Assert that the TPOT predict function returns a numpy matrix of shape (num_testing_rows,)"""

    tpot_obj = TPOTClassifier(random_state=49)
    tpot_obj._optimized_pipeline = tpot_obj._toolbox.individual()
    tpot_obj._fitted_pipeline = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj._fitted_pipeline.fit(training_features, training_classes)

    result = tpot_obj.predict(testing_features)

    assert result.shape == (testing_features.shape[0],)


def test_predict_proba():
    """Assert that the TPOT predict_proba function returns a numpy matrix of shape (num_testing_rows, num_testing_classes)"""

    tpot_obj = TPOTClassifier(random_state=51)
    tpot_obj._optimized_pipeline = tpot_obj._toolbox.individual()
    tpot_obj._fitted_pipeline = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj._fitted_pipeline.fit(training_features, training_classes)

    result = tpot_obj.predict_proba(testing_features)
    num_labels = np.amax(testing_classes) + 1

    assert result.shape == (testing_features.shape[0], num_labels)


def test_predict_proba2():
    """Assert that the TPOT predict_proba function returns a numpy matrix filled with probabilities (float)"""

    tpot_obj = TPOTClassifier(random_state=53)
    tpot_obj._optimized_pipeline = tpot_obj._toolbox.individual()
    tpot_obj._fitted_pipeline = tpot_obj._toolbox.compile(expr=tpot_obj._optimized_pipeline)
    tpot_obj._fitted_pipeline.fit(training_features, training_classes)

    result = tpot_obj.predict_proba(testing_features)

    rows = result.shape[0]
    columns = result.shape[1]

    try:
        for i in range(rows):
            for j in range(columns):
                float_range(result[i][j])
        assert True
    except Exception:
        assert False

def test_warm_start():
    """Assert that the TPOT warm_start flag stores the pop and pareto_front from the first run"""
    tpot_obj = TPOTClassifier(random_state=42, population_size=2, offspring_size=4, generations=1, verbosity=0, warm_start=True)
    tpot_obj.fit(training_features, training_classes)

    assert tpot_obj._pop != None
    assert tpot_obj._pareto_front != None

    first_pop = tpot_obj._pop
    first_pareto_front = tpot_obj._pareto_front

    tpot_obj.random_state = 21
    tpot_obj.fit(training_features, training_classes)

    assert tpot_obj._pop == first_pop


def test_fit():
    """Assert that the TPOT fit function provides an optimized pipeline"""
    tpot_obj = TPOTClassifier(random_state=42, population_size=2, offspring_size=4, generations=1, verbosity=0)
    tpot_obj.fit(training_features, training_classes)

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert not (tpot_obj._start_datetime is None)


def testTPOTOperatorClassFactory():
    """Assert that the TPOT operators class factory"""
    test_operator_dict = {
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
    for key in sorted(test_operator_dict.keys()):
        op,args = TPOTOperatorClassFactory(key,test_operator_dict[key])
        tpot_operator_list.append(op)
        tpot_argument_list += args
    assert len(tpot_operator_list) == 3
    assert len(tpot_argument_list) == 9
    assert tpot_operator_list[0].root == True
    assert tpot_operator_list[1].root == False
    assert tpot_operator_list[2].type() == "Classifier or Regressor"
    assert tpot_argument_list[1].values == [True, False]


def check_export(op, tpot_obj):
    """Assert that a TPOT operator exports as expected"""

    prng = np.random.RandomState(42)
    np.random.seed(42)

    args = []
    for type_ in op.parameter_types()[0][1:]:
        args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)
    export_string = op.export(*args)

    assert export_string.startswith(op.__name__ + "(") and export_string.endswith(")")


def test_operators():
    """Assert that the TPOT operators match the output of their sklearn counterparts"""
    tpot_obj = TPOTClassifier(random_state=42)
    for op in tpot_obj.operators:
        check_export.description = ("Assert that the TPOT {} operator exports "
                                    "as expected".format(op.__name__))
        yield check_export, op, tpot_obj


def test_export():
    """Assert that TPOT's export function throws a ValueError when no optimized pipeline exists"""
    tpot_obj = TPOTClassifier()

    try:
        tpot_obj.export("test_export.py")
        assert False  # Should be unreachable
    except ValueError:
        pass


def test_generate_pipeline_code():
    """Assert that generate_pipeline_code() returns the correct code given a specific pipeline"""
    tpot_obj = TPOTClassifier()
    pipeline = ['KNeighborsClassifier',
        ['CombineDFs',
            ['GradientBoostingClassifier',
                'input_matrix',
                38.0,
                5,
                5,
                5,
                0.05,
                0.5],
            ['GaussianNB',
                ['ZeroCount',
                    'input_matrix']]],
        18,
        'uniform',
        2]

    expected_code = """make_pipeline(
    make_union(
        make_union(VotingClassifier([('branch',
            GradientBoostingClassifier(learning_rate=38.0, max_depth=5, max_features=5, min_samples_leaf=5, min_samples_split=0.05, subsample=0.5)
        )]), FunctionTransformer(lambda X: X)),
        make_union(VotingClassifier([('branch',
            make_pipeline(
                ZeroCount(),
                GaussianNB()
            )
        )]), FunctionTransformer(lambda X: X))
    ),
    KNeighborsClassifier(n_neighbors=18, p="uniform", weights=2)
)"""
    assert expected_code == generate_pipeline_code(pipeline, tpot_obj.operators)


def test_generate_import_code():
    """Assert that generate_import_code() returns the correct set of dependancies for a given pipeline"""
    tpot_obj = TPOTClassifier()
    pipeline = creator.Individual.\
        from_string('GaussianNB(RobustScaler(input_matrix))', tpot_obj._pset)

    expected_code = """import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \\
    train_test_split(features, tpot_data['class'], random_state=42)
"""
    assert expected_code == generate_import_code(pipeline, tpot_obj.operators)

def test_mutNodeReplacement():
    """Assert that mutNodeReplacement() returns the correct type of mutation node in a fixed pipeline"""
    tpot_obj = TPOTClassifier(random_state=42)
    pipeline = tpot_obj._toolbox.individual()
    old_ret_type_list = [node.ret for node in pipeline]
    old_prims_list = [node for node in pipeline if node.arity != 0]
    mut_ind = mutNodeReplacement(pipeline, pset = tpot_obj._pset)
    new_ret_type_list = [node.ret for node in mut_ind[0]]
    new_prims_list = [node for node in mut_ind[0] if node.arity != 0]
    if new_prims_list == old_prims_list: # Terminal mutated
        assert new_ret_type_list == old_ret_type_list
    else: # Primitive mutated
        diff_prims = list(set(new_prims_list).symmetric_difference(old_prims_list))
        assert diff_prims[0].ret == diff_prims[1].ret
    assert mut_ind[0][0].ret == Output_DF


def test_export_pipeline():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed complex pipeline"""
    tpot_obj = TPOTClassifier()
    pipeline = creator.Individual.\
        from_string("GaussianNB(CombineDFs(ZeroCount(input_matrix), RobustScaler(input_matrix)))", tpot_obj._pset)

    expected_code = """import numpy as np

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from tpot.build_in_operators import ZeroCount

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \\
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    make_union(
        ZeroCount(),
        RobustScaler()
    ),
    GaussianNB()
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline,tpot_obj.operators)


def test_export_pipeline_2():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed simple pipeline (only one classifier)"""
    tpot_obj = TPOTClassifier()
    pipeline = creator.Individual.\
        from_string("GaussianNB(input_matrix)", tpot_obj._pset)
    expected_code = """import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \\
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = GaussianNB()

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators)

def test_export_pipeline_3():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed simple pipeline with a preprocessor"""
    tpot_obj = TPOTClassifier()
    pipeline = creator.Individual.\
        from_string("GaussianNB(MaxAbsScaler(input_matrix))", tpot_obj._pset)

    expected_code = """import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \\
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    GaussianNB()
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators)


def test_operator_export():
    """Assert that a TPOT operator can export properly with a function as a parameter to a classifier"""
    export_string = TPOTSelectKBest.export(5)
    assert export_string == "SelectKBest(score_func=f_classif, k=5)"


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
    assert TPOTSelectKBest.type() == "Preprocessor or Selector"


def test_get_by_name():
    """Assert that the Operator class returns operators by name appropriately"""
    tpot_obj = TPOTClassifier()
    assert get_by_name("SelectKBest", tpot_obj.operators).__class__ == TPOTSelectKBest.__class__


def test_gen():
    """Assert that TPOT's gen_grow_safe function returns a pipeline of expected structure"""
    tpot_obj = TPOTClassifier()

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
