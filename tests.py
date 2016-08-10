# -*- coding: utf-8 -*-

"""
TPOT Unit Tests
"""

from tpot import TPOT
from tpot.tpot import positive_integer, float_range
from tpot.export_utils import export_pipeline, generate_import_code, _indent, generate_pipeline_code
from tpot.decorators import _gp_new_generation
from tpot.types import Output_DF
from tpot.indices import non_feature_columns, GUESS_COL

from tpot.operators import Operator, CombineDFs
from tpot.operators.classifiers import Classifier, TPOTDecisionTreeClassifier
from tpot.operators.preprocessors import Preprocessor
from tpot.operators.selectors import Selector, TPOTSelectKBest

import numpy as np
from collections import Counter
import warnings
import inspect

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.feature_selection import RFE, SelectPercentile, f_classif, SelectKBest, SelectFwe, VarianceThreshold
from sklearn import metrics

from deap import creator
from tqdm import tqdm

# Set up the MNIST data set for testing
mnist_data = load_digits()
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(mnist_data.data.astype(np.float64), mnist_data.target.astype(np.float64), random_state=42)

# Training data group is 0 testing data group is 1
training_data = np.insert(training_features, 0, training_classes, axis=1)  # Insert the classes
training_data = np.insert(training_data, 0, np.zeros((training_data.shape[0],)), axis=1)  # Insert the group
testing_data = np.insert(testing_features, 0, np.zeros((testing_features.shape[0],)), axis=1)  # Insert the classes
testing_data = np.insert(testing_data, 0, np.ones((testing_data.shape[0],)), axis=1)  # Insert the group

# Insert guess
most_frequent_class = Counter(training_classes).most_common(1)[0][0]
data = np.concatenate([training_data, testing_data])
data = np.insert(data, 0, np.array([most_frequent_class] * data.shape[0]), axis=1)


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
    assert tpot_obj._training_classes is None
    assert tpot_obj._training_features is None
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
    tpot_obj._training_classes = training_classes
    tpot_obj._training_features = training_features
    tpot_obj.pbar = tqdm(total=1, disable=True)
    known_score = 0.9202817574915823  # Assumes use of the TPOT balanced_accuracy function

    # Reify pipeline with known score
    tpot_obj._optimized_pipeline = creator.Individual.\
        from_string('DecisionTreeClassifier(input_matrix, 0.5)', tpot_obj._pset)

    # Get score from TPOT
    score = tpot_obj.score(testing_features, testing_classes)

    # http://stackoverflow.com/questions/5595425/
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

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
    tpot_obj._training_classes = training_classes
    tpot_obj._training_features = training_features
    tpot_obj._optimized_pipeline = creator.Individual.\
        from_string('DecisionTreeClassifier(input_matrix, 0.5)', tpot_obj._pset)

    result = tpot_obj.predict(testing_features)

    assert result.shape == (testing_features.shape[0],)


def test_fit():
    """Assert that the TPOT fit function provides an optimized pipeline"""
    tpot_obj = TPOT(random_state=42, population_size=1, generations=1, verbosity=0)
    tpot_obj.fit(training_features, training_classes)

    assert isinstance(tpot_obj._training_features, np.ndarray)
    assert isinstance(tpot_obj._training_classes, np.ndarray)
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


def check_classifier(op):
    """Assert that a TPOT classifier outputs the same as its sklearn counterpart"""
    tpot_obj = TPOT(random_state=42)

    prng = np.random.RandomState(42)

    args = []
    for type_ in op.parameter_types()[0][1:]:
        args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)

    result = op(data, *args)

    clf = op._merge_with_default_params(op.preprocess_args(*args))
    clf.fit(training_features, training_classes)

    all_features = np.delete(data, non_feature_columns, axis=1)

    assert np.array_equal(result[:, GUESS_COL], clf.predict(all_features))


def check_selector(op):
    """Assert that a TPOT feature selector outputs the same as its sklearn counterpart"""
    tpot_obj = TPOT(random_state=42)

    prng = np.random.RandomState(42)

    args = []
    for type_ in op.parameter_types()[0][1:]:
        args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)

    result = op(data, *args)

    sel = op._merge_with_default_params(op.preprocess_args(*args))

    with warnings.catch_warnings():
        # Ignore warnings about constant features
        warnings.simplefilter('ignore', category=UserWarning)
        sel.fit(training_features, training_classes)

    mask = sel.get_support(True)

    assert np.array_equal(
        np.delete(data, non_feature_columns + [x + len(non_feature_columns) for x in mask], axis=1),
        np.delete(result, non_feature_columns, axis=1)
    )


def check_preprocessor(op):
    """Assert that a TPOT feature preprocessor outputs the same as its sklearn counterpart"""
    tpot_obj = TPOT(random_state=42)

    prng = np.random.RandomState(42)

    args = []
    for type_ in op.parameter_types()[0][1:]:
        args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)

    result = op(data, *args)

    prp = op._merge_with_default_params(op.preprocess_args(*args))
    prp.fit(training_features.astype(np.float64))
    all_features = np.delete(data, non_feature_columns, axis=1)
    sklearn_result = prp.transform(all_features.astype(np.float64))

    assert np.allclose(np.delete(result, non_feature_columns, axis=1), sklearn_result)


def check_export(op):
    """Assert that a TPOT operator exports as expected"""
    tpot_obj = TPOT(random_state=42)

    prng = np.random.RandomState(42)

    args = []
    for type_ in op.parameter_types()[0][1:]:
        args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)

    export_string = op.export(*args)

    assert export_string.startswith(op.__name__)


def test_operators():
    """Assert that the TPOT operators match the output of their sklearn counterparts"""
    for op in Operator.inheritors():
        check_export.description = ("Assert that the TPOT {} operator exports "
                                    "as expected".format(op.__name__))
        yield check_export, op

        if isinstance(op, Classifier):
            check_classifier.description = ("Assert that the TPOT {} classifier "
                                            "matches the output of the sklearn "
                                            "counterpart".format(op.__name__))
            yield check_classifier, op
        elif isinstance(op, Preprocessor):
            check_preprocessor.description = ("Assert that the TPOT {} feature preprocessor "
                                              "matches the output of the sklearn "
                                              "counterpart".format(op.__name__))
            yield check_preprocessor, op
        elif isinstance(op, Selector):
            check_selector.description = ("Assert that the TPOT {} feature selector "
                                          "matches the output of the sklearn "
                                          "counterpart".format(op.__name__))
            yield check_selector, op


def test_operators_2():
    """Assert that TPOT operators return the input_matrix when no features are supplied"""
    assert np.array_equal(
        data[:, :3],
        TPOTDecisionTreeClassifier()(data[:, :3], 0.5)
    )


def test_combine_dfs():
    """Assert that the TPOT CombineDFs operator creates a combined feature set from two input sets"""
    features1 = np.delete(data, non_feature_columns, axis=1)
    features2 = np.delete(data, non_feature_columns, axis=1)

    combined_features = np.concatenate([features1, features2], axis=1)

    for col in non_feature_columns:
        combined_features = np.insert(combined_features, 0, data[:, col], axis=1)

    assert np.array_equal(CombineDFs()(data, data), combined_features)


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
                30.0],
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
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
features = tpot_data.view((np.float64, len(tpot_data.dtype.names)))
features = np.delete(features, tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
train_test_split(features, tpot_data['class'], random_state=42)
"""

    assert expected_code == generate_import_code(pipeline)


def test_export_pipeline():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed pipeline"""
    tpot_obj = TPOT()
    pipeline = creator.Individual.\
        from_string("KNeighborsClassifier(CombineDFs(GradientBoostingClassifier(input_matrix, 38.0, 0.87, 30.0), RFE(input_matrix, 0.17999999999999999)), 18, 33)", tpot_obj._pset)

    expected_code = """import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
features = tpot_data.view((np.float64, len(tpot_data.dtype.names)))
features = np.delete(features, tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
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

def test_scoring_functions_1():
    """Assert that the default _balanced_accuracy is used when no scoring function is passed"""
    tpot_obj = TPOT()

    assert(tpot_obj.scoring_function == tpot_obj._balanced_accuracy)
'''
def test_scoring_functions_2():
    """Assert that a custom classification-based scoring function uses the predict function of each classifier"""
    def custom_scoring_function(y_true, y_pred):
        return 1.0

    tpot_obj = TPOT(scoring_function=custom_scoring_function)
    
    tpot_obj._training_classes = training_classes
    tpot_obj._training_features = training_features
    tpot_obj.pbar = tqdm(total=1, disable=True)

    # Reify pipeline with known score
    tpot_obj._optimized_pipeline = creator.Individual.\
        from_string('_logistic_regression(input_df, 1.0, 0, True)', tpot_obj._pset)

    # Get score from TPOT
    score = tpot_obj.score(testing_features, testing_classes)

    assert(tpot_obj.scoring_function == custom_scoring_function)
    

def test_scoring_functions_3():
    """Assert that the parse_scoring_docstring works for classification metrics"""

    tpot_obj = TPOT()

    assert(tpot_obj._parse_scoring_docstring(tpot_obj._balanced_accuracy) == 'predict')

    for scoring_func in [metrics.fbeta_score,
                         metrics.jaccard_similarity_score,
                         metrics.matthews_corrcoef,
                         metrics.f1_score,
                         metrics.precision_score,
                         metrics.silhouette_score,
                         metrics.zero_one_loss,
                         metrics.accuracy_score,
                         metrics.recall_score,
                         metrics.hamming_loss]:

        tpot_obj = TPOT(scoring_function=scoring_func)
        assert(tpot_obj._parse_scoring_docstring(tpot_obj.scoring_function) == 'predict')

    for scoring_func in [metrics.log_loss]:
        tpot_obj = TPOT(scoring_function=scoring_func)
        assert(tpot_obj._parse_scoring_docstring(tpot_obj.scoring_function) == 'predict_proba')

    for scoring_func in [metrics.hinge_loss]:
        tpot_obj = TPOT(scoring_function=scoring_func)
        assert(tpot_obj._parse_scoring_docstring(tpot_obj.scoring_function) == 'decision_function')

def test_scoring_functions_4():
    """ Assert that a loss function gets the sign flipped and the correct function is used in evaluation """

    tpot_obj = TPOT(population_size=1, generations=1, scoring_function=metrics.hamming_loss)
    tpot_obj.fit(training_features, training_classes)
    
    assert(tpot_obj.score_sign == -1)
    assert(tpot_obj.clf_eval_func == 'predict')

def test_train_model_and_predict_2():
    """ Assert that training an individual classifier and predicting makes use of correct function, un/pickling as necessary"""

    tpot_obj = TPOT(population_size=1, generations=1, scoring_function=metrics.hamming_loss)
    tpot_obj.clf_eval_func = tpot_obj._parse_scoring_docstring(tpot_obj.scoring_function)

    try:
        result = tpot_obj._train_model_and_predict(training_testing_data, LinearSVC, C=5., penalty='l1', dual=False)
        [np.loads(x) for x in result.loc[:, 'guess']]
        assert False # Should be unreachable
    except:
        pass

    tpot_obj = TPOT(population_size=1, generations=1, scoring_function=metrics.log_loss)
    tpot_obj.clf_eval_func = tpot_obj._parse_scoring_docstring(tpot_obj.scoring_function)

    try:
        result = tpot_obj._train_model_and_predict(training_testing_data, GaussianNB)
        [np.loads(x) for x in result.loc[:, 'guess']]
    except:
        assert False # Should be unreachable

    tpot_obj = TPOT(population_size=1, generations=1, scoring_function=metrics.hinge_loss)
    tpot_obj.clf_eval_func = tpot_obj._parse_scoring_docstring(tpot_obj.scoring_function)

    try:
        result = tpot_obj._train_model_and_predict(training_testing_data, LinearSVC, C=5., penalty='l1', dual=False)
        [np.loads(x) for x in result.loc[:, 'guess']]
    except:
        assert False # Should be unreachable
'''

def test_float_range_3():
    """Assert that the TPOT CLI interface's float range throws an exception when input is not a float"""
    try:
        float_range('foobar')
        assert False  # Should be unreachable
    except Exception:
        pass
