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
<<<<<<< HEAD
from collections import Counter
from itertools import compress
import warnings
import inspect
import hashlib
=======
import inspect
import random

>>>>>>> 589a020adff6a725584cb283849e09bb2c37b8b2
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

from deap import creator
from tqdm import tqdm

# Set up the MNIST data set for testing
mnist_data = load_digits()
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(mnist_data.data.astype(np.float64), mnist_data.target.astype(np.float64), random_state=42)

<<<<<<< HEAD
training_testing_data = pd.concat([training_data, testing_data])
most_frequent_class = Counter(training_classes).most_common(1)[0][0]
training_testing_data['guess'] = most_frequent_class

expert_source = [[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, \
0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, \
1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1], np.random.rand(64)]

for column in training_testing_data.columns.values:
    if type(column) != str:
        training_testing_data.rename(columns={column: str(column).zfill(5)}, inplace=True)
=======
np.random.seed(42)
random.seed(42)
>>>>>>> 589a020adff6a725584cb283849e09bb2c37b8b2


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
<<<<<<< HEAD
    assert tpot_obj.non_feature_columns

'''
def test_unroll_nested():
    """Ensure that export utils' unroll_nested_fuction_calls outputs pipeline_list as expected"""

    tpot_obj = TPOT()

    expected_list = [['result1', '_logistic_regression', 'input_df', '1.0', '0', 'True']]

    pipeline = creator.Individual.\
        from_string('_logistic_regression(input_df, 1.0, 0, True)', tpot_obj._pset)

    pipeline_list = unroll_nested_fuction_calls(pipeline)

    assert expected_list == pipeline_list


def test_unroll_nested_2():
    """Ensure that export utils' unroll_nested_fuction_calls outputs pipelines with nested function calls as expectd"""

    tpot_obj = TPOT()

    expected_list = [['result1', '_select_percentile', 'input_df', '40'], ['result2', '_extra_trees', 'result1', '32', '0.62', '0.45']]

    pipeline = creator.Individual.\
        from_string('_extra_trees(_select_percentile(input_df, 40), 32, 0.62, 0.45000000000000001)', tpot_obj._pset)

    pipeline_list = unroll_nested_fuction_calls(pipeline)

    assert expected_list == pipeline_list
'''

def test_generate_import_code():
    """Ensure export utils' generate_import_code outputs as expected"""

    reference_code = """\
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)
"""

    pipeline = [['result1', '_variance_threshold', 'input_df', '100.0'],
                ['result2', '_pca', 'input_df', '66', '34'],
                ['result3', '_combine_dfs', 'result2', 'result1'],
                ['result4', '_logistic_regression', 'result3', '0.12030075187969924', '0', 'True']]

    import_code = generate_import_code(pipeline)

    assert reference_code == import_code


def test_generate_import_code_2():
    """Ensure export utils' generate_import_code outputs as expected when using multiple classes from the same module"""

    reference_code = """\
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.decomposition import FastICA, RandomizedPCA
from sklearn.linear_model import LogisticRegression

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)
"""

    pipeline = [['result1', '_fast_ica', 'input_df', '5', '0.1'],
                ['result2', '_pca', 'input_df', '66', '34'],
                ['result3', '_combine_dfs', 'result2', 'result1'],
                ['result4', '_logistic_regression', 'result3', '0.12030075187969924', '0', 'True']]

    import_code = generate_import_code(pipeline)

    assert reference_code == import_code


def test_replace_function_calls():
    """Ensure export utils' replace_function_calls outputs as expected"""

    reference_code = """
result1 = tpot_data.copy()

# Use Scikit-learn's SelectKBest for feature selection
training_features = result1.loc[training_indices].drop('class', axis=1)
training_class_vals = result1.loc[training_indices, 'class'].values

if len(training_features.columns.values) == 0:
    result1 = result1.copy()
else:
    selector = SelectKBest(f_classif, k=min(26, len(training_features.columns)))
    selector.fit(training_features.values, training_class_vals)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    result1 = result1[mask_cols]

# Perform classification with a decision tree classifier
dtc2 = DecisionTreeClassifier(min_weight_fraction_leaf=0.1)
dtc2.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)
result2 = result1.copy()
result2['dtc2-classification'] = dtc2.predict(result2.drop('class', axis=1).values)
"""

    pipeline = [['result1', '_select_kbest', 'input_df', '26'],
                ['result2', '_decision_tree', 'result1', '0.1']]

    exported_code = replace_function_calls(pipeline)

    assert reference_code == exported_code


def test_replace_function_calls_2():
    """Ensure export utils' replace_function_calls generates no exceptions"""

    tpot_obj = TPOT()

    for prim in tpot_obj._pset.primitives[pd.DataFrame]:
        simple_pipeline = ['result1']
        simple_pipeline.append(prim.name)

        for arg in prim.args:
            simple_pipeline.append(tpot_obj._pset.terminals[arg][0].value)

        replace_function_calls([simple_pipeline])
=======
>>>>>>> 589a020adff6a725584cb283849e09bb2c37b8b2


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

<<<<<<< HEAD
def test_train_model_and_predict():
    """Ensure that the TPOT train_model_and_predict returns the input dataframe when it has only 3 columns i.e. class, group, guess"""

    tpot_obj = TPOT()

    assert np.array_equal(training_testing_data.ix[:, -3:], tpot_obj._train_model_and_predict(training_testing_data.ix[:, -3:], LinearSVC, C=5., penalty='l1', dual=False))

=======
>>>>>>> 589a020adff6a725584cb283849e09bb2c37b8b2

def test_score():
    """Assert that the TPOT score function raises a ValueError when no optimized pipeline exists"""

    tpot_obj = TPOT()

    try:
        tpot_obj.score(testing_features, testing_classes)
        assert False  # Should be unreachable
    except ValueError:
        pass


<<<<<<< HEAD
=======
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


>>>>>>> 589a020adff6a725584cb283849e09bb2c37b8b2
def test_predict():
    """Assert that the TPOT predict function raises a ValueError when no optimized pipeline exists"""

    tpot_obj = TPOT()

    try:
        tpot_obj.predict(testing_features)
        assert False  # Should be unreachable
    except ValueError:
        pass

<<<<<<< HEAD
def test_export():
    """Ensure that the TPOT export function raises a ValueError when no optimized pipeline exists"""

    tpot_obj = TPOT()

    try:
        tpot_obj.export('will_not_output')
        assert False  # Should be unreachable
    except ValueError:
        pass


def test_combine_dfs():
    """Check combine_dfs operator"""
    tpot_obj = TPOT()

    df1 = pd.DataFrame({'a': range(10),
                        'b': range(10, 20)})

    df2 = pd.DataFrame({'b': range(10, 20),
                        'c': range(20, 30)})

    combined_df = pd.DataFrame({'a': range(10),
                                'b': range(10, 20),
                                'c': range(20, 30)})

    assert tpot_obj._combine_dfs(df1, df2).equals(combined_df)


def test_combine_dfs_2():
    """Check combine_dfs operator when the dataframes are equal"""
    tpot_obj = TPOT()

    df1 = pd.DataFrame({'a': range(10),
                        'b': range(10, 20)})

    df2 = pd.DataFrame({'a': range(10),
                        'b': range(10, 20)})

    combined_df = pd.DataFrame({'a': range(10),
                                'b': range(10, 20)})

    assert tpot_obj._combine_dfs(df1, df2).equals(combined_df)

def test_select_kbest():
    """Ensure that the TPOT select kbest outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._select_kbest(training_testing_data.ix[:, -3:], 1), training_testing_data.ix[:, -3:])


def test_select_kbest_2():
    """Ensure that the TPOT select kbest outputs the same result as sklearn select kbest when k<0"""
    tpot_obj = TPOT()
    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    training_class_vals = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        selector = SelectKBest(f_classif, k=1)
        selector.fit(training_features, training_class_vals)
        mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

    assert np.array_equal(tpot_obj._select_kbest(training_testing_data, -1), training_testing_data[mask_cols])


def test_select_kbest_3():
    """Ensure that the TPOT select kbest outputs the same result as sklearn select kbest when k> no. of features"""
    tpot_obj = TPOT()
    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    training_class_vals = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        selector = SelectKBest(f_classif, k=64)
        selector.fit(training_features, training_class_vals)
        mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

    assert np.array_equal(tpot_obj._select_kbest(training_testing_data, 100), training_testing_data[mask_cols])


def test_select_kbest_4():
    """Ensure that the TPOT select kbest outputs the same result as sklearn select kbest when 0< k< features"""
    tpot_obj = TPOT()
    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    training_class_vals = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values
=======

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
>>>>>>> 589a020adff6a725584cb283849e09bb2c37b8b2

    assert isinstance(tpot_obj._optimized_pipeline, creator.Individual)
    assert tpot_obj.gp_generation == 0

<<<<<<< HEAD
    assert np.array_equal(tpot_obj._select_kbest(training_testing_data, 42), training_testing_data[mask_cols])


def test_ekf_1():
    """Ensure that the expert knowledge provided mask chooses the right subset of input data to train"""
    tpot_obj = TPOT()
    tpot_obj.expert_source = expert_source
    non_feature_columns = ['class', 'group', 'guess']
    
    ekf_index = 0
    ekf_source_test = np.array(tpot_obj.expert_source[ekf_index])
    ekf_subset_test = list(compress(training_testing_data.columns.values, ekf_source_test)) + non_feature_columns

    ekf_subset_array = training_testing_data.loc[:, ekf_subset_test].copy()
    # ekf_subset_array = pd.concat([ekf_subset_array, ekf_training_testing[non_feature_columns]], axis=1)

    assert np.array_equal(tpot_obj._ekf(training_testing_data, ekf_index, k_best=10), ekf_subset_array)

def test_ekf_2():
    """ Ensure that the expert knowledge provided subset chooses the right subset of input data to train"""
    tpot_obj = TPOT()
    tpot_obj.expert_source = expert_source
    non_feature_columns = ['class', 'group', 'guess']

    ekf_index = 1
    k_best = 5

    ekf_source_test = np.argsort(expert_source[ekf_index])[::-1][:]
    ekf_source_test = ekf_source_test[:k_best]

    ekf_subset_test = (training_testing_data.columns.values[ekf_source_test]).tolist() + non_feature_columns
    ekf_subset_array = training_testing_data.loc[:, ekf_subset_test].copy()
    # ekf_subset_array = pd.concat([ekf_subset_array, training_testing_data[non_feature_columns]], axis=1)
=======

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
>>>>>>> 589a020adff6a725584cb283849e09bb2c37b8b2

    assert np.array_equal(tpot_obj._ekf(training_testing_data, ekf_index=1, k_best=5), ekf_subset_array)

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


<<<<<<< HEAD
    assert(tpot_obj.gp_generation == 1)

'''
def test_mdr_2_from_2():
    """Assert that the mdr integration chooses the right set of features to combine, and outputs the right training dataset with the new constructed feature. """
    training_features = np.array([[2,0],
                        [0, 0],
                        [0, 1],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [0, 1],
                        [0, 0],
                        [0, 0],
                        [0, 1],
                        [0, 0],
                        [0, 0],
                        [0, 0],
                        [1, 1],
                        [1, 1]])
    training_classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    testing_features =          np.array([[2, 2],
                                [1, 1], 
                                [0, 0], 
                                [0, 0], 
                                [0, 0], 
                                [0, 0], 
                                [1, 1], 
                                [0, 0], 
                                [0, 0], 
                                [0, 0], 
                                [0, 1], 
                                [1, 0], 
                                [0, 0], 
                                [1, 0], 
                                [0, 0]])
    testing_classes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    training_data = pd.DataFrame(training_features)
    training_data['class'] = training_classes
    training_data['group'] = 'training'

    testing_data = pd.DataFrame(testing_features)
    testing_data['class'] = testing_classes
    testing_data['group'] = 'testing'

    training_testing_data = pd.concat([training_data, testing_data])
    most_frequent_class = Counter(training_classes).most_common(1)[0][0]
    training_testing_data['guess'] = most_frequent_class

    for column in training_testing_data.columns.values:
        if type(column) != str:
            training_testing_data.rename(columns={column: str(column).zfill(5)}, inplace=True)
    tpot_obj = TPOT()
    tie_break = 7
    default_label = 7
    num_features_to_combined = 2
    result = tpot_obj._mdr(training_testing_data, tie_break = tie_break, default_label = default_label, num_features_to_combined = num_features_to_combined)

    non_feature_columns = ['class', 'group', 'guess']
    all_features = training_testing_data.drop(non_feature_columns, axis=1).columns.values.tolist()
    mdr_hash = '-'.join(sorted(all_features))
    mdr_hash += 'MDR'
    mdr_hash += '-'.join([str(param) for param in [tie_break, default_label, num_features_to_combined]])
    mdr_identifier = 'ConstructedFeature-{}'.format(hashlib.sha224(mdr_hash.encode('UTF-8')).hexdigest())
    print (result[mdr_identifier])
    assert np.array_equal(result[mdr_identifier], [1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
'''
=======
def test_float_range_3():
    """Assert that the TPOT CLI interface's float range throws an exception when input is not a float"""
    try:
        float_range('foobar')
        assert False  # Should be unreachable
    except Exception:
        pass
>>>>>>> 589a020adff6a725584cb283849e09bb2c37b8b2
