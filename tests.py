"""
    Unit tests for TPOT.
"""

from tpot import TPOT
from tpot.export_utils import generate_import_code, replace_function_calls, unroll_nested_fuction_calls
from tpot.decorators import _gp_new_generation

import pandas as pd
import numpy as np
from collections import Counter
from itertools import compress
import warnings
import inspect
import hashlib
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.feature_selection import RFE, SelectPercentile, f_classif, SelectKBest, SelectFwe, VarianceThreshold

from deap import creator
from tqdm import tqdm

# Set up the MNIST data set for testing
mnist_data = load_digits()
training_features, testing_features, training_classes, testing_classes =\
        train_test_split(mnist_data.data, mnist_data.target, random_state=42)

training_data = pd.DataFrame(training_features)
training_data['class'] = training_classes
training_data['group'] = 'training'

testing_data = pd.DataFrame(testing_features)
testing_data['class'] = 0
testing_data['group'] = 'testing'

training_testing_data = pd.concat([training_data, testing_data])
most_frequent_class = Counter(training_classes).most_common(1)[0][0]
training_testing_data['guess'] = most_frequent_class

expert_source = [[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, \
0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, \
1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1], np.random.rand(64)]

for column in training_testing_data.columns.values:
    if type(column) != str:
        training_testing_data.rename(columns={column: str(column).zfill(5)}, inplace=True)


def test_init():
    """Ensure that the TPOT instantiator stores the TPOT variables properly"""

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
    assert tpot_obj.update_checked is True
    assert tpot_obj._optimized_pipeline is None
    assert tpot_obj._training_classes is None
    assert tpot_obj._training_features is None
    assert tpot_obj.scoring_function == dummy_scoring_func
    assert tpot_obj._pset
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


def test_get_params():
    """Ensure that get_params returns the exact dictionary of parameters used by TPOT"""
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

def test_train_model_and_predict():
    """Ensure that the TPOT train_model_and_predict returns the input dataframe when it has only 3 columns i.e. class, group, guess"""

    tpot_obj = TPOT()

    assert np.array_equal(training_testing_data.ix[:, -3:], tpot_obj._train_model_and_predict(training_testing_data.ix[:, -3:], LinearSVC, C=5., penalty='l1', dual=False))


def test_score():
    """Ensure that the TPOT score function raises a ValueError when no optimized pipeline exists"""

    tpot_obj = TPOT()

    try:
        tpot_obj.score(testing_features, testing_classes)
        assert False  # Should be unreachable
    except ValueError:
        pass


def test_predict():
    """Ensure that the TPOT predict function raises a ValueError when no optimized pipeline exists"""

    tpot_obj = TPOT()

    try:
        tpot_obj.predict(testing_features)
        assert False  # Should be unreachable
    except ValueError:
        pass

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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        selector = SelectKBest(f_classif, k=42)
        selector.fit(training_features, training_class_vals)
        mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

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

    assert np.array_equal(tpot_obj._ekf(training_testing_data, ekf_index=1, k_best=5), ekf_subset_array)

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
