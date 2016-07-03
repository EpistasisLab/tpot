"""
    Unit tests for TPOT.
"""

from tpot import TPOT
from tpot.export_utils import generate_import_code, replace_function_calls, unroll_nested_fuction_calls
from tpot.decorators import _gp_new_generation

import pandas as pd
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


def test_decision_tree():
    """Ensure that the TPOT decision tree method outputs the same as the sklearn decision tree"""

    tpot_obj = TPOT()
    result = tpot_obj._decision_tree(training_testing_data, 0.1)
    result = result[result['group'] == 'testing']

    dtc = DecisionTreeClassifier(min_weight_fraction_leaf=0.1, random_state=42)
    dtc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, dtc.predict(testing_features))


def test_decision_tree_2():
    """Ensure that the TPOT decision tree method outputs the same as the sklearn decision tree when min_weight=0"""

    tpot_obj = TPOT()
    result = tpot_obj._decision_tree(training_testing_data, 0.)
    result = result[result['group'] == 'testing']

    dtc = DecisionTreeClassifier(min_weight_fraction_leaf=0., random_state=42)
    dtc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, dtc.predict(testing_features))


def test_decision_tree_3():
    """Ensure that the TPOT decision tree method outputs the same as the sklearn decision tree when min_weight>0.5"""

    tpot_obj = TPOT()
    result = tpot_obj._decision_tree(training_testing_data, 0.6)
    result = result[result['group'] == 'testing']

    dtc = DecisionTreeClassifier(min_weight_fraction_leaf=0.5, random_state=42)
    dtc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, dtc.predict(testing_features))


def test_random_forest():
    """Ensure that the TPOT random forest method outputs the same as the sklearn random forest when"""

    tpot_obj = TPOT()
    result = tpot_obj._random_forest(training_testing_data, 0.1)
    result = result[result['group'] == 'testing']

    rfc = RandomForestClassifier(n_estimators=500, min_weight_fraction_leaf=0.1, random_state=42, n_jobs=-1)
    rfc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, rfc.predict(testing_features))


def test_random_forest_2():
    """Ensure that the TPOT random forest method outputs the same as the sklearn random forest when min_weight>0.5"""

    tpot_obj = TPOT()
    result = tpot_obj._random_forest(training_testing_data, 0.6)
    result = result[result['group'] == 'testing']

    rfc = RandomForestClassifier(n_estimators=500, min_weight_fraction_leaf=0.5, random_state=42, n_jobs=-1)
    rfc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, rfc.predict(testing_features))


def test_logistic_regression():
    """Ensure that the TPOT logistic regression classifier outputs the same as the sklearn LogisticRegression"""

    tpot_obj = TPOT()
    result = tpot_obj._logistic_regression(training_testing_data, 5., 0, True)
    result = result[result['group'] == 'testing']

    lrc = LogisticRegression(C=5., penalty='l1', dual=False, random_state=42)
    lrc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, lrc.predict(testing_features))


def test_knnc():
    """Ensure that the TPOT k-nearest neighbor classifier outputs the same as the sklearn classifier"""

    tpot_obj = TPOT()
    result = tpot_obj._knnc(training_testing_data, 100, 0)
    result = result[result['group'] == 'testing']

    knnc = KNeighborsClassifier(n_neighbors=100, weights='uniform')
    knnc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, knnc.predict(testing_features))


def test_knnc_2():
    """Ensure that the TPOT k-nearest neighbor classifier outputs the same as the sklearn classifier when n_neighbor=0"""

    tpot_obj = TPOT()
    result = tpot_obj._knnc(training_testing_data, 0, 0)
    result = result[result['group'] == 'testing']

    knnc = KNeighborsClassifier(n_neighbors=2, weights='uniform')
    knnc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, knnc.predict(testing_features))


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


def test_score_2():
    """Ensure that the TPOT score function outputs a known score for a fixed pipeline"""

    tpot_obj = TPOT()
    tpot_obj._training_classes = training_classes
    tpot_obj._training_features = training_features
    tpot_obj.pbar = tqdm(total=1, disable=True)
    known_score = 0.981993770448  # Assumes use of the TPOT balanced_accuracy function

    # Reify pipeline with known score
    tpot_obj._optimized_pipeline = creator.Individual.\
        from_string('_logistic_regression(input_df, 1.0, 0, True)', tpot_obj._pset)

    # Get score from TPOT
    score = tpot_obj.score(testing_features, testing_classes)

    # http://stackoverflow.com/questions/5595425/
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    assert isclose(known_score, score)


def test_predict():
    """Ensure that the TPOT predict function raises a ValueError when no optimized pipeline exists"""

    tpot_obj = TPOT()

    try:
        tpot_obj.predict(testing_features)
        assert False  # Should be unreachable
    except ValueError:
        pass


def test_predict_2():
    """Ensure that the TPOT predict function returns a DataFrame of shape (num_testing_rows,)"""

    tpot_obj = TPOT()
    tpot_obj._training_classes = training_classes
    tpot_obj._training_features = training_features
    tpot_obj._optimized_pipeline = creator.Individual.\
        from_string('_logistic_regression(input_df, 1.0, 0, True)', tpot_obj._pset)

    result = tpot_obj.predict(testing_features)

    assert result.shape == (testing_features.shape[0],)


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


def test_rfe():
    """Ensure that the TPOT RFE outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._rfe(training_testing_data.ix[:, -3:], 0, 0.1), training_testing_data.ix[:, -3:])


def test_rfe_2():
    """Ensure that the TPOT RFE outputs the same result as the sklearn rfe when num_features>no. of features in the dataframe """
    tpot_obj = TPOT()

    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    estimator = LinearSVC()
    rfe = RFE(estimator, 100, step=0.1)
    rfe.fit(training_features, training_classes)
    mask = rfe.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

    assert np.array_equal(training_testing_data[mask_cols], tpot_obj._rfe(training_testing_data, 64, 0.1))


def test_select_percentile():
    """Ensure that the TPOT select percentile outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._select_percentile(training_testing_data.ix[:, -3:], 0), training_testing_data.ix[:, -3:])


def test_select_percentile_2():
    """Ensure that the TPOT select percentile outputs the same result as sklearn Select Percentile when percentile < 0"""
    tpot_obj = TPOT()
    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    training_class_vals = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        selector = SelectPercentile(f_classif, percentile=0)
        selector.fit(training_features, training_class_vals)
        mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

    assert np.array_equal(tpot_obj._select_percentile(training_testing_data, -1), training_testing_data[mask_cols])


def test_select_percentile_3():
    """Ensure that the TPOT select percentile outputs the same result as sklearn select percentile when percentile > 100"""
    tpot_obj = TPOT()
    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    training_class_vals = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        selector = SelectPercentile(f_classif, percentile=100)
        selector.fit(training_features, training_class_vals)
        mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

    assert np.array_equal(tpot_obj._select_percentile(training_testing_data, 120), training_testing_data[mask_cols])


def test_select_percentile_4():
    """Ensure that the TPOT select percentile outputs the same result as sklearn select percentile when 0 < percentile < 100"""
    tpot_obj = TPOT()
    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    training_class_vals = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        selector = SelectPercentile(f_classif, percentile=42)
        selector.fit(training_features, training_class_vals)
        mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

    assert np.array_equal(tpot_obj._select_percentile(training_testing_data, 42), training_testing_data[mask_cols])


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


def test_select_fwe():
    """Ensure that the TPOT select fwe outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._select_fwe(training_testing_data.ix[:, -3:], 0.005), training_testing_data.ix[:, -3:])


def test_select_fwe_2():
    """Ensure that the TPOT select fwe outputs the same result as sklearn fwe when alpha > 0.05"""
    tpot_obj = TPOT()
    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    training_class_vals = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        selector = SelectFwe(f_classif, alpha=0.05)
        selector.fit(training_features, training_class_vals)
        mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

    assert np.array_equal(tpot_obj._select_fwe(training_testing_data, 1), training_testing_data[mask_cols])


def test_select_fwe_3():
    """Ensure that the TPOT select fwe outputs the same result as sklearn fwe when alpha < 0.001"""
    tpot_obj = TPOT()
    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    training_class_vals = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        selector = SelectFwe(f_classif, alpha=0.001)
        selector.fit(training_features, training_class_vals)
        mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

    assert np.array_equal(tpot_obj._select_fwe(training_testing_data, 0.0001), training_testing_data[mask_cols])


def test_select_fwe_4():
    """Ensure that the TPOT select fwe outputs the same result as sklearn fwe when 0.001 < alpha < 0.05"""
    tpot_obj = TPOT()
    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    training_class_vals = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        selector = SelectFwe(f_classif, alpha=0.042)
        selector.fit(training_features, training_class_vals)
        mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

    assert np.array_equal(tpot_obj._select_fwe(training_testing_data, 0.042), training_testing_data[mask_cols])


def test_variance_threshold():
    """Ensure that the tpot variance_threshold function behaves the same as the sklearn classifier"""
    tpot_obj = TPOT()
    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    selector = VarianceThreshold(threshold=0)
    selector.fit(training_features)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

    assert np.array_equal(tpot_obj._variance_threshold(training_testing_data, 0), training_testing_data[mask_cols])


def test_standard_scaler():
    """Ensure that the TPOT standard scaler outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()
    assert np.array_equal(tpot_obj._standard_scaler(training_testing_data.ix[:, -3:]), training_testing_data.ix[:, -3:])


def test_robust_scaler():
    """Ensure that the TPOT robust scaler outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._robust_scaler(training_testing_data.ix[:, -3:]), training_testing_data.ix[:, -3:])


def test_polynomial_features():
    """Ensure that the TPOT polynomial features outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._polynomial_features(training_testing_data.ix[:, -3:]), training_testing_data.ix[:, -3:])


def test_min_max_scaler():
    """Ensure that the TPOT min max scaler outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._min_max_scaler(training_testing_data.ix[:, -3:]), training_testing_data.ix[:, -3:])


def test_max_abs_scaler():
    """Ensure that the TPOT max abs scaler outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._max_abs_scaler(training_testing_data.ix[:, -3:]), training_testing_data.ix[:, -3:])


def test_rbf():
    """Assert that the TPOT RBFSampler outputs the input dataframe when # of
    training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._rbf(training_testing_data.ix[:, -3:], 0.1),
                          training_testing_data.ix[:, -3:])


def test_rbf_2():
    """Assert that RBF returns an object of the same type as the input dataframe

    Also assert that the number of rows is identical between the input dataframe
    and output dataframe.
    """
    tpot_obj = TPOT()

    input_df = training_testing_data
    output_df = tpot_obj._rbf(input_df, 0.1)

    assert type(input_df) == type(output_df)

    (in_rows, in_cols) = input_df.shape
    (out_rows, out_cols) = output_df.shape

    assert in_rows == out_rows


def test_fast_ica():
    """Assert that the TPOT FastICA preprocessor outputs the input dataframe
    when the number of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._fast_ica(training_testing_data.ix[:, -3:], 1.0),
                          training_testing_data.ix[:, -3:])


def test_fast_ica_2():
    """Assert that FastICA returns the same object type as the input object type.

    Also assert that the number of rows is identical between the input dataframe
    and output dataframe.
    """
    tpot_obj = TPOT()

    input_df = training_testing_data
    output_df = tpot_obj._fast_ica(input_df, 1.0)

    assert type(input_df) == type(output_df)

    (in_rows, in_cols) = input_df.shape
    (out_rows, out_cols) = output_df.shape

    assert in_rows == out_rows


def test_feat_agg():
    """Assert that the TPOT FeatureAgglomeration preprocessor outputs the input dataframe
    when the number of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._feat_agg(training_testing_data.ix[:, -3:], 5, 1, 1),
                          training_testing_data.ix[:, -3:])


def test_feat_agg_2():
    """Assert that FeatureAgglomeration returns the same object type as the input object type.

    Also assert that the number of rows is identical between the input dataframe
    and output dataframe.
    """
    tpot_obj = TPOT()

    input_df = training_testing_data
    output_df = tpot_obj._feat_agg(input_df, 5, 1, 1)

    assert type(input_df) == type(output_df)

    (in_rows, in_cols) = input_df.shape
    (out_rows, out_cols) = output_df.shape

    assert in_rows == out_rows


def test_nystroem():
    """Assert that the TPOT Nystroem preprocessor outputs the input dataframe
    when the number of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._nystroem(training_testing_data.ix[:, -3:], 1, 0.1, 1),
                          training_testing_data.ix[:, -3:])


def test_nystroem_2():
    """Assert that Nystroem returns the same object type as the input object type.

    Also assert that the number of rows is identical between the input dataframe
    and output dataframe.
    """
    tpot_obj = TPOT()

    input_df = training_testing_data
    output_df = tpot_obj._nystroem(input_df, 1, 0.1, 1)

    assert type(input_df) == type(output_df)

    (in_rows, in_cols) = input_df.shape
    (out_rows, out_cols) = output_df.shape

    assert in_rows == out_rows


def test_binarizer():
    """Ensure that the TPOT binarizer outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._binarizer(training_testing_data.ix[:, -3:], 0), training_testing_data.ix[:, -3:])


def test_pca():
    """Ensure that the TPOT PCA outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._pca(training_testing_data.ix[:, -3:], 1), training_testing_data.ix[:, -3:])


def test_zero_count():
    """Ensure that the TPOT _zero_count preprocessor outputs the input dataframe when no. of training features is 0"""
    tpot_obj = TPOT()

    assert np.array_equal(tpot_obj._zero_count(training_testing_data.ix[:, -3:]), training_testing_data.ix[:, -3:])


def test_zero_count_2():
    """Assert that the Zero Count preprocessor adds two columns to the dataframe"""
    tpot_obj = TPOT()

    input_df = training_testing_data
    output_df = tpot_obj._zero_count(input_df)

    assert type(input_df) == type(output_df)

    (in_rows, in_cols) = input_df.shape
    (out_rows, out_cols) = output_df.shape

    assert in_rows == out_rows
    assert in_cols == (out_cols - 2)


def test_ada_boost():
    """Ensure that the TPOT AdaBoostClassifier outputs the same as the sklearn AdaBoostClassifier"""

    tpot_obj = TPOT()
    result = tpot_obj._ada_boost(training_testing_data, 1.0)
    result = result[result['group'] == 'testing']

    adaboost = AdaBoostClassifier(n_estimators=500, random_state=42, learning_rate=1.0)
    adaboost.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, adaboost.predict(testing_features))


def test_ada_boost_2():
    """Ensure that the TPOT AdaBoostClassifier outputs the same as the sklearn classifer when learning_rate == 0.0"""

    tpot_obj = TPOT()
    result = tpot_obj._ada_boost(training_testing_data, 0.0)
    result = result[result['group'] == 'testing']

    adaboost = AdaBoostClassifier(n_estimators=500, random_state=42, learning_rate=0.0001)
    adaboost.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, adaboost.predict(testing_features))


def test_bernoulli_nb():
    """Ensure that the TPOT BernoulliNB outputs the same as the sklearn BernoulliNB"""

    tpot_obj = TPOT()
    result = tpot_obj._bernoulli_nb(training_testing_data, 1.0, 0.0)
    result = result[result['group'] == 'testing']

    bnb = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True)
    bnb.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, bnb.predict(testing_features))


def test_extra_trees():
    """Ensure that the TPOT ExtraTreesClassifier outputs the same as the sklearn ExtraTreesClassifier"""

    tpot_obj = TPOT()
    result = tpot_obj._extra_trees(training_testing_data, 0, 1., 0.1)
    result = result[result['group'] == 'testing']

    etc = ExtraTreesClassifier(n_estimators=500, random_state=42, max_features=1., min_weight_fraction_leaf=0.1, criterion='gini')
    etc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, etc.predict(testing_features))


def test_extra_trees_2():
    """Ensure that the TPOT ExtraTreesClassifier outputs the same as the sklearn version when max_features > 1"""

    tpot_obj = TPOT()
    result = tpot_obj._extra_trees(training_testing_data, 0, 2., 0.1)
    result = result[result['group'] == 'testing']

    etc = ExtraTreesClassifier(n_estimators=500, random_state=42, max_features=1., min_weight_fraction_leaf=0.1, criterion='gini')
    etc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, etc.predict(testing_features))


def test_extra_trees_3():
    """Ensure that the TPOT ExtraTreesClassifier outputs the same as the sklearn version when min_weight > 0.5"""
    tpot_obj = TPOT()

    result = tpot_obj._extra_trees(training_testing_data, 0, 1., 0.6)
    result = result[result['group'] == 'testing']

    etc = ExtraTreesClassifier(n_estimators=500, random_state=42, max_features=1., min_weight_fraction_leaf=0.5, criterion='gini')
    etc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, etc.predict(testing_features))


def test_gaussian_nb():
    """Ensure that the TPOT GaussianNB outputs the same as the sklearn GaussianNB"""

    tpot_obj = TPOT()
    result = tpot_obj._gaussian_nb(training_testing_data)
    result = result[result['group'] == 'testing']

    gnb = GaussianNB()
    gnb.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, gnb.predict(testing_features))


def test_multinomial_nb():
    """Ensure that the TPOT MultinomialNB outputs the same as the sklearn MultinomialNB"""

    tpot_obj = TPOT()
    result = tpot_obj._multinomial_nb(training_testing_data, 1.0)
    result = result[result['group'] == 'testing']

    mnb = MultinomialNB(alpha=1.0, fit_prior=True)
    mnb.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, mnb.predict(testing_features))


def test_linear_svc():
    """Ensure that the TPOT LinearSVC outputs the same as the sklearn LinearSVC"""

    tpot_obj = TPOT()
    result = tpot_obj._linear_svc(training_testing_data, 1.0, 0, True)
    result = result[result['group'] == 'testing']

    lsvc = LinearSVC(C=1.0, penalty='l1', dual=False, random_state=42)
    lsvc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, lsvc.predict(testing_features))


def test_linear_svc_2():
    """Ensure that the TPOT LinearSVC outputs the same as the sklearn LinearSVC when C == 0.0"""

    tpot_obj = TPOT()
    result = tpot_obj._linear_svc(training_testing_data, 0.0, 0, True)
    result = result[result['group'] == 'testing']

    lsvc = LinearSVC(C=0.0001, penalty='l1', dual=False, random_state=42)
    lsvc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, lsvc.predict(testing_features))


def test_passive_aggressive():
    """Ensure that the TPOT PassiveAggressiveClassifier outputs the same as the sklearn PassiveAggressiveClassifier"""

    tpot_obj = TPOT()
    result = tpot_obj._passive_aggressive(training_testing_data, 1.0, 0)
    result = result[result['group'] == 'testing']

    pagg = PassiveAggressiveClassifier(C=1.0, loss='hinge', fit_intercept=True, random_state=42)
    pagg.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, pagg.predict(testing_features))


def test_passive_aggressive_2():
    """Ensure that the TPOT PassiveAggressiveClassifier outputs the same as the sklearn classifier when C == 0.0"""

    tpot_obj = TPOT()
    result = tpot_obj._passive_aggressive(training_testing_data, 0.0, 0)
    result = result[result['group'] == 'testing']

    pagg = PassiveAggressiveClassifier(C=0.0001, loss='hinge', fit_intercept=True, random_state=42)
    pagg.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, pagg.predict(testing_features))


def test_gradient_boosting():
    """Ensure that the TPOT GradientBoostingClassifier outputs the same as the sklearn classifier"""

    tpot_obj = TPOT()
    result = tpot_obj._gradient_boosting(training_testing_data, 1.0, 1.0, 0.1)
    result = result[result['group'] == 'testing']

    gbc = GradientBoostingClassifier(learning_rate=1.0, max_features=1.0,
        min_weight_fraction_leaf=0.1, n_estimators=500, random_state=42)
    gbc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, gbc.predict(testing_features))


def test_gradient_boosting_2():
    """Ensure that the TPOT GradientBoostingClassifier outputs the same as the sklearn classifier when min_weight > 0.5"""

    tpot_obj = TPOT()
    result = tpot_obj._gradient_boosting(training_testing_data, 1.0, 1.0, 0.6)
    result = result[result['group'] == 'testing']

    gbc = GradientBoostingClassifier(learning_rate=1.0, max_features=1.0,
        min_weight_fraction_leaf=0.5, n_estimators=500, random_state=42)
    gbc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, gbc.predict(testing_features))


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
