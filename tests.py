"""
    Unit tests for TPOT.
"""

from tpot import TPOT
from tpot.export_utils import unroll_nested_fuction_calls, export_pipeline
from tpot.decorators import _gp_new_generation

from tpot.operators import Operator
from tpot.operators.classifiers import Classifier, TPOTDecisionTreeClassifier
from tpot.operators.preprocessors import Preprocessor
from tpot.operators.selectors import Selector

import pandas as pd
import numpy as np
from collections import Counter
import warnings
import inspect

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
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

non_feature_columns = ['guess', 'class', 'group']


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
    known_score = 0.9202817574915823  # Assumes use of the TPOT balanced_accuracy function

    # Reify pipeline with known score
    tpot_obj._optimized_pipeline = creator.Individual.\
        from_string('DecisionTreeClassifier(input_df, 0.5)', tpot_obj._pset)

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
        from_string('DecisionTreeClassifier(input_df, 0.5)', tpot_obj._pset)

    result = tpot_obj.predict(testing_features)

    assert result.shape == (testing_features.shape[0],)


def test_unroll_nested():
    """Assert that export utils' unroll_nested_fuction_calls outputs pipeline_list as expected"""

    tpot_obj = TPOT()

    expected_list = [['result1', 'LogisticRegression', 'input_df', '1.0', '0', 'True']]

    pipeline = creator.Individual.\
        from_string('LogisticRegression(input_df, 1.0, 0, True)', tpot_obj._pset)

    pipeline_list = unroll_nested_fuction_calls(pipeline)

    assert expected_list == pipeline_list


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

    result = op(training_testing_data, *args)

    clf = op._merge_with_default_params(op.preprocess_args(*args))
    clf.fit(training_features, training_classes)

    all_features = training_testing_data.drop(non_feature_columns, axis=1).values

    assert np.array_equal(result['guess'].values, clf.predict(all_features))


def check_selector(op):
    """Assert that a TPOT feature selector outputs the same as its sklearn counterpart"""
    tpot_obj = TPOT(random_state=42)

    prng = np.random.RandomState(42)

    args = []
    for type_ in op.parameter_types()[0][1:]:
        args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)

    result = op(training_testing_data, *args)

    sel = op._merge_with_default_params(op.preprocess_args(*args))
    training_features_df = training_testing_data.loc[training_testing_data['group'] == 'training'].\
        drop(non_feature_columns, axis=1)

    with warnings.catch_warnings():
        # Ignore warnings about constant features
        warnings.simplefilter('ignore', category=UserWarning)
        sel.fit(training_features_df, training_classes)

    mask = sel.get_support(True)
    mask_cols = list(training_features_df.iloc[:, mask].columns) + non_feature_columns

    assert np.array_equal(
        training_testing_data[mask_cols].drop(non_feature_columns, axis=1).values,
        result.drop(non_feature_columns, axis=1).values
    )


def check_preprocessor(op):
    """Assert that a TPOT feature preprocessor outputs the same as its sklearn counterpart"""
    tpot_obj = TPOT(random_state=42)

    prng = np.random.RandomState(42)

    args = []
    for type_ in op.parameter_types()[0][1:]:
        args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)

    result = op(training_testing_data, *args)

    prp = op._merge_with_default_params(op.preprocess_args(*args))
    prp.fit(training_features.astype(np.float64))
    all_features = training_testing_data.drop(non_feature_columns, axis=1).values
    sklearn_result = prp.transform(all_features.astype(np.float64))

    assert np.allclose(result.drop(non_feature_columns, axis=1).values, sklearn_result)


def test_operators():
    """Assert that the TPOT operators match the output of their sklearn counterparts"""
    for op in Operator.inheritors():
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


def test_operators_2():
    """Assert that TPOT operators return the input_df when no features are supplied"""
    assert np.array_equal(
        training_testing_data.ix[:, -3:],
        TPOTDecisionTreeClassifier()(training_testing_data.ix[:, -3:], 0.5)
    )


def test_export_pipeline():
    """Assert that TPOT's export utils outputs a pipeline as expected"""
    tpot_obj = TPOT(random_state=42)
    pipeline = creator.Individual.\
        from_string('ExtraTreesClassifier(PolynomialFeatures(input_df), 42, 0.93, 0.7)', tpot_obj._pset)

    expected_output = """import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)

exported_pipeline = Pipeline([
    ("PolynomialFeatures", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
    ("ExtraTreesClassifier", ExtraTreesClassifier(criterion="gini", max_features=0.93, min_weight_fraction_leaf=0.5, n_estimators=500))
])

exported_pipeline.fit(tpot_data.loc[training_indices].drop('class', axis=1).values,
                      tpot_data.loc[training_indices, 'class'].values)
results = exported_pipeline.predict(tpot_data.loc[testing_indices].drop('class', axis=1))
"""

    assert expected_output == export_pipeline(pipeline)


def test_export():
    """Assert that TPOT's export function throws a ValueError when no optimized pipeline exists"""
    tpot_obj = TPOT()

    try:
        tpot_obj.export("test.py")
        assert False  # Should be unreachable
    except ValueError:
        pass
