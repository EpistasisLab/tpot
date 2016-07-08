"""
    Unit tests for TPOT.
"""

from tpot import TPOT
from tpot.export_utils import unroll_nested_fuction_calls, export_pipeline
from tpot.decorators import _gp_new_generation

from tpot.operators import Operator
from tpot.operators.classifiers import Classifier
from tpot.operators.preprocessors import Preprocessor
from tpot.operators.selectors import Selector

import pandas as pd
import numpy as np
from collections import Counter

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


def check_array_equal(arr1, arr2):
    """Assert that the TPOT foo operator outputs the same as the sklearn counterpart"""
    assert np.allclose(arr1, arr2)


def test_classifiers():
    """Assert that the TPOT classifiers match the output of their sklearn counterparts"""
    tpot_obj = TPOT(random_state=42)

    for op in Operator.inheritors():
        if not isinstance(op, Classifier):
            continue

        prng = np.random.RandomState(42)

        args = []
        for type_ in op.parameter_types()[0][1:]:
            args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)

        result = op(training_testing_data, *args)

        clf = op._merge_with_default_params(op.preprocess_args(*args))
        clf.fit(training_features, training_classes)

        all_features = training_testing_data.drop(non_feature_columns, axis=1).values

        check_array_equal.description = ("Assert that the TPOT {} operator matches "
                                         "the output of the sklearn counterpart".
                                         format(op.__name__))
        yield check_array_equal, result['guess'].values, clf.predict(all_features)


# def test_preprocessors():
#     """Assert that the TPOT preprocessors match the output of their sklearn counterparts"""
#     tpot_obj = TPOT(random_state=42)
#
#     for op in Operator.inheritors():
#         if not isinstance(op, Preprocessor):
#             continue
#
#         prng = np.random.RandomState(42)
#
#         args = []
#         for type_ in op.parameter_types()[0][1:]:
#             args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)
#
#         result = op(training_testing_data, *args)
#
#         prp = op._merge_with_default_params(op.preprocess_args(*args))
#         prp.fit(training_features.astype(np.float64))
#         all_features = training_testing_data.drop(non_feature_columns, axis=1).values
#         sklearn_result = prp.transform(all_features.astype(np.float64))
#
#         check_array_equal.description = ("Assert that the TPOT {} operator matches "
#                                          "the output of the sklearn counterpart".
#                                          format(op.__name__))
#         yield check_array_equal, result.drop(non_feature_columns, axis=1).values, sklearn_result


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
