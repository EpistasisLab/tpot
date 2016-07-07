"""
    Unit tests for TPOT.
"""

from tpot import TPOT
from tpot.export_utils import unroll_nested_fuction_calls
from tpot.decorators import _gp_new_generation

from tpot.operators import Operator
from tpot.operators.classifiers import Classifier

import pandas as pd
import numpy as np
from collections import Counter

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from deap import creator
from tqdm import tqdm

np.random.seed(42)

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


def test_classifiers():
    """Assert that the TPOT classifiers match the output of their sklearn counterparts"""
    tpot_obj = TPOT(random_state=42)

    print()

    for op in Operator.inheritors():
        print("Assert that the TPOT {} classifier matches the output of the sklearn counterpart ... ".format(op.sklearn_class.__name__), end="")
        if not isinstance(op, Classifier):
            continue

        args = []
        for type_ in op.parameter_types()[0][1:]:
            args.append(np.random.choice(tpot_obj._pset.terminals[type_]).value)

        result = op(training_testing_data, *args)

        clf = op._merge_with_default_params(op.preprocess_args(*args))
        clf.fit(training_features, training_classes)

        assert np.array_equal(result['guess'].values, clf.predict(testing_features))

        print("ok")
