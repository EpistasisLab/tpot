'''
    Unit tests for TPOT.
'''

from tpot import TPOT

import pandas as pd
import numpy as np
from collections import Counter
import random

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Set up the iris data set for testing
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
    '''
        Ensure that the TPOT instantiator stores the TPOT variables properly.
    '''

    tpot_obj = TPOT(population_size=500, generations=1000,
                    mutation_rate=0.05, crossover_rate=0.9, verbosity=1)

    assert tpot_obj.population_size == 500
    assert tpot_obj.generations == 1000
    assert tpot_obj.mutation_rate == 0.05
    assert tpot_obj.crossover_rate == 0.9
    assert tpot_obj.verbosity == 1

def test_decision_tree():
    '''
        Ensure that the TPOT decision tree method outputs the same as the sklearn decision tree.
    '''

    tpot_obj = TPOT()
    result = tpot_obj.decision_tree(training_testing_data, 0, 0)
    result = result[result['group'] == 'testing']

    dtc = DecisionTreeClassifier(max_features='auto', max_depth=None, random_state=42)
    dtc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, dtc.predict(testing_features))

def test_random_forest():
    '''
        Ensure that the TPOT random forest method outputs the same as the sklearn random forest.
    '''

    tpot_obj = TPOT()
    result = tpot_obj.random_forest(training_testing_data, 100, 0)
    result = result[result['group'] == 'testing']

    rfc = RandomForestClassifier(n_estimators=100, max_features='auto', random_state=42, n_jobs=-1)
    rfc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, rfc.predict(testing_features))

def test_combine_dfs():
    tpot_obj = TPOT()

    df1 = pd.DataFrame({'a': range(10),
                        'b': range(10, 20)})

    df2 = pd.DataFrame({'b': range(10, 20),
                        'c': range(20, 30)})

    combined_df = pd.DataFrame({'a': range(10),
                                'b': range(10, 20),
                                'c': range(20, 30)})

    assert tpot_obj._combine_dfs(df1, df2).equals(combined_df)

def test_df_feature_selection():
    tpot_obj = TPOT()

    top_10_feature_pairs = ['00002', '00013', '00020', '00021', '00026', '00042',
                            '00043', '00058', '00061', 'class', 'group', 'guess']

    assert np.array_equal(tpot_obj._dt_feature_selection(training_testing_data, 10).columns.values,
                          top_10_feature_pairs)
