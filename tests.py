"""
    Unit tests for TPOT.
"""

from tpot import TPOT

import pandas as pd
import numpy as np
from collections import Counter
import random

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

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
    """Ensure that the TPOT instantiator stores the TPOT variables properly"""

    tpot_obj = TPOT(population_size=500, generations=1000,
                    mutation_rate=0.05, crossover_rate=0.9, verbosity=1)

    assert tpot_obj.population_size == 500
    assert tpot_obj.generations == 1000
    assert tpot_obj.mutation_rate == 0.05
    assert tpot_obj.crossover_rate == 0.9
    assert tpot_obj.verbosity == 1

def test_decision_tree():
    """Ensure that the TPOT decision tree method outputs the same as the sklearn decision tree"""

    tpot_obj = TPOT()
    result = tpot_obj._decision_tree(training_testing_data, 0, 0)
    result = result[result['group'] == 'testing']

    dtc = DecisionTreeClassifier(max_features='auto', max_depth=None, random_state=42)
    dtc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, dtc.predict(testing_features))

def test_random_forest():
    """Ensure that the TPOT random forest method outputs the same as the sklearn random forest"""

    tpot_obj = TPOT()
    result = tpot_obj._random_forest(training_testing_data, 100, 0)
    result = result[result['group'] == 'testing']

    rfc = RandomForestClassifier(n_estimators=100, max_features='auto', random_state=42, n_jobs=-1)
    rfc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, rfc.predict(testing_features))

def test_xgboost():
    """Ensure that the TPOT xgboost method outputs the same as the xgboost classfier method"""

    tpot_obj = TPOT()
    result = tpot_obj._xgradient_boosting(training_testing_data, n_estimators=100, learning_rate=0, max_depth=3)
    result = result[result['group'] == 'testing']

    xgb = XGBClassifier(n_estimators=100, learning_rate=0.0001, max_depth=3, seed=42)
    xgb.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, xgb.predict(testing_features))

def test_combine_dfs():
    tpot_obj = TPOT()

    df1 = pd.DataFrame({'a': list(range(10)),
                        'b': list(range(10, 20))})

    df2 = pd.DataFrame({'b': list(range(10, 20)),
                        'c': list(range(20, 30))})

    combined_df = pd.DataFrame({'a': list(range(10)),
                                'b': list(range(10, 20)),
                                'c': list(range(20, 30))})

    assert tpot_obj._combine_dfs(df1, df2).equals(combined_df)

def test_static_models():
    """Ensure that the TPOT classifiers output the same predictions as the sklearn output"""
    tpot_obj = TPOT()
    models = [(tpot_obj._decision_tree, DecisionTreeClassifier, {'max_features':0, 'max_depth':0}, {'max_features':'auto', 'max_depth':None}),
              (tpot_obj._svc, SVC , {'C':0.0001}, {'C':0.0001}),
              (tpot_obj._random_forest, RandomForestClassifier,{'n_estimators':100, 'max_features':0}, {'n_estimators':100, 'max_features':'auto', 'n_jobs':-1}),
              (tpot_obj._logistic_regression, LogisticRegression, {'C':0.0001}, {'C':0.0001}),
              (tpot_obj._knnc, KNeighborsClassifier, {'n_neighbors':100}, {'n_neighbors':100})]

    for model, sklearn_model, model_params, sklearn_params in models:

        result = model(training_testing_data, **model_params)
        try:
            sklearn_model_obj = sklearn_model(random_state=42, **sklearn_params)
            sklearn_model_obj.fit(training_features, training_classes)
        except TypeError:
            sklearn_model_obj = sklearn_model(**sklearn_params)
            sklearn_model_obj.fit(training_features, training_classes)

        result = result[result['group'] == 'testing']

        assert np.array_equal(result['guess'].values, sklearn_model_obj.predict(testing_features)), "Model {} failed".format(str(model))

def test_consensus():
    tpot_obj = TPOT()

    df1 = pd.DataFrame({'00001': [2,2,2,2,2,2,2,2,2,2],
                        '00002': [3,3,3,3,3,3,3,3,3,3],
                        'class': [1,1,1,1,1,0,0,0,0,0],
                        'guess': [1,0,1,0,1,0,1,0,1,0]})

    df2 = pd.DataFrame({'00001': [2,2,2,2,2,2,2,2,2,2],
                        '00002': [3,3,3,3,3,3,3,3,3,3],
                        'class': [1,1,1,1,1,0,0,0,0,0],
                        'guess': [0,0,0,0,0,1,1,1,1,1]})

    df3 = pd.DataFrame({'00001': [2,2,2,2,2,2,2,2,2,2],
                        '00002': [3,3,3,3,3,3,3,3,3,3],
                        'class': [1,1,1,1,1,0,0,0,0,0],
                        'guess': [0,1,0,1,0,1,0,1,0,1]})

    df4 = pd.DataFrame({'00001': [2,2,2,2,2,2,2,2,2,2],
                        '00002': [3,3,3,3,3,3,3,3,3,3],
                        'class': [1,1,1,1,1,0,0,0,0,0],
                        'guess': [1,1,0,0,1,1,0,0,1,1]})

    uniform = [1.0] * 10
    adaboost = [1.0] * 10

    accuracy1 = 0.60
    accuracy2 = 0.0
    accuracy3 = 0.40
    accuracy4 = 0.50
    res_two_acc_max = pd.DataFrame({'class': [1,1,1,1,1,0,0,0,0,0],
                                    'guess': [1,0,1,0,1,0,1,0,1,0]})
    res_two_acc_mean = pd.DataFrame({'class': [1,1,1,1,1,0,0,0,0,0],
                                     'guess': [1,0,1,0,1,0,1,0,1,0]})
    res_two_acc_med = pd.DataFrame({'class': [1,1,1,1,1,0,0,0,0,0],
                                    'guess': [1,0,1,0,1,0,1,0,1,0]})
    res_two_acc_min = pd.DataFrame({'class': [1,1,1,1,1,0,0,0,0,0],
                                    'guess': [0,0,0,0,0,1,1,1,1,1]})
    res_two_uni_max = pd.DataFrame({'class': [1,1,1,1,1,0,0,0,0,0],
                                    'guess': [1,0,1,0,1,0,1,0,1,0]})
    res_two_ada_max = pd.DataFrame({'class': [1,1,1,1,1,0,0,0,0,0],
                                    'guess': [1,0,1,0,1,0,1,0,1,0]})
    res_three_ada_max = pd.DataFrame({'class': [1,1,1,1,1,0,0,0,0,0],
                                      'guess': [1,1,0,0,1,1,0,0,1,1]})
    res_four_ada_max = pd.DataFrame({'class': [1,1,1,1,1,0,0,0,0,0],
                                     'guess': [1,0,0,0,1,1,1,0,1,1]})
    
    tests = [(tpot_obj._consensus_two, 0, 2, df1, df2), #accuracy, max
             (tpot_obj._consensus_two, 0, 3, df1, df2), #accuracy mean
             (tpot_obj._consensus_two, 0, 4, df1, df2), # accuracy median
             (tpot_obj._consensus_two, 0, 5, df1, df2), # accuracy min
             (tpot_obj._consensus_two, 1, 2, df1, df2), # uniform max
             #(tpot_obj._consensus_two, 2, 3, df1, df2), # adaboost max
             (tpot_obj._consensus_three, 1, 2, df1, df3, df4), # uni max
             (tpot_obj._consensus_four, 1, 2, df1, df2, df3, df4) #uni  max
             ]
    results = [res_two_acc_max,
           res_two_acc_mean,
           res_two_acc_med,
           res_two_acc_min,
           res_two_uni_max,
           #res_two_ada_max,
           res_three_ada_max,
           res_four_ada_max
           ]

    for test, res in zip(tests, results):
        #print(test[1:3])
        #print(test[0](*test[1:])['guess'].values)
        assert np.array_equal(test[0](*test[1:])['guess'].values, res['guess'].values)

