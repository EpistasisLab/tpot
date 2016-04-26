"""
    Unit tests for TPOT.
"""

from tpot import TPOT

import pandas as pd
import numpy as np
from collections import Counter
import random
import warnings

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, SelectPercentile, f_classif


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
                    mutation_rate=0.05, crossover_rate=0.9, verbosity=1, disable_update_check=True, scoring_function="_balanced_accuracy")

    assert tpot_obj.population_size == 500
    assert tpot_obj.generations == 1000
    assert tpot_obj.mutation_rate == 0.05
    assert tpot_obj.crossover_rate == 0.9
    assert tpot_obj.verbosity == 1
    assert tpot_obj.update_checked == True
    assert tpot_obj.scoring_function == "_balanced_accuracy"

def test_decision_tree():
    """Ensure that the TPOT decision tree method outputs the same as the sklearn decision tree"""

    tpot_obj = TPOT()
    result = tpot_obj._decision_tree(training_testing_data, 0, 0)
    result = result[result['group'] == 'testing']

    dtc = DecisionTreeClassifier(max_features='auto', max_depth=None, random_state=42)
    dtc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, dtc.predict(testing_features))

def test_decision_tree_2():
    """Ensure that the TPOT decision tree method outputs the same as the sklearn decision tree when max_features=1"""

    tpot_obj = TPOT()
    result = tpot_obj._decision_tree(training_testing_data, 1, 0)
    result = result[result['group'] == 'testing']

    dtc = DecisionTreeClassifier(max_features=None, max_depth=None, random_state=42)
    dtc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, dtc.predict(testing_features))

def test_decision_tree_3():
    """Ensure that the TPOT decision tree method outputs the same as the sklearn decision tree when max_features>no. of features"""

    tpot_obj = TPOT()
    result = tpot_obj._decision_tree(training_testing_data, 100, 0)
    result = result[result['group'] == 'testing']

    dtc = DecisionTreeClassifier(max_features=64, max_depth=None, random_state=42)
    dtc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, dtc.predict(testing_features))

def test_random_forest():
    """Ensure that the TPOT random forest method outputs the same as the sklearn random forest when max_features<1"""

    tpot_obj = TPOT()
    result = tpot_obj._random_forest(training_testing_data, 0)
    result = result[result['group'] == 'testing']

    rfc = RandomForestClassifier(n_estimators=500, max_features='auto', random_state=42, n_jobs=-1)
    rfc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, rfc.predict(testing_features))

def test_random_forest_2():
    """Ensure that the TPOT random forest method outputs the same as the sklearn random forest when max_features=1"""

    tpot_obj = TPOT()
    result = tpot_obj._random_forest(training_testing_data, 1)
    result = result[result['group'] == 'testing']

    rfc = RandomForestClassifier(n_estimators=500, max_features=None, random_state=42, n_jobs=-1)
    rfc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, rfc.predict(testing_features))

def test_random_forest_3():
    """Ensure that the TPOT random forest method outputs the same as the sklearn random forest when max_features>no. of features"""

    tpot_obj = TPOT()
    result = tpot_obj._random_forest(training_testing_data, 100)
    result = result[result['group'] == 'testing']

    rfc = RandomForestClassifier(n_estimators=500, max_features=64, random_state=42, n_jobs=-1)
    rfc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, rfc.predict(testing_features))

def test_svc():
    """Ensure that the TPOT random forest method outputs the same as the sklearn svc when C>0.0001"""

    tpot_obj = TPOT()
    result = tpot_obj._svc(training_testing_data, 1.0)
    result = result[result['group'] == 'testing']

    svc = SVC(C=1.0, random_state=42)
    svc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, svc.predict(testing_features))

def test_svc_2():
    """Ensure that the TPOT random forest method outputs the same as the sklearn svc when C<0.0001"""

    tpot_obj = TPOT()
    result = tpot_obj._svc(training_testing_data, 0.00001)
    result = result[result['group'] == 'testing']

    svc = SVC(C=0.0001, random_state=42)
    svc.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, svc.predict(testing_features))

def test_xgboost():
    """Ensure that the TPOT xgboost method outputs the same as the xgboost classfier method"""

    tpot_obj = TPOT()
    result = tpot_obj._xgradient_boosting(training_testing_data, learning_rate=0, max_depth=3)
    result = result[result['group'] == 'testing']

    xgb = XGBClassifier(n_estimators=500, learning_rate=0.0001, max_depth=3, seed=42)
    xgb.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, xgb.predict(testing_features))

def test_xgboost_2():
    """Ensure that the TPOT xgboost method outputs the same as the xgboost classfier method when max_depth<1"""

    tpot_obj = TPOT()
    result = tpot_obj._xgradient_boosting(training_testing_data, learning_rate=0, max_depth=0)
    result = result[result['group'] == 'testing']

    xgb = XGBClassifier(n_estimators=500, learning_rate=0.0001, max_depth=None, seed=42)
    xgb.fit(training_features, training_classes)

    assert np.array_equal(result['guess'].values, xgb.predict(testing_features))

def test_train_model_and_predict():
    """Ensure that the TPOT train_model_and_predict returns the input dataframe when it has only 3 columns i.e. class, group, guess"""

    tpot_obj = TPOT()

    assert np.array_equal(training_testing_data.ix[:,-3:],tpot_obj._train_model_and_predict(training_testing_data.ix[:,-3:], SVC))

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

        assert np.array_equal(tpot_obj._rfe(training_testing_data.ix[:,-3:], 0, 0.1),training_testing_data.ix[:,-3:])

def test_rfe_2():
    """Ensure that the TPOT RFE outputs the same result as the sklearn rfe when num_features>no. of features in the dataframe """
    tpot_obj = TPOT()

    non_feature_columns = ['class', 'group', 'guess']
    training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
    estimator = SVC(kernel='linear')
    rfe = RFE(estimator, 100, step=0.1)
    rfe.fit(training_features, training_classes)
    mask = rfe.get_support(True)
    mask_cols = list(training_features.iloc[:,mask].columns) + non_feature_columns

    assert np.array_equal(training_testing_data[mask_cols],tpot_obj._rfe(training_testing_data, 64, 0.1))

def test_select_percentile():
        """Ensure that the TPOT select percentile outputs the input dataframe when no. of training features is 0"""
        tpot_obj = TPOT()

        assert np.array_equal(tpot_obj._select_percentile(training_testing_data.ix[:,-3:], 0),training_testing_data.ix[:,-3:])

def test_select_percentile_2():
        """Ensure that the TPOT select percentile outputs the same result as sklearn Select Percentile when percentile is 0"""
        tpot_obj = TPOT()
        non_feature_columns = ['class', 'group', 'guess']
        training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
        training_class_vals = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values
        #percentile = max(min(100, percentile), 0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            selector = SelectPercentile(f_classif, percentile=0)
            selector.fit(training_features, training_class_vals)
            mask = selector.get_support(True)
        mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

        assert np.array_equal(tpot_obj._select_percentile(training_testing_data, 0), training_testing_data[mask_cols])

def test_select_percentile_3():
        tpot_obj = TPOT()
        non_feature_columns = ['class', 'group', 'guess']
        training_features = training_testing_data.loc[training_testing_data['group'] == 'training'].drop(non_feature_columns, axis=1)
        training_class_vals = training_testing_data.loc[training_testing_data['group'] == 'training', 'class'].values
        #percentile = max(min(100, percentile), 0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            selector = SelectPercentile(f_classif, percentile=100)
            selector.fit(training_features, training_class_vals)
            mask = selector.get_support(True)
        mask_cols = list(training_features.iloc[:, mask].columns) + non_feature_columns

        assert np.array_equal(tpot_obj._select_percentile(training_testing_data, 120), training_testing_data[mask_cols])

def test_select_kbest():
        """Ensure that the TPOT select kbest outputs the input dataframe when no. of training features is 0"""
        tpot_obj = TPOT()

        assert np.array_equal(tpot_obj._select_kbest(training_testing_data.ix[:,-3:], 1),training_testing_data.ix[:,-3:])

def test_select_fwe():
        """Ensure that the TPOT select fwe outputs the input dataframe when no. of training features is 0"""
        tpot_obj = TPOT()

        assert np.array_equal(tpot_obj._select_fwe(training_testing_data.ix[:,-3:], 0.005),training_testing_data.ix[:,-3:])

def test_standard_scaler():
        """Ensure that the TPOT standard scaler outputs the input dataframe when no. of training features is 0"""
        tpot_obj = TPOT()

        assert np.array_equal(tpot_obj._standard_scaler(training_testing_data.ix[:,-3:]),training_testing_data.ix[:,-3:])

def test_robust_scaler():
        """Ensure that the TPOT robust scaler outputs the input dataframe when no. of training features is 0"""
        tpot_obj = TPOT()

        assert np.array_equal(tpot_obj._robust_scaler(training_testing_data.ix[:,-3:]),training_testing_data.ix[:,-3:])

def test_polynomial_features():
        """Ensure that the TPOT polynomial features outputs the input dataframe when no. of training features is 0"""
        tpot_obj = TPOT()

        assert np.array_equal(tpot_obj._polynomial_features(training_testing_data.ix[:,-3:]),training_testing_data.ix[:,-3:])

def test_min_max_scaler():
        """Ensure that the TPOT min max scaler outputs the input dataframe when no. of training features is 0"""
        tpot_obj = TPOT()

        assert np.array_equal(tpot_obj._min_max_scaler(training_testing_data.ix[:,-3:]),training_testing_data.ix[:,-3:])

def test_max_abs_scaler():
        """Ensure that the TPOT max abs scaler outputs the input dataframe when no. of training features is 0"""
        tpot_obj = TPOT()

        assert np.array_equal(tpot_obj._max_abs_scaler(training_testing_data.ix[:,-3:]),training_testing_data.ix[:,-3:])

def test_static_models():
    """Ensure that the TPOT classifiers output the same predictions as the sklearn output"""
    tpot_obj = TPOT()
    models = [(tpot_obj._decision_tree, DecisionTreeClassifier, {'max_features':0, 'max_depth':0}, {'max_features':'auto', 'max_depth':None}),
              (tpot_obj._svc, SVC , {'C':0.0001}, {'C':0.0001}),
              (tpot_obj._random_forest, RandomForestClassifier,{'max_features':0}, {'n_estimators':500, 'max_features':'auto', 'n_jobs':-1}),
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

def test_div():
        """Ensure that the TPOT protected division function outputs 0 when the divisor is 0"""

        tpot_obj = TPOT()
        assert tpot_obj._div(5,0)==0

def test_binarizer():
        """Ensure that the TPOT binarizer outputs the input dataframe when no. of training features is 0"""
        tpot_obj = TPOT()

        assert np.array_equal(tpot_obj._binarizer(training_testing_data.ix[:,-3:], 0),training_testing_data.ix[:,-3:])

def test_pca():
        """Ensure that the TPOT PCA outputs the input dataframe when no. of training features is 0"""
        tpot_obj = TPOT()

        assert np.array_equal(tpot_obj._pca(training_testing_data.ix[:,-3:], 1, 1),training_testing_data.ix[:,-3:])