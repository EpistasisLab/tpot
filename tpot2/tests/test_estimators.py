import pytest
import tpot2
from sklearn.datasets import load_iris
import random
import sklearn

#standard test
@pytest.fixture
def tpot_estimator():
    return tpot2.TPOTEstimator(  population_size=50,
                            scorers=['roc_auc_ovr'],
                            scorers_weights=[1],
                            classification=True,
                            n_jobs=1, 
                            early_stop=5,
                            other_objective_functions= [],
                            other_objective_functions_weights=[],
                            max_time_seconds=300,
                            verbose=3)

@pytest.fixture
def sample_dataset():
    X_train, y_train = load_iris(return_X_y=True)
    return X_train, y_train

def test_tpot_estimator_fit(tpot_estimator,sample_dataset):
    #load iris dataset
    X_train = sample_dataset[0]
    y_train = sample_dataset[1]
    tpot_estimator.fit(X_train, y_train)
    assert tpot_estimator.fitted_pipeline_ is not None

@pytest.fixture
def tpot_estimator_with_pipeline(tpot_estimator,sample_dataset):
    tpot_estimator.fit(sample_dataset[0], sample_dataset[1])
    return tpot_estimator

def test_tpot_estimator_predict(tpot_estimator_with_pipeline,sample_dataset):
    #X_test = [[1, 2, 3], [4, 5, 6]]
    X_test = sample_dataset[0]
    y_pred = tpot_estimator_with_pipeline.predict(X_test)
    assert len(y_pred) == len(X_test)

def test_tpot_estimator_score(tpot_estimator_with_pipeline,sample_dataset):
    random.seed(42)
    #random sample 10% of the dataset
    X_test = random.sample(list(sample_dataset[0]), int(len(sample_dataset[0])*0.1))
    y_test = random.sample(list(sample_dataset[1]), int(len(sample_dataset[1])*0.1))
    scorer = sklearn.metrics.get_scorer('roc_auc_ovo')
    assert isinstance(scorer(tpot_estimator_with_pipeline, X_test, y_test), float)

def test_tpot_estimator_generations_type():
    with pytest.raises(TypeError):
        tpot2.TPOTEstimator(generations="two", population_size=10, verbosity=2)

def test_tpot_estimator_population_size_type():
    with pytest.raises(TypeError):
        tpot2.TPOTEstimator(generations=2, population_size='ten', verbosity=2)

def test_tpot_estimator_verbosity_type():
    with pytest.raises(TypeError):
        tpot2.TPOTEstimator(generations=2, population_size=10, verbosity='high')

def test_tpot_estimator_scoring_type():
    with pytest.raises(TypeError):
        tpot2.TPOTEstimator(generations=2, population_size=10, verbosity=2, scoring=0.5)

def test_tpot_estimator_cv_type():
    with pytest.raises(TypeError):
        tpot2.TPOTEstimator(generations=2, population_size=10, verbosity=2, cv='kfold')

def test_tpot_estimator_n_jobs_type():
    with pytest.raises(TypeError):
        tpot2.TPOTEstimator(generations=2, population_size=10, verbosity=2, n_jobs='all')

def test_tpot_estimator_config_dict_type():
    with pytest.raises(TypeError):
        tpot2.TPOTEstimator(generations=2, population_size=10, verbosity=2, config_dict='config')



@pytest.fixture
def tpot_classifier():
    return tpot2.tpot_estimator.templates.TPOTClassifier(max_time_seconds=300,verbose=3)

@pytest.fixture
def tpot_regressor():
    return tpot2.tpot_estimator.templates.TPOTRegressor(max_time_seconds=300,verbose=3)

def test_tpot_classifier_fit(tpot_classifier,sample_dataset):
    #load iris dataset
    X_train = sample_dataset[0]
    y_train = sample_dataset[1]
    tpot_classifier.fit(X_train, y_train)
    assert tpot_classifier.fitted_pipeline_ is not None

def test_tpot_regressor_fit(tpot_regressor):

    scorer = sklearn.metrics.get_scorer('neg_mean_squared_error')
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.75, test_size=0.25)
    tpot_regressor.fit(X_train, y_train)
    assert tpot_regressor.fitted_pipeline_ is not None

