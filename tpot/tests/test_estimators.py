"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Gabriel Ketron (gabriel.ketron@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
import pytest
import tpot
from sklearn.datasets import load_iris
import random
import sklearn

@pytest.fixture
def sample_dataset():
    X_train, y_train = load_iris(return_X_y=True)
    return X_train, y_train

#standard test
@pytest.fixture
def tpot_estimator():

    n_classes=3
    n_samples=100
    n_features=100

    search_space = tpot.search_spaces.pipelines.GraphSearchPipeline(
            root_search_space= tpot.config.get_search_space("classifiers", n_samples=n_samples, n_features=n_features, n_classes=n_classes),
            leaf_search_space = None, 
            inner_search_space = tpot.config.get_search_space(["selectors","transformers"],n_samples=n_samples, n_features=n_features, n_classes=n_classes),
            max_size = 10,
        )
    return tpot.TPOTEstimator(  
                            search_space=search_space,
                            population_size=10,
                            generations=2,
                            scorers=['roc_auc_ovr'],
                            scorers_weights=[1],
                            classification=True,
                            n_jobs=4, 
                            early_stop=5,
                            other_objective_functions= [],
                            other_objective_functions_weights=[],
                            max_time_mins=20/60,
                            verbose=3)

@pytest.fixture
def tpot_classifier():
    return tpot.tpot_estimator.templates.TPOTClassifier(max_time_mins=10/60,verbose=0)

@pytest.fixture
def tpot_regressor():
    return tpot.tpot_estimator.templates.TPOTRegressor(max_time_mins=10/60,verbose=0)


@pytest.fixture
def tpot_estimator_with_pipeline(tpot_estimator,sample_dataset):
    tpot_estimator.fit(sample_dataset[0], sample_dataset[1])
    return tpot_estimator

def test_tpot_estimator_predict(tpot_estimator_with_pipeline,sample_dataset):
    #X_test = [[1, 2, 3], [4, 5, 6]]
    X_test = sample_dataset[0]
    y_pred = tpot_estimator_with_pipeline.predict(X_test)
    assert len(y_pred) == len(X_test)
    assert tpot_estimator_with_pipeline.fitted_pipeline_ is not None

def test_tpot_estimator_generations_type():
    with pytest.raises(TypeError):
        tpot.TPOTEstimator(generations="two", population_size=10, verbosity=2)

def test_tpot_estimator_population_size_type():
    with pytest.raises(TypeError):
        tpot.TPOTEstimator(generations=2, population_size='ten', verbosity=2)

def test_tpot_estimator_verbosity_type():
    with pytest.raises(TypeError):
        tpot.TPOTEstimator(generations=2, population_size=10, verbosity='high')

def test_tpot_estimator_scoring_type():
    with pytest.raises(TypeError):
        tpot.TPOTEstimator(generations=2, population_size=10, verbosity=2, scoring=0.5)

def test_tpot_estimator_cv_type():
    with pytest.raises(TypeError):
        tpot.TPOTEstimator(generations=2, population_size=10, verbosity=2, cv='kfold')

def test_tpot_estimator_n_jobs_type():
    with pytest.raises(TypeError):
        tpot.TPOTEstimator(generations=2, population_size=10, verbosity=2, n_jobs='all')

def test_tpot_estimator_config_dict_type():
    with pytest.raises(TypeError):
        tpot.TPOTEstimator(generations=2, population_size=10, verbosity=2, config_dict='config')





def test_tpot_classifier_fit(tpot_classifier,sample_dataset):
    #load iris dataset
    X_train = sample_dataset[0]
    y_train = sample_dataset[1]
    tpot_classifier.fit(X_train, y_train)
    assert tpot_classifier.fitted_pipeline_ is not None

def test_tpot_regressor_fit(tpot_regressor):

    scorer = sklearn.metrics.get_scorer('neg_mean_squared_error')
    X, y = sklearn.datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.05, test_size=0.95)
    tpot_regressor.fit(X_train, y_train)
    assert tpot_regressor.fitted_pipeline_ is not None

