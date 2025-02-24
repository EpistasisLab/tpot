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
from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal



def get_RandomForestRegressor_ConfigurationSpace(random_state):
    space = {
        'n_estimators': 100,
        'max_features': Float("max_features", bounds=(0.05, 1.0)),
        'bootstrap': Categorical("bootstrap", [True, False]),
        'min_samples_split': Integer("min_samples_split", bounds=(2, 21)),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 21)),
    }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )


def get_KNeighborsRegressor_ConfigurationSpace(n_samples):
    return ConfigurationSpace(
        space = {
            'n_neighbors': Integer("n_neighbors", bounds=(1, max(n_samples, 100))),
            'weights': Categorical("weights", ['uniform', 'distance']),
        }
    )


def get_Ridge_ConfigurationSpace(random_state):
    space = {
        'alpha': Float("alpha", bounds=(0.0, 1.0)),
        'fit_intercept': Categorical("fit_intercept", [True]),
        'tol': Float("tol", bounds=(1e-5, 1e-1)),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

def get_Lasso_ConfigurationSpace(random_state):
    space = {
        'alpha': Float("alpha", bounds=(0.0, 1.0)),
        'fit_intercept': Categorical("fit_intercept", [True]),
        'precompute': Categorical("precompute", [True, False, 'auto']),
        'tol': 0.001,
        'positive': Categorical("positive", [True, False]),
        'selection': Categorical("selection", ['cyclic', 'random']),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

def get_ElasticNet_ConfigurationSpace(random_state):
    space = {
        'alpha': Float("alpha", bounds=(0.0, 1.0)),
        'l1_ratio': Float("l1_ratio", bounds=(0.0, 1.0)),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )


def get_SVR_ConfigurationSpace(random_state):
    space = {
        'kernel': Categorical("kernel", ['poly', 'rbf', 'linear', 'sigmoid']),
        'C': Float("C", bounds=(1e-4, 25), log=True),
        'degree': Integer("degree", bounds=(1, 4)),
        'max_iter': 3000,
        'tol': 0.001,
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

def get_NuSVR_ConfigurationSpace(random_state):
    space = {
        'nu': Float("nu", bounds=(0.05, 1.0)),
        'kernel': Categorical("kernel", ['poly', 'rbf', 'linear', 'sigmoid']),
        'C': Float("C", bounds=(1e-4, 25), log=True),
        'degree': Integer("degree", bounds=(1, 4)),
        'max_iter': 3000,
        'tol': 0.005,
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )