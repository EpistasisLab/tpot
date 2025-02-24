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


def get_RandomForestClassifier_ConfigurationSpace(random_state, n_jobs=1):
    space = {
            'n_estimators': 100, #TODO make this a higher number? learned?
            'bootstrap': Categorical("bootstrap", [True, False]),
            'min_samples_split': Integer("min_samples_split", bounds=(2, 20)),
            'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 20)),
            'n_jobs': n_jobs,
            
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )

def get_KNeighborsClassifier_ConfigurationSpace(n_samples):
    return ConfigurationSpace(
        space = {
            'n_neighbors': Integer("n_neighbors", bounds=(1, max(n_samples, 100)), log=True),
            'weights': Categorical("weights", ['uniform', 'distance']),
        }
    )


#TODO add conditionals
def get_LogisticRegression_ConfigurationSpace(random_state):
    space = {
        'solver': Categorical("solver", ['liblinear', 'sag', 'saga']),
        'penalty': Categorical("penalty", ['l1', 'l2']),
        'dual': Categorical("dual", [True, False]),
        'C': Float("C", bounds=(1e-4, 1e4), log=True),
        'max_iter': 1000,
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

def get_SVC_ConfigurationSpace(random_state):
    space = {
        'kernel': Categorical("kernel", ['poly', 'rbf', 'linear', 'sigmoid']),
        'C': Float("C", bounds=(1e-4, 25), log=True),
        'degree': Integer("degree", bounds=(1, 4)),
        'max_iter': 3000,
        'tol': 0.001,
        'probability': Categorical("probability", [True]), # configspace doesn't allow bools as a default value? but does allow them as a value inside a Categorical
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

def get_NuSVC_ConfigurationSpace(random_state):
    space = {
        'nu': Float("nu", bounds=(0.05, 1.0)),
        'kernel': Categorical("kernel", ['poly', 'rbf', 'linear', 'sigmoid']),
        #'C': Float("C", bounds=(1e-4, 25), log=True),
        'degree': Integer("degree", bounds=(1, 4)),
        'class_weight': Categorical("class_weight", [None, 'balanced']),
        'max_iter': 3000,
        'tol': 0.005,
        'probability': Categorical("probability", [True]), # configspace doesn't allow bools as a default value? but does allow them as a value inside a Categorical
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )