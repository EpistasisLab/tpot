"""
Copyright 2016 Randal S. Olson

This file is part of the TPOT library.

The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along
with the TPOT library. If not, see http://www.gnu.org/licenses/.

"""

"""
dictionary format (json-like format):
key:
    operator name
value:
    source: module source (e.g sklearn.tree)
    dependencies: depended module (e.g. SVC in selectors RFE); None for no dependency
    params: a dictionary of parameter names (keys) and parameter ranges (values); None for no params
"""
import numpy as np

regressor_config_dict = {


    'ElasticNetCV': {
        'source': 'sklearn.linear_model',
        'dependencies': None,
        'params':{
            'l1_ratio': np.arange(0.0, 1.01, 0.05),
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            }
    },

    'ExtraTreesRegressor': {
        'source': 'sklearn.ensemble',
        'dependencies': None,
        'params':{
            'max_features': np.arange(0, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
            }
    },

    'GradientBoostingRegressor': {
        'source': 'sklearn.ensemble',
        'dependencies': None,
        'params':{
            'loss': ["ls", "lad", "huber", "quantile"],
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'subsample': np.arange(0.05, 1.01, 0.05),
            'max_features': np.arange(0, 1.01, 0.05),
            'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
            }
    },

    'AdaBoostRegressor': {
        'source': 'sklearn.ensemble',
        'dependencies': None,
        'params':{
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'loss': ["linear", "square", "exponential"],
            'max_depth': range(1, 11)
            }
    },

    'DecisionTreeRegressor': {
        'source': 'sklearn.tree',
        'dependencies': None,
        'params':{
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21)
            }
    },

    'KNeighborsRegressor': {
        'source': 'sklearn.neighbors',
        'dependencies': None,
        'params':{
            'n_neighbors': range(1, 101),
            'weights': ["uniform", "distance"],
            'p': [1, 2]
            }
    },

    'LassoLarsCV': {
        'source': 'sklearn.linear_model',
        'dependencies': None,
        'params':{
            'normalize': [True, False]
            }
    },

    'LinearSVR': {
        'source': 'sklearn.svm',
        'dependencies': None,
        'params':{
            'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
            'dual': [True, False],
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
            }
    },

    'RandomForestRegressor': {
        'source': 'sklearn.ensemble',
        'dependencies': None,
        'params':{
            'max_features': np.arange(0, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
            }
    },

    'RidgeCV': {
        'source': 'sklearn.linear_model',
        'dependencies': None,
        'params': None
    },


    'XGBRegressor': {
        'source': 'xgboost',
        'dependencies': None,
        'params':{
            'max_depth': range(1, 11),
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'subsample': np.arange(0.05, 1.01, 0.05),
            'min_child_weight': range(1, 21)
            }
    }


}
