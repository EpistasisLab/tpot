# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

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

import numpy as np

# Check the TPOT documentation for information on the structure of config dicts

resAdj_regressor_config_dict = {

    'tpot.builtins.resAdjElasticNetCV': {
        'l1_ratio': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },

    'tpot.builtins.resAdjExtraTreesRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'tpot.builtins.resAdjGradientBoostingRegressor': {
        'n_estimators': [100],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    'tpot.builtins.resAdjAdaBoostRegressor': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"]
    },

    'tpot.builtins.resAdjDecisionTreeRegressor': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'tpot.builtins.resAdjKNeighborsRegressor': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'tpot.builtins.resAdjLassoLarsCV': {
        'normalize': [True, False]
    },

    'tpot.builtins.resAdjLinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    'tpot.builtins.resAdjRandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'tpot.builtins.resAdjRidgeCV': {
    },

    'tpot.builtins.resAdjXGBRegressor': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1],
        'objective': ['reg:squarederror']
    },

    'tpot.builtins.resAdjSGDRegressor': {
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 0.01, 0.001] ,
        'learning_rate': ['invscaling', 'constant'] ,
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'eta0': [0.1, 1.0, 0.01],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    },

    # Preprocessors
    'tpot.builtins.identity': {
    },

    'tpot.builtins.resAdjBinarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'tpot.builtins.resAdjFastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'tpot.builtins.resAdjFeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'tpot.builtins.resAdjMaxAbsScaler': {
    },

    'tpot.builtins.resAdjMinMaxScaler': {
    },

    'tpot.builtins.resAdjNormalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'tpot.builtins.resAdjNystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'tpot.builtins.resAdjPCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'tpot.builtins.resAdjPolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'tpot.builtins.resAdjRBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'tpot.builtins.resAdjRobustScaler': {
    },

    'tpot.builtins.resAdjStandardScaler': {
    },

    'tpot.builtins.resAdjZeroCount': {
    },

    'tpot.builtins.resAdjOneHotEncoder': {
        'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
        'sparse': [False],
        'threshold': [10]
    },


    # Selectors
    'tpot.builtins.resAdjSelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'tpot.builtins.resAdjSelectPercentile': {
        'percentile': range(1, 101), # this range includes the identity transformation so to make a template including Selector more flexible
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'tpot.builtins.resAdjVarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'tpot.builtins.resAdjSelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}
