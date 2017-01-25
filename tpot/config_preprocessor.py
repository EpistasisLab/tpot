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

preprocessor_config_dict = {

    'Binarizer': {
        'source': 'sklearn.preprocessing',
        'dependencies': None,
        'params':{
            'threshold': np.arange(0.0, 1.01, 0.05)
            }
    },

    'FastICA': {
        'source': 'sklearn.decomposition',
        'dependencies': None,
        'params':{
            'tol': np.arange(0.0, 1.01, 0.05)
            }
    },

    'FeatureAgglomeration': {
        'source': 'sklearn.cluster',
        'dependencies': None,
        'params':{
            'linkage': ['ward', 'complete', 'average'],
            'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
            }
    },

    'MaxAbsScaler': {
        'source': 'sklearn.preprocessing',
        'dependencies': None,
        'params': None
    },

    'MinMaxScaler': {
        'source': 'sklearn.preprocessing',
        'dependencies': None,
        'params': None
    },

    'Normalizer': {
        'source': 'sklearn.preprocessing',
        'dependencies': None,
        'params': {
            'norm': ['l1', 'l2', 'max']
            }
    },

    'Nystroem': {
        'source': 'sklearn.kernel_approximation',
        'dependencies': None,
        'params': {
            'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
            'gamma': np.arange(0.0, 1.01, 0.05),
            'n_components': range(1, 11)
            }
    },

    'PCA': {
        'source': 'sklearn.decomposition',
        'dependencies': None,
        'params': {
            'svd_solver': ['randomized'],
            'iterated_power': range(1, 11)
            }
    },

    'PolynomialFeatures': {
        'source': 'sklearn.preprocessing',
        'dependencies': None,
        'params': {
            'degree': [2],
            'include_bias': [False],
            'interaction_only': [False]
            }
    },

    'RBFSampler': {
        'source': 'sklearn.kernel_approximation',
        'dependencies': None,
        'params': {
            'gamma': np.arange(0.0, 1.01, 0.05)
            }
    },

    'RobustScaler': {
        'source': 'sklearn.preprocessing',
        'dependencies': None,
        'params': None
    },

    'StandardScaler': {
        'source': 'sklearn.preprocessing',
        'dependencies': None,
        'params': None
    }

    'ZeroCount': {
        'source': 'tpot.build_in_operators',
        'dependencies': None,
        'params': None
    }

}
