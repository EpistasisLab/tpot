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
    alias: scikit-learn operator name, available for duplicted key in 'selector_config_dict' dictionary
    operator name
value:
    source: module source (e.g sklearn.tree)
    dependencies: depended module (e.g. SVC in selectors RFE); None for no dependency
    params: a dictionary of parameter names (keys) and parameter ranges (values); None for no params
"""
import numpy as np

selector_config_dict = {
    'RFE': {
        'source': 'sklearn.feature_selection',
        'dependencies': {
            'sklearn.svm.SVC': {
                'kernel': ['linear'],
                'random_state': [42]
                }
            'classification': True
            'regression': False
            },
        'params':{
            'step': np.arange(0.1, 1.01, 0.05),
            'estimator': 'sklearn.svm.SVC' # read from dependencies ! need add an exception in preprocess_args
            }
    },

    'SelectFromModel_R': {
        'alias': 'SelectFromModel', # need add an exception for this case
        'source': 'sklearn.feature_selection',
        'dependencies': {
            'sklearn.ensemble.ExtraTreesRegressor': {
                'max_features': np.arange(0, 1.01, 0.05)
                }
            'classification': False
            'regression': True
            },
        'params':{
            'threshold': np.arange(0, 1.01, 0.05),
            'estimator': 'sklearn.ensemble.ExtraTreesRegressor' # read from dependencies ! need add an exception in preprocess_args
            }
    },

    'SelectFromModel': {
        'source': 'sklearn.feature_selection',
        'dependencies': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0, 1.01, 0.05)
                }
            'classification': True
            'regression': False
            },
        'params':{
            'threshold': np.arange(0, 1.01, 0.05),
            'estimator': 'sklearn.ensemble.ExtraTreesRegressor' # read from dependencies ! need add an exception in preprocess_args
            }
    },

    'SelectFwe': {
        'source': 'sklearn.feature_selection',
        'dependencies': {
            'sklearn.feature_selection.f_classif': None
            'classification': True
            'regression': True
            },
        'params':{
            'alpha': np.arange(0, 0.05, 0.001),
            'score_func': 'sklearn.feature_selection.f_classif' # read from dependencies ! need add an exception in preprocess_args
            }
    },

    'SelectKBest': {
        'source': 'sklearn.feature_selection',
        'dependencies': {
            'sklearn.feature_selection.f_classif': None
            'classification': True
            'regression': True
            },
        'params':{
            'k': range(1, 100), # need check range!
            'score_func': 'sklearn.feature_selection.f_classif' # read from dependencies ! need add an exception in preprocess_args
            }
    },

    'SelectPercentile': {
        'source': 'sklearn.feature_selection',
        'dependencies': {
            'sklearn.feature_selection.f_classif': None
            'classification': True
            'regression': True
            },
        'params':{
            'percentile': range(1, 100),
            'score_func': 'sklearn.feature_selection.f_classif' # read from dependencies ! need add an exception in preprocess_args
            }
    },

    'VarianceThreshold': {
        'source': 'sklearn.feature_selection',
        'dependencies': None
        'params':{
            'threshold': np.arange(0, 0.05, 0.001)
            }
    }

}
