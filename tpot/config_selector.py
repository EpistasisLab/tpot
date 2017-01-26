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
    unique operator name
value:
    source: module source (e.g sklearn.tree)
    dependencies: depended module (e.g. SVC in selectors RFE); None for no dependency
    params: a dictionary of parameter names (keys) and parameter ranges (values); None for no params
"""
import numpy as np

selector_config_dict = {

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
                'sklearn.ensemble.ExtraTreesRegressor': {
                    'max_features': np.arange(0, 1.01, 0.05)
                    }
                }

    },

    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
            } # read from dependencies ! need add an exception in preprocess_args

    },

    'sklearn.feature_selection.SelectKBest': {
        'k': range(1, 100), # need check range!
        'score_func': {
            'sklearn.feature_selection.f_classif': None
            }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
            }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': np.arange(0, 1.01, 0.05)
    }

}

"""'TPOTRFE': {
    'source': 'sklearn.feature_selection.RFE',
    'dependencies': {
        'sklearn.svm.SVC': {
            'kernel': ['linear'],
            'random_state': [42]
            },
        'regression': False
        },
    'params':{
        'step': np.arange(0.1, 1.01, 0.05),
        'estimator': 'SVC(kernel=\'linear\', random_state=42)' # read from dependencies ! need add an exception in preprocess_args
        }
},"""


"""    'TPOTSelectFromModel': {
        'source': 'sklearn.feature_selection.SelectFromModel',
        'dependencies': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0, 1.01, 0.05)
                },
            'regression': False
            },
        'params':{
            'threshold': np.arange(0, 1.01, 0.05),
            'estimator': 'ExtraTreesClassifier(criterion=criterion_selection, max_features=max_features)' # read from dependencies ! need add an exception in preprocess_args
            }
    },
"""
