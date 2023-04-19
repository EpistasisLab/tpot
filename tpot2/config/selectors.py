#TODO: how to best support transformers/selectors that take other transformers with their own hyperparameters? 
import numpy as np
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
import sklearn.feature_selection
from functools import partial
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

from .classifiers import params_ExtraTreesClassifier
from .regressors import params_ExtraTreesRegressor

def params_sklearn_feature_selection_SelectFwe(trial, name=None):
    return {
        'alpha': trial.suggest_float(f'alpha_{name}', 1e-4, 0.05, log=True),
        'score_func' : sklearn.feature_selection.f_classif,
    }

def params_sklearn_feature_selection_SelectPercentile(trial, name=None):
    return {
        'percentile': trial.suggest_float(f'percentile_{name}', 1, 100.0),
        'score_func' : sklearn.feature_selection.f_classif,
    }

def params_sklearn_feature_selection_VarianceThreshold(trial, name=None):
    return {
        'threshold': trial.suggest_float(f'threshold_{name}', 1e-4, .2, log=True)
    }
    

#TODO add more estimator options? How will that interact with optuna?
def params_sklearn_feature_selection_RFE(trial, name=None, classifier=True):
    if classifier:
        estimator = ExtraTreesClassifier(**params_ExtraTreesClassifier(trial, name=f"RFE_{name}"))
    else:
        estimator = ExtraTreesRegressor(**params_ExtraTreesRegressor(trial, name=f"RFE_{name}"))
    
    params = {
            'step': trial.suggest_float(f'step_{name}', 1e-4, 1.0, log=False),
            'estimator' : estimator,
            }

    return params


def params_sklearn_feature_selection_SelectFromModel(trial, name=None, classifier=True):
    if classifier:
        estimator = ExtraTreesClassifier(**params_ExtraTreesClassifier(trial, name=f"SFM_{name}"))
    else:
        estimator = ExtraTreesRegressor(**params_ExtraTreesRegressor(trial, name=f"SFM_{name}"))
    
    params = {
            'threshold': trial.suggest_float(f'threshold_{name}', 1e-4, 1.0, log=True),
            'estimator' : estimator,
            }

    return params



def make_selector_config_dictionary(classifier=True):
    return {
                SelectFwe: params_sklearn_feature_selection_SelectFwe,
                SelectPercentile: params_sklearn_feature_selection_SelectPercentile,
                VarianceThreshold: params_sklearn_feature_selection_VarianceThreshold,
                RFE: partial(params_sklearn_feature_selection_RFE, classifier=classifier),
                SelectFromModel: partial(params_sklearn_feature_selection_SelectFromModel, classifier=classifier),
            }