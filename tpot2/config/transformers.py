from functools import partial
import numpy as np

from tpot2.builtin_modules import ZeroCount, OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import FastICA
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


def params_sklearn_preprocessing_Binarizer(trial, name=None):
    return {
        'threshold': trial.suggest_float(f'threshold_{name}', 0.0, 1.0),
    }

def params_sklearn_decomposition_FastICA(trial, random_state=None, name=None, n_features=100):
    return {
        'n_components': trial.suggest_int(f'n_components_{name}', 1, n_features), # number of components wrt number of features
        'algorithm': trial.suggest_categorical(f'algorithm_{name}', ['parallel', 'deflation']),
        'whiten':'unit-variance',
        'random_state': random_state
    }

def params_sklearn_cluster_FeatureAgglomeration(trial, name=None, n_features=100):

    linkage = trial.suggest_categorical(f'linkage_{name}', ['ward', 'complete', 'average'])
    if linkage == 'ward':
        metric = 'euclidean'
    else:
        metric = trial.suggest_categorical(f'metric_{name}', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'])
    return {
        'linkage': linkage,
        'metric': metric,
        'n_clusters': trial.suggest_int(f'n_clusters_{name}', 2, n_features-1), #TODO perhaps a percentage of n_features
    }

def params_sklearn_preprocessing_Normalizer(trial, name=None):
    return {
        'norm': trial.suggest_categorical(f'norm_{name}', ['l1', 'l2', 'max']),
    }

def params_sklearn_kernel_approximation_Nystroem(trial, random_state=None, name=None, n_features=100):
    return {
        'gamma': trial.suggest_float(f'gamma_{name}', 0.0, 1.0),
        'kernel': trial.suggest_categorical(f'kernel_{name}', ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid']),
        'n_components': trial.suggest_int(f'n_components_{name}', 1, n_features),
        'random_state': random_state
    }

def params_sklearn_decomposition_PCA(trial, random_state=None, name=None, n_features=100):
    # keep the number of components required to explain 'variance_explained' of the variance
    variance_explained = 1.0 - trial.suggest_float(f'n_components_{name}', 0.001, 0.5, log=True) #values closer to 1 are more likely

    return {
        'n_components': variance_explained,
        'random_state': random_state
    }

def params_sklearn_kernel_approximation_RBFSampler(trial, random_state=None, name=None, n_features=100):
    return {
        'n_components': trial.suggest_int(f'n_components_{name}', 1, n_features),
        'gamma': trial.suggest_float(f'gamma_{name}', 0.0, 1.0),
        'random_state': random_state
    }

def params_tpot_builtins_ZeroCount(trial, name=None):

    return {}

def params_tpot_builtins_OneHotEncoder(trial, name=None):

    return {}

def make_transformer_config_dictionary(random_state=None, n_features=10):
    #n_features = min(n_features,100) #TODO optimize this
    return {
                Binarizer: params_sklearn_preprocessing_Binarizer,
                FastICA: partial(params_sklearn_decomposition_FastICA, random_state=random_state, n_features=n_features),
                FeatureAgglomeration: partial(params_sklearn_cluster_FeatureAgglomeration,n_features=n_features),
                MaxAbsScaler: {},
                MinMaxScaler: {},
                Normalizer: params_sklearn_preprocessing_Normalizer,
                Nystroem: partial(params_sklearn_kernel_approximation_Nystroem, random_state=random_state, n_features=n_features),
                PCA: partial(params_sklearn_decomposition_PCA, random_state=random_state, n_features=n_features),
                PolynomialFeatures: {
                                        'degree': 2,
                                        'include_bias': False,
                                        'interaction_only': False,
                                    },
                RBFSampler: partial(params_sklearn_kernel_approximation_RBFSampler, random_state=random_state, n_features=n_features),
                RobustScaler: {},
                StandardScaler: {},
                ZeroCount: params_tpot_builtins_ZeroCount,
                OneHotEncoder: params_tpot_builtins_OneHotEncoder,
            }
