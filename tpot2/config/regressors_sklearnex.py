from sklearnex.linear_model import LinearRegression
from sklearnex.linear_model import Ridge
from sklearnex.linear_model import Lasso
from sklearnex.linear_model import ElasticNet

from sklearnex.svm import SVR
from sklearnex.svm import NuSVR

from sklearnex.ensemble import RandomForestRegressor
from sklearnex.neighbors import KNeighborsRegressor


def params_RandomForestRegressor(trial, name=None):
    return {
        'n_estimators': 100,
        'max_features': trial.suggest_float(f'max_features_{name}', 0.05, 1.0),
        'bootstrap': trial.suggest_categorical(name=f'bootstrap_{name}', choices=[True, False]),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 21),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 21),
    }

def params_KNeighborsRegressor(trial, name=None, n_samples=100):
    n_neighbors_max = max(n_samples, 100)
    return {
        'n_neighbors': trial.suggest_int(f'n_neighbors_{name}', 1, n_neighbors_max),
        'weights': trial.suggest_categorical(f'weights_{name}', ['uniform', 'distance']),
        }

def params_LinearRegression(trial, name=None):
    return {}

def params_Ridge(trial, name=None):
    return {
        'alpha': trial.suggest_float(f'alpha_{name}', 0.0, 1.0),
        'fit_intercept': True,
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True),
    }

def params_Lasso(trial, name=None):
    return {
        'alpha': trial.suggest_float(f'alpha_{name}', 0.0, 1.0),
        'fit_intercept': True,
        'precompute': trial.suggest_categorical(f'precompute_{name}', [True, False, 'auto']),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True),
        'positive': trial.suggest_categorical(f'positive_{name}', [True, False]),
        'selection': trial.suggest_categorical(f'selection_{name}', ['cyclic', 'random']),
    }

def params_ElasticNet(trial, name=None):
    return {
        'alpha': 1 - trial.suggest_float(f'alpha_{name}', 0.0, 1.0),
        'l1_ratio': 1- trial.suggest_float(f'l1_ratio_{name}',0.0, 1.0),
        }

def params_SVR(trial, name=None):
    return {
        'kernel': trial.suggest_categorical(name=f'kernel_{name}', choices=['poly', 'rbf', 'linear', 'sigmoid']),
        'C': trial.suggest_float(f'C_{name}', 1e-4, 25, log=True),
        'degree': trial.suggest_int(f'degree_{name}', 1, 4),
        'max_iter': 3000,
        'tol': 0.005,
    }

def params_NuSVR(trial, name=None):
    return {
        'nu': trial.suggest_float(f'subsample_{name}', 0.05, 1.0),
        'kernel': trial.suggest_categorical(name=f'kernel_{name}', choices=['poly', 'rbf', 'linear', 'sigmoid']),
        'C': trial.suggest_float(f'C_{name}', 1e-4, 25, log=True),
        'degree': trial.suggest_int(f'degree_{name}', 1, 4),
        'max_iter': 3000,
        'tol': 0.005,
    }

def make_sklearnex_regressor_config_dictionary(n_samples=10):
    return {
        RandomForestRegressor: params_RandomForestRegressor,
        KNeighborsRegressor: params_KNeighborsRegressor,
        LinearRegression: params_LinearRegression,
        Ridge: params_Ridge,
        Lasso: params_Lasso,
        ElasticNet: params_ElasticNet,
        SVR: params_SVR,
        NuSVR: params_NuSVR,
    }
