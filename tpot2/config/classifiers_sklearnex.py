from sklearnex.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearnex.neighbors import KNeighborsClassifier
from sklearnex.svm import SVC
from sklearnex.svm import NuSVC
from sklearnex.linear_model import LogisticRegression

from functools import partial


def params_RandomForestClassifier(trial, name=None):
    return {
        'n_estimators': 100,
        'bootstrap': trial.suggest_categorical(name=f'bootstrap_{name}', choices=[True, False]),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 20),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 20),
        'n_jobs': 1,
    }

def params_KNeighborsClassifier(trial, name=None, n_samples=10):
    n_neighbors_max = max(n_samples, 100)
    return {
        'n_neighbors': trial.suggest_int(f'n_neighbors_{name}', 1, n_neighbors_max, log=True ),
        'weights': trial.suggest_categorical(f'weights_{name}', ['uniform', 'distance']),
    }

def params_LogisticRegression(trial, name=None):
    params = {}
    params['dual'] = False
    params['penalty'] = 'l2'
    params['solver'] = trial.suggest_categorical(name=f'solver_{name}', choices=['liblinear', 'sag', 'saga']),
    if params['solver'] == 'liblinear':
        params['penalty'] = trial.suggest_categorical(name=f'penalty_{name}', choices=['l1', 'l2'])
        if params['penalty'] == 'l2':
            params['dual'] = trial.suggest_categorical(name=f'dual_{name}', choices=[True, False])
        else:
            params['penalty'] = 'l1'
    return {
        'solver': params['solver'],
        'penalty': params['penalty'],
        'dual': params['dual'],
        'C': trial.suggest_float(f'C_{name}', 1e-4, 1e4, log=True),
        'max_iter': 1000,
    }

def params_SVC(trial, name=None):
    return {
        'kernel': trial.suggest_categorical(name=f'kernel_{name}', choices=['poly', 'rbf', 'linear', 'sigmoid']),
        'C': trial.suggest_float(f'C_{name}', 1e-4, 25, log=True),
        'degree': trial.suggest_int(f'degree_{name}', 1, 4),
        'class_weight': trial.suggest_categorical(name=f'class_weight_{name}', choices=[None, 'balanced']),
        'max_iter': 3000,
        'tol': 0.005,
        'probability': True,
    }

def params_NuSVC(trial, name=None):
    return {
        'nu': trial.suggest_float(f'subsample_{name}', 0.05, 1.0),
        'kernel': trial.suggest_categorical(name=f'kernel_{name}', choices=['poly', 'rbf', 'linear', 'sigmoid']),
        'C': trial.suggest_float(f'C_{name}', 1e-4, 25, log=True),
        'degree': trial.suggest_int(f'degree_{name}', 1, 4),
        'class_weight': trial.suggest_categorical(name=f'class_weight_{name}', choices=[None, 'balanced']),
        'max_iter': 3000,
        'tol': 0.005,
        'probability': True,
    }

def params_XGBClassifier(trial, name=None):
    return {
        'learning_rate': trial.suggest_float(f'learning_rate_{name}', 1e-3, 1, log=True),
        'subsample': trial.suggest_float(f'subsample_{name}', 0.1, 1.0),
        'min_child_weight': trial.suggest_int(f'min_child_weight_{name}', 1, 21),
        'n_estimators': 100,
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 11),
        'n_jobs': 1,
    }

def make_sklearnex_classifier_config_dictionary(n_samples=10, n_classes=None):
    return {
            RandomForestClassifier: params_RandomForestClassifier,
            KNeighborsClassifier: params_KNeighborsClassifier,
            LogisticRegression: params_LogisticRegression,
            SVC: params_SVC,
            NuSVC: params_NuSVC,
            XGBClassifier: params_XGBClassifier,
        }
