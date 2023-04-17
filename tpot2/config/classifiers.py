from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC

from functools import partial
#import GaussianNB

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB



def params_LogisticRegression(trial, name=None):
    params = {}
    params['solver'] = trial.suggest_categorical(name=f'solver_{name}',
                                                 choices=[f'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    params['dual'] = False
    params['penalty'] = 'l2'
    params['C'] = trial.suggest_float(f'C_{name}', 1e-4, 1e4, log=True)
    params['l1_ratio'] = None
    if params['solver'] == 'liblinear':
        params['penalty'] = trial.suggest_categorical(name=f'penalty_{name}', choices=['l1', 'l2'])
        if params['penalty'] == 'l2':
            params['dual'] = trial.suggest_categorical(name=f'dual_{name}', choices=[True, False])
        else:
            params['penalty'] = 'l1'

    params['class_weight'] = trial.suggest_categorical(name=f'class_weight_{name}', choices=['balanced'])
    param_grid = {'solver': params['solver'],
                  'penalty': params['penalty'],
                  'dual': params['dual'],
                  'multi_class': 'auto',
                  'l1_ratio': params['l1_ratio'],
                  'C': params['C'],
                  'n_jobs': 1,
                  }
    return param_grid


def params_KNeighborsClassifier(trial, name=None, n_samples=10):
    return {
        #'n_neighbors': trial.suggest_int(f'n_neighbors_{name}', 1, 20 ), #TODO: set as a function of the number of samples
        'n_neighbors': trial.suggest_int(f'n_neighbors_{name}', 1, n_samples, log=True ), #TODO: set as a function of the number of samples
        'weights': trial.suggest_categorical(f'weights_{name}', ['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 3),
        'metric': trial.suggest_categorical(f'metric_{name}', ['euclidean', 'minkowski']),
        'n_jobs': 1,
    }


def params_DecisionTreeClassifier(trial, name=None):
    return {
        'criterion': trial.suggest_categorical(f'criterion_{name}', ['gini', 'entropy']),
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 11),
        # 'max_depth_factor' : trial.suggest_float(f'max_depth_factor_{name}', 0, 2, step=0.1),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 21),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 21),
        'min_weight_fraction_leaf': 0.0,
        'max_features': trial.suggest_categorical(f'max_features_{name}', [ 'sqrt', 'log2']),
        'max_leaf_nodes': None,
    }


def params_SVC(trial, name=None):
    return {
        'kernel': trial.suggest_categorical(name=f'kernel_{name}', choices=['poly', 'rbf', 'linear', 'sigmoid']),
        'C': trial.suggest_float(f'C_{name}', 1e-4, 25, log=True),
        #'gamma': trial.suggest_categorical(name='fgamma_{name}', choices=['scale', 'auto']),
        'degree': trial.suggest_int(f'degree_{name}', 1, 4),
        'class_weight': trial.suggest_categorical(name=f'class_weight_{name}', choices=[None, 'balanced']),
        #'coef0': trial.suggest_float(f'coef0_{name}', 0, 10, step=0.1),
        'max_iter': 3000,
        'tol': 0.005,
        'probability': True,
    }


def params_LinearSVC(trial, name=None):
    penalty = trial.suggest_categorical(name=f'penalty_{name}', choices=['l1', 'l2'])
    if penalty == 'l1':
        loss = 'squared_hinge'
    else:
        loss = trial.suggest_categorical(name=f'loss_{name}', choices=['hinge', 'squared_hinge'])
    
    if loss == 'hinge' and penalty == 'l2':
        dual = True
    else:
        dual = trial.suggest_categorical(name=f'dual_{name}', choices=[True, False])

    return {
        'penalty': penalty,
        'loss': loss,
        'dual': dual, 
        'C': trial.suggest_float(f'C_{name}', 1e-4, 25, log=True),
    }


def params_RandomForestClassifier(trial, name=None):
    params = {
        'n_estimators': 100,
        'criterion': trial.suggest_categorical(name=f'criterion_{name}', choices=['gini', 'entropy']),
        #'max_features': trial.suggest_categorical('max_features_{name}', ['auto', 'sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical(name=f'bootstrap_{name}', choices=[True, False]),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 20),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 20),
        'n_jobs': 1,
    }
    return params


def params_GradientBoostingClassifier(trial, name=None):
    params = {
        'n_estimators': 100,
        'loss': trial.suggest_categorical(name=f'loss_{name}', choices=['log_loss', 'exponential']),
        'learning_rate': trial.suggest_float(f'learning_rate_{name}', 1e-3, 1, log=True),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 20),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 20),
        'subsample': trial.suggest_float(f'subsample_{name}', 0.1, 1.0),
        'max_features': trial.suggest_float(f'max_features_{name}', 0.1, 1.0),
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 10),
        'tol': 1e-4,
    }
    return params


def params_XGBClassifier(trial, name=None):
    return {
        'learning_rate': trial.suggest_float(f'learning_rate_{name}', 1e-3, 1, log=True),
        'subsample': trial.suggest_float(f'subsample_{name}', 0.1, 1.0),
        'min_child_weight': trial.suggest_int(f'min_child_weight_{name}', 1, 21),
        #'booster': trial.suggest_categorical(name='booster_{name}', choices=['gbtree', 'dart']),
        'n_estimators': 100,
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 11),
        'n_jobs': 1,
        #'use_label_encoder' : True,
    }


def params_LGBMClassifier(trial, name=None):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': trial.suggest_categorical(name=f'boosting_type_{name}', choices=['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int(f'num_leaves_{name}', 2, 256),
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 10),
        'n_estimators': trial.suggest_int(f'n_estimators_{name}', 10, 100),  # 200-6000 by 200
        'deterministic': True,
        'force_row_wise': True,
        'n_jobs': 1,
    }
    if 2 ** params['max_depth'] > params['num_leaves']:
        params['num_leaves'] = 2 ** params['max_depth']
    return params


def params_ExtraTreesClassifier(trial, name=None):
    params = {
        'n_estimators': 100,
        'criterion': trial.suggest_categorical(name=f'criterion_{name}', choices=["gini", "entropy"]),
        'max_features': trial.suggest_float('max_features', 0.05, 1.00),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 21,step=1),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 21, step=1),
        'bootstrap': trial.suggest_categorical(f'bootstrap_{name}', [True, False]),
        'n_jobs': 1,
    }
    return params

def params_SGDClassifier(trial, name=None):
    params = {
        'loss': trial.suggest_categorical(f'loss_{name}', ['log_loss', 'modified_huber',]),
        'penalty': 'elasticnet',
        'alpha': trial.suggest_float(f'alpha_{name}', 1e-5, 0.01, log=True),
        'learning_rate': trial.suggest_categorical(f'learning_rate_{name}', ['invscaling', 'constant']),
        'fit_intercept': trial.suggest_categorical(f'fit_intercept_{name}', [True, False]),
        'l1_ratio': trial.suggest_float(f'l1_ratio_{name}', 0.0, 1.0),
        'eta0': trial.suggest_float(f'eta0_{name}', 0.01, 1.0),
        'power_t': trial.suggest_float(f'power_t_{name}', 1e-5, 100.0, log=True),
        'n_jobs': 1,
    }

    return params

def params_MLPClassifier_tpot(trial, name=None):
    params = {
        'alpha': trial.suggest_float(f'alpha_{name}', 1e-4, 1e-1, log=True),
        'learning_rate_init': trial.suggest_float(f'learning_rate_init_{name}', 1e-3, 1., log=True)
    }

    return params

def params_MLPClassifier_large(trial, name=None):

    n_layers = trial.suggest_int(f'n_layers_{name}', 2, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_neurons_{i}_{name}', 4, 128))

    params = {
        'activation':  trial.suggest_categorical(name=f'activation_{name}', choices=['identity', 'logistic', 'tanh', 'relu']),
        'solver':  trial.suggest_categorical(name=f'solver_{name}', choices=['lbfgs', 'sgd', 'adam']),
        'alpha':  trial.suggest_float(f'alpha_{name}', 0.0001, 1.0, log=True),
        'hidden_layer_sizes':  tuple(layers),
        'max_iter' : 10000
    }

    return params  

def params_BernoulliNB(trial, name=None):
    params = {
        'alpha': trial.suggest_float(f'alpha_{name}', 1e-3, 100, log=True),
        'fit_prior': trial.suggest_categorical(f'fit_prior_{name}', [True, False]),
    }
    return params


def params_MultinomialNB(trial, name=None):
    params = {
        'alpha': trial.suggest_float(f'alpha_{name}', 1e-3, 100, log=True),
        'fit_prior': trial.suggest_categorical(f'fit_prior_{name}', [True, False]),
    }
    return params


def make_classifier_config_dictionary(n_samples=10):
    n_samples = min(n_samples,100) #TODO optimize this


    return {
            LogisticRegression: params_LogisticRegression,
            DecisionTreeClassifier: params_DecisionTreeClassifier,
            KNeighborsClassifier:  partial(params_KNeighborsClassifier,n_samples=n_samples),
            GradientBoostingClassifier: params_GradientBoostingClassifier,
            ExtraTreesClassifier:params_ExtraTreesClassifier,
            RandomForestClassifier: params_RandomForestClassifier,
            SGDClassifier:params_SGDClassifier,
            GaussianNB: {},
            BernoulliNB: params_BernoulliNB,
            MultinomialNB: params_MultinomialNB,
            XGBClassifier: params_XGBClassifier,
            #LinearSVC: params_LinearSVC,
            SVC: params_SVC,
            #: params_LGBMClassifier, # logistic regression and SVM/SVC are just special cases of this one? remove?
            MLPClassifier: params_MLPClassifier_tpot,
             
            
        }