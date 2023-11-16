from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars, LassoLarsCV
from sklearn.linear_model import RidgeCV


from sklearn.svm import SVR
from sklearn.svm import LinearSVR

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNetCV

from xgboost import XGBRegressor
from functools import partial




#TODO: fill in remaining
#TODO check for places were we could use log scaling

def params_RandomForestRegressor(trial, random_state=None, name=None):
    return {
        'n_estimators': 100,
        'max_features': trial.suggest_float(f'max_features_{name}', 0.05, 1.0),
        'bootstrap': trial.suggest_categorical(name=f'bootstrap_{name}', choices=[True, False]),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 21),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 21),
        'random_state': random_state
    }


# SGDRegressor parameters
def params_SGDRegressor(trial, random_state=None, name=None):
    params = {
        'loss': trial.suggest_categorical(f'loss_{name}', ['huber', 'squared_error', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
        'penalty': 'elasticnet',
        'alpha': trial.suggest_float(f'alpha_{name}', 1e-5, 0.01, log=True),
        'learning_rate': trial.suggest_categorical(f'learning_rate_{name}', ['invscaling', 'constant']),
        'fit_intercept':True,
        'l1_ratio': trial.suggest_float(f'l1_ratio_{name}', 0.0, 1.0),
        'eta0': trial.suggest_float(f'eta0_{name}', 0.01, 1.0),
        'power_t': trial.suggest_float(f'power_t_{name}', 1e-5, 100.0, log=True),
        'random_state': random_state

    }
    return params

# Ridge parameters
def params_Ridge(trial, random_state=None, name=None):
    params = {
        'alpha': trial.suggest_float(f'alpha_{name}', 0.0, 1.0),
        'fit_intercept': True,


        #'max_iter': trial.suggest_int(f'max_iter_{name}', 100, 1000),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True),
        'solver': trial.suggest_categorical(f'solver_{name}', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
        'random_state': random_state
    }
    return params


# Lasso parameters
def params_Lasso(trial, random_state=None, name=None):
    params = {
        'alpha': trial.suggest_float(f'alpha_{name}', 0.0, 1.0),
        'fit_intercept': True,
        # 'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False]),
        'precompute': trial.suggest_categorical(f'precompute_{name}', [True, False, 'auto']),

        #'max_iter': trial.suggest_int(f'max_iter_{name}', 100, 1000),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True),

        'positive': trial.suggest_categorical(f'positive_{name}', [True, False]),
        'selection': trial.suggest_categorical(f'selection_{name}', ['cyclic', 'random']),
        'random_state': random_state
    }
    return params

# ElasticNet parameters
def params_ElasticNet(trial, random_state=None, name=None):
    params = {
        'alpha': 1 - trial.suggest_float(f'alpha_{name}', 0.0, 1.0, log=True),
        'l1_ratio': 1- trial.suggest_float(f'l1_ratio_{name}',0.0, 1.0),
        'random_state': random_state
        }
    return params

# Lars parameters
def params_Lars(trial, random_state=None, name=None):
    params = {
        'fit_intercept': True,
        'verbose': trial.suggest_categorical(f'verbose_{name}', [True, False]),
        'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False]),

        # 'precompute': trial.suggest_categorical(f'precompute_{name}', ['auto_{name}', True, False]),
        'n_nonzero_coefs': trial.suggest_int(f'n_nonzero_coefs_{name}', 1, 100),
        'eps': trial.suggest_float(f'eps_{name}', 1e-5, 1e-1, log=True),
        'copy_X': trial.suggest_categorical(f'copy_X_{name}', [True, False]),
        'fit_path': trial.suggest_categorical(f'fit_path_{name}', [True, False]),
        # 'positive': trial.suggest_categorical(f'positive_{name}', [True, False]),
        'random_state': random_state
    }
    return params

# OrthogonalMatchingPursuit parameters
def params_OrthogonalMatchingPursuit(trial, name=None):
    params = {
        'n_nonzero_coefs': trial.suggest_int(f'n_nonzero_coefs_{name}', 1, 100),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True),
        'fit_intercept': True,
        'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False]),
        'precompute': trial.suggest_categorical(f'precompute_{name}', ['auto', True, False]),
    }
    return params

# BayesianRidge parameters
def params_BayesianRidge(trial, name=None):
    params = {
        'n_iter': trial.suggest_int(f'n_iter_{name}', 100, 1000),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True),
        'alpha_1': trial.suggest_float(f'alpha_1_{name}', 1e-6, 1e-1, log=True),
        'alpha_2': trial.suggest_float(f'alpha_2_{name}', 1e-6, 1e-1, log=True),
        'lambda_1': trial.suggest_float(f'lambda_1_{name}', 1e-6, 1e-1, log=True),
        'lambda_2': trial.suggest_float(f'lambda_2_{name}', 1e-6, 1e-1, log=True),
        'compute_score': trial.suggest_categorical(f'compute_score_{name}', [True, False]),
        'fit_intercept': True,
        'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False]),
        'copy_X': trial.suggest_categorical(f'copy_X_{name}', [True, False]),
    }
    return params

# LassoLars parameters
def params_LassoLars(trial, random_state=None, name=None):
    params = {
        'alpha': trial.suggest_float(f'alpha_{name}', 0.0, 1.0),
        # 'fit_intercept': True,
        # 'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False]),
        # 'precompute': trial.suggest_categorical(f'precompute_{name}', ['auto_{name}', True, False]),
        #'max_iter': trial.suggest_int(f'max_iter_{name}', 100, 1000),
        'eps': trial.suggest_float(f'eps_{name}', 1e-5, 1e-1, log=True),
        # 'copy_X': trial.suggest_categorical(f'copy_X_{name}', [True, False]),
        # 'positive': trial.suggest_categorical(f'positive_{name}', [True, False]),
        'random_state': random_state
    }
    return params

# LassoLars parameters
def params_LassoLarsCV(trial, cv, name=None):
    params = {
        'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False]),
        'cv': cv,
    }
    return params

# BaggingRegressor parameters
def params_BaggingRegressor(trial, random_state=None, name=None):
    params = {
        'n_estimators': trial.suggest_int(f'n_estimators_{name}', 10, 100),
        'max_samples': trial.suggest_float(f'max_samples_{name}', 0.05, 1.00),
        'max_features': trial.suggest_float(f'max_features_{name}', 0.05, 1.00),
        'bootstrap': trial.suggest_categorical(f'bootstrap_{name}', [True, False]),
        'bootstrap_features': trial.suggest_categorical(f'bootstrap_features_{name}', [True, False]),
        'random_state': random_state
    }
    return params

# ARDRegression parameters
def params_ARDRegression(trial, name=None):
    params = {
        'n_iter': trial.suggest_int(f'n_iter_{name}', 100, 1000),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True),
        'alpha_1': trial.suggest_float(f'alpha_1_{name}', 1e-6, 1e-1, log=True),
        'alpha_2': trial.suggest_float(f'alpha_2_{name}', 1e-6, 1e-1, log=True),
        'lambda_1': trial.suggest_float(f'lambda_1_{name}', 1e-6, 1e-1, log=True),
        'lambda_2': trial.suggest_float(f'lambda_2_{name}', 1e-6, 1e-1, log=True),
        'compute_score': trial.suggest_categorical(f'compute_score_{name}', [True, False]),
        'threshold_lambda': trial.suggest_int(f'threshold_lambda_{name}', 100, 1000),
        'fit_intercept': True,
        'normalize': trial.suggest_categorical(f'normalize_{name}', [True, False]),
        'copy_X': trial.suggest_categorical(f'copy_X_{name}', [True, False]),
    }
    return params



# TheilSenRegressor parameters
def params_TheilSenRegressor(trial, random_state=None, name=None):
    params = {
        'n_subsamples': trial.suggest_int(f'n_subsamples_{name}', 10, 100),
        'max_subpopulation': trial.suggest_int(f'max_subpopulation_{name}', 100, 1000),
        'fit_intercept': True,
        'copy_X': trial.suggest_categorical(f'copy_X_{name}', [True, False]),
        'verbose': trial.suggest_categorical(f'verbose_{name}', [True, False]),
        'random_state': random_state
    }
    return params


# SVR parameters
def params_SVR(trial, name=None):
    params = {
        'kernel': trial.suggest_categorical(name=f'kernel_{name}', choices=['poly', 'rbf', 'linear', 'sigmoid']),
        'C': trial.suggest_float(f'C_{name}', 1e-4, 25, log=True),
        'degree': trial.suggest_int(f'degree_{name}', 1, 4),
        'max_iter': 3000,
        'tol': 0.005,
    }
    return params

# Perceptron parameters
def params_Perceptron(trial, random_state=None, name=None):
    params = {
        'penalty': trial.suggest_categorical(f'penalty_{name}', [None, 'l2', 'l1', 'elasticnet']),
        'alpha': trial.suggest_float(f'alpha_{name}', 1e-5, 1e-1, log=True),
        'l1_ratio': trial.suggest_float(f'l1_ratio_{name}', 0.0, 1.0),
        'fit_intercept': True,
        #'max_iter': trial.suggest_int(f'max_iter_{name}', 100, 1000),
        'tol': trial.suggest_float(f'tol_{name}', 1e-5, 1e-1, log=True),
        'shuffle': trial.suggest_categorical(f'shuffle_{name}', [True, False]),
        'verbose': trial.suggest_categorical(f'verbose_{name}', [0, 1, 2, 3, 4, 5]),
        'eta0': trial.suggest_float(f'eta0_{name}', 1e-5, 1e-1, log=True),
        'learning_rate': trial.suggest_categorical(f'learning_rate_{name}', ['constant', 'optimal', 'invscaling']),
        'early_stopping': trial.suggest_categorical(f'early_stopping_{name}', [True, False]),
        'validation_fraction': trial.suggest_float(f'validation_fraction_{name}', 0.05, 1.00),
        'n_iter_no_change': trial.suggest_int(f'n_iter_no_change_{name}', 1, 100),
        'class_weight': trial.suggest_categorical(f'class_weight_{name}', [None, 'balanced']),
        'warm_start': trial.suggest_categorical(f'warm_start_{name}', [True, False]),
        'average': trial.suggest_categorical(f'average_{name}', [True, False]),
        'random_state': random_state
    }
    return params

def params_MLPRegressor(trial, random_state=None, name=None):
    params = {
        'alpha': trial.suggest_float(f'alpha_{name}', 1e-4, 1e-1, log=True),
        'learning_rate_init': trial.suggest_float(f'learning_rate_init_{name}', 1e-3, 1., log=True),
        'random_state': random_state
    }

    return params


#GradientBoostingRegressor parameters
def params_GradientBoostingRegressor(trial, random_state=None, name=None):
    loss = trial.suggest_categorical(f'loss_{name}', ['ls', 'lad', 'huber', 'quantile'])

    params = {

        'n_estimators': 100,
        'loss': loss,
        'learning_rate': trial.suggest_float(f'learning_rate_{name}', 1e-4, 1, log=True),
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 11),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 21),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 21),
        'subsample': 1-trial.suggest_float(f'subsample_{name}', 0.05, 1.00, log=True),
        'max_features': 1-trial.suggest_float(f'max_features_{name}', 0.05, 1.00, log=True),
        'random_state': random_state

    }

    if loss == 'quantile' or loss == 'huber':
        alpha = trial.suggest_float(f'alpha_{name}', 0.05, 0.95)
        params['alpha'] = alpha

    return params



def params_DecisionTreeRegressor(trial, random_state=None, name=None):
    params = {
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1,11),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 21),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 21),
        # 'criterion': trial.suggest_categorical(f'criterion_{name}', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
        # 'splitter': trial.suggest_categorical(f'splitter_{name}', ['best', 'random']),
        #'max_features': trial.suggest_categorical(f'max_features_{name}', [None, 'auto', 'sqrt', 'log2']),
        #'ccp_alpha': trial.suggest_float(f'ccp_alpha_{name}',  1e-1, 10.0),
        'random_state': random_state

    }
    return params

def params_KNeighborsRegressor(trial, name=None, n_samples=100):
    params = {
        'n_neighbors': trial.suggest_int(f'n_neighbors_{name}', 1, n_samples, log=True ),
        'weights': trial.suggest_categorical(f'weights_{name}', ['uniform', 'distance']),
        'p': trial.suggest_int(f'p_{name}', 1, 3),
        'metric': trial.suggest_categorical(f'metric_{name}', ['minkowski', 'euclidean', 'manhattan']),

        }
    return params

def params_LinearSVR(trial, random_state=None, name=None):
    params = {
        'epsilon': trial.suggest_float(f'epsilon_{name}', 1e-4, 1.0, log=True),
        'C': trial.suggest_float(f'C_{name}', 1e-4,25.0, log=True),
        'dual': trial.suggest_categorical(f'dual_{name}', [True,False]),
        'loss': trial.suggest_categorical(f'loss_{name}', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
        'random_state': random_state

    }
    return params


# XGBRegressor parameters
def params_XGBRegressor(trial, random_state=None, name=None):
    return {
        'learning_rate': trial.suggest_float(f'learning_rate_{name}', 1e-3, 1, log=True),
        'subsample': trial.suggest_float(f'subsample_{name}', 0.05, 1.0),
        'min_child_weight': trial.suggest_int(f'min_child_weight_{name}', 1, 21),
        #'booster': trial.suggest_categorical(name='booster_{name}', choices=['gbtree', 'dart']),
        'n_estimators': 100,
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 11),
        'nthread': 1,
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'random_state': random_state
    }


def params_AdaBoostRegressor(trial, random_state=None, name=None):
    params = {
        'n_estimators': 100,
        'learning_rate': trial.suggest_float(f'learning_rate_{name}', 1e-3, 1.0, log=True),
        'loss': trial.suggest_categorical(f'loss_{name}', ['linear', 'square', 'exponential']),
        'random_state': random_state

    }
    return params

# ExtraTreesRegressor parameters
def params_ExtraTreesRegressor(trial, random_state=None, name=None):
    params = {
        'n_estimators': 100,
        'max_features': trial.suggest_float(f'max_features_{name}', 0.05, 1.0),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 21),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 21),
        'bootstrap': trial.suggest_categorical(f'bootstrap_{name}', [True, False]),

        #'criterion': trial.suggest_categorical(f'criterion_{name}', ['squared_error', 'poisson', 'absolute_error', 'friedman_mse']),

        #'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 10),

        #'min_weight_fraction_leaf': trial.suggest_float(f'min_weight_fraction_leaf_{name}', 0.0, 0.5),
        # 'max_features': trial.suggest_categorical(f'max_features_{name}', [None, 'auto', 'sqrt', 'log2']),
        #'max_leaf_nodes': trial.suggest_int(f'max_leaf_nodes_{name}', 2, 100),
        #'min_impurity_decrease': trial.suggest_float(f'min_impurity_decrease_{name}', 1e-5, 1e-1, log=True),
        # 'min_impurity_split': trial.suggest_float(f'min_impurity_split_{name}', 1e-5, 1e-1, log=True),

        #if bootstrap is True
        #'oob_score': trial.suggest_categorical(f'oob_score_{name}', [True, False]),

        #'ccp_alpha': trial.suggest_float(f'ccp_alpha_{name}', 1e-5, 1e-1, log=True),
        # 'max_samples': trial.suggest_float(f'max_samples_{name}', 0.05, 1.00),

        'random_state': random_state
    }
    return params



def make_regressor_config_dictionary(random_state=None, cv=None, n_samples=10):
    n_samples = min(n_samples,100) #TODO optimize this


    regressor_config_dictionary = {
        #ElasticNet: params_ElasticNet,
        ElasticNetCV: {
                        'l1_ratio': [.1, .5, .7, .9, .95, .99, 1],
                        'cv': cv,
                        },
        ExtraTreesRegressor: partial(params_ExtraTreesRegressor, random_state=random_state),
        GradientBoostingRegressor: partial(params_GradientBoostingRegressor, random_state=random_state),
        AdaBoostRegressor: partial(params_AdaBoostRegressor, random_state=random_state),
        DecisionTreeRegressor: partial(params_DecisionTreeRegressor, random_state=random_state),
        KNeighborsRegressor: partial(params_KNeighborsRegressor,n_samples=n_samples),
        LassoLarsCV: partial(params_LassoLarsCV, cv=cv),
        SVR: params_SVR,
        RandomForestRegressor: partial(params_RandomForestRegressor, random_state=random_state),
        RidgeCV: {'cv': cv},
        XGBRegressor: partial(params_XGBRegressor, random_state=random_state),
        SGDRegressor: partial(params_SGDRegressor, random_state= random_state),

    }

    return regressor_config_dictionary