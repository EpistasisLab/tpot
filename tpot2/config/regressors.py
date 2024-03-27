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


from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal


#TODO: fill in remaining
#TODO check for places were we could use log scaling

def get_RandomForestRegressor_ConfigurationSpace(random_state=None):
    space =  {
        'n_estimators': 100,
        'max_features': Float("max_features", bounds=(0.05, 1.0)),
        'bootstrap': Categorical("bootstrap", [True, False]),
        'min_samples_split': Integer("min_samples_split", bounds=(2, 21)),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 21)),
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
        )


def get_SGDRegressor_ConfigurationSpace(random_state=None):
    space = {
            'loss': Categorical("loss", ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
            'penalty': 'elasticnet',
            'alpha': Float("alpha", bounds=(1e-5, 0.01), log=True),
            'learning_rate': Categorical("learning_rate", ['invscaling', 'constant']),
            'l1_ratio': Float("l1_ratio", bounds=(0.0, 1.0)),
            'eta0': Float("eta0", bounds=(0.01, 1.0)),
            'power_t': Float("power_t", bounds=(1e-5, 100.0), log=True),
            'fit_intercept': Categorical("fit_intercept", [True]),
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )


def get_Ridge_ConfigurationSpace(random_state=None):
    space = {
        'alpha': Float("alpha", bounds=(0.0, 1.0)),
        'fit_intercept': Categorical("fit_intercept", [True]),
        'tol': Float("tol", bounds=(1e-5, 1e-1), log=True),
        'solver': Categorical("solver", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
    }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )

def get_Lasso_ConfigurationSpace(random_state=None):
    space = {
        'alpha': Float("alpha", bounds=(0.0, 1.0)),
        'fit_intercept': Categorical("fit_intercept", [True]),
        'tol': 0.0001,
    }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )

def get_ElasticNet_ConfigurationSpace(random_state=None):
    space = {
        'alpha': Float("alpha", bounds=(0.0, 1.0)),
        'l1_ratio': Float("l1_ratio", bounds=(0.0, 1.0)),
    }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )


def get_Lars_ConfigurationSpace(random_state=None):
    space = {
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

def get_OthogonalMatchingPursuit_ConfigurationSpace():
    return ConfigurationSpace(
        space = {
        }
    )

def get_BayesianRidge_ConfigurationSpace():
    return ConfigurationSpace(
        space = {
            'tol': 0.0001,
            'alpha_1': Float("alpha_1", bounds=(1e-6, 1e-1), log=True),
            'alpha_2': Float("alpha_2", bounds=(1e-6, 1e-1), log=True),
            'lambda_1': Float("lambda_1", bounds=(1e-6, 1e-1), log=True),
            'lambda_2': Float("lambda_2", bounds=(1e-6, 1e-1), log=True),
        }
    )


def get_LassoLars_ConfigurationSpace(random_state=None):
    space = {
        'alpha': Float("alpha", bounds=(0.0, 1.0)),
        'eps': Float("eps", bounds=(1e-5, 1e-1), log=True),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

def get_LassoLarsCV_ConfigurationSpace(cv):
    return ConfigurationSpace(
        space = {
            'cv': cv,
        }
    )


def get_BaggingRegressor_ConfigurationSpace(random_state=None):
    space = {
        'max_samples': Float("max_samples", bounds=(0.05, 1.00)),
        'max_features': Float("max_features", bounds=(0.05, 1.00)),
        'bootstrap': Categorical("bootstrap", [True, False]),
        'bootstrap_features': Categorical("bootstrap_features", [True, False]),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

def get_ARDRegression_ConfigurationSpace():
    return ConfigurationSpace(
        space = {

            'alpha_1': Float("alpha_1", bounds=(1e-6, 1e-1), log=True),
            'alpha_2': Float("alpha_2", bounds=(1e-6, 1e-1), log=True),
            'lambda_1': Float("lambda_1", bounds=(1e-6, 1e-1), log=True),
            'lambda_2': Float("lambda_2", bounds=(1e-6, 1e-1), log=True),
            'threshold_lambda': Integer("threshold_lambda", bounds=(100, 1000)),

        }
    )

def get_TheilSenRegressor_ConfigurationSpace(random_state=None):
    space = {
        'n_subsamples': Integer("n_subsamples", bounds=(10, 100)),
        'max_subpopulation': Integer("max_subpopulation", bounds=(100, 1000)),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )


def get_SVR_ConfigurationSpace():
    return ConfigurationSpace(
        space = {
            'kernel': Categorical("kernel", ['poly', 'rbf', 'linear', 'sigmoid']),
            'C': Float("C", bounds=(1e-4, 25), log=True),
            'degree': Integer("degree", bounds=(1, 4)),
            'max_iter': 3000,
            'tol': 0.005,
        }
    )


def get_Perceptron_ConfigurationSpace(random_state=None):
    space = {
        'penalty': Categorical("penalty", [None, 'l2', 'l1', 'elasticnet']),
        'alpha': Float("alpha", bounds=(1e-5, 1e-1), log=True),
        'l1_ratio': Float("l1_ratio", bounds=(0.0, 1.0)),
        'learning_rate': Categorical("learning_rate", ['constant', 'optimal', 'invscaling']),
        'validation_fraction': Float("validation_fraction", bounds=(0.05, 1.00)),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

def get_MLPRegressor_ConfigurationSpace(random_state=None):
    space = {
        'alpha': Float("alpha", bounds=(1e-4, 1e-1), log=True),
        'learning_rate_init': Float("learning_rate_init", bounds=(1e-3, 1.), log=True),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )


def get_GradientBoostingRegressor_ConfigurationSpace(random_state=None):
    space = {
        'n_estimators': 100,
        'loss': Categorical("loss", ['ls', 'lad', 'huber', 'quantile']),
        'learning_rate': Float("learning_rate", bounds=(1e-4, 1), log=True),
        'max_depth': Integer("max_depth", bounds=(1, 11)),
        'min_samples_split': Integer("min_samples_split", bounds=(2, 21)),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 21)),
        'subsample': Float("subsample", bounds=(0.05, 1.00)),
        'max_features': Float("max_features", bounds=(0.05, 1.00)),
    }


def get_DecisionTreeRegressor_ConfigurationSpace(random_state=None):
    space = {
        'max_depth': Integer("max_depth", bounds=(1, 11)),
        'min_samples_split': Integer("min_samples_split", bounds=(2, 21)),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 21)),
    }

    return ConfigurationSpace(
        space = space
    )


def get_KNeighborsRegressor_ConfigurationSpace(n_samples=100):
    return ConfigurationSpace(
        space = {
            'n_neighbors': Integer("n_neighbors", bounds=(1, n_samples)),
            'weights': Categorical("weights", ['uniform', 'distance']),
            'p': Integer("p", bounds=(1, 3)),
            'metric': Categorical("metric", ['minkowski', 'euclidean', 'manhattan']),
        }
    )

def get_LinearSVR_ConfigurationSpace(random_state=None):
    space = {
        'epsilon': Float("epsilon", bounds=(1e-4, 1.0), log=True),
        'C': Float("C", bounds=(1e-4, 25.0), log=True),
        'dual': Categorical("dual", [True, False]),
        'loss': Categorical("loss", ['epsilon_insensitive', 'squared_epsilon_insensitive']),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )


def get_XGBRegressor_ConfigurationSpace(random_state=None):
    space = {
        'learning_rate': Float("learning_rate", bounds=(1e-3, 1), log=True),
        'subsample': Float("subsample", bounds=(0.05, 1.0)),
        'min_child_weight': Integer("min_child_weight", bounds=(1, 21)),
        'n_estimators': 100,
        'max_depth': Integer("max_depth", bounds=(1, 11)),
        'nthread': 1,
        'verbosity': 0,
        'objective': 'reg:squarederror',
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )


def get_AdaBoostRegressor_ConfigurationSpace(random_state=None):

    space = {
        'n_estimators': Integer("n_estimators", bounds=(50, 100)),
        'learning_rate': Float("learning_rate", bounds=(1e-3, 1.0), log=True),
        'loss': Categorical("loss", ['linear', 'square', 'exponential']),
    }


    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

def get_ExtraTreesRegressor_ConfigurationSpace(random_state=None):
    space = {
        'n_estimators': 100,
        'max_features': Float("max_features", bounds=(0.05, 1.0)),
        'min_samples_split': Integer("min_samples_split", bounds=(2, 21)),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 21)),
        'bootstrap': Categorical("bootstrap", [True, False]),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )