import sklearn
from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
from ConfigSpace import EqualsCondition, OrConjunction, NotEqualsCondition, InCondition
from ..search_spaces.nodes.estimator_node import NONE_SPECIAL_STRING, TRUE_SPECIAL_STRING, FALSE_SPECIAL_STRING
import numpy as np
#TODO: fill in remaining
#TODO check for places were we could use log scaling


ElasticNetCV_configspace = {
    "l1_ratio" :  np.arange(0.0, 1.01, 0.05),
}

def get_RandomForestRegressor_ConfigurationSpace(random_state):
    space =  {
        'n_estimators': 100,
        'criterion': Categorical("criterion", ['mse', 'mae', "friedman_mse"]),
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


def get_SGDRegressor_ConfigurationSpace(random_state):
    space = {
            'alpha': Float("alpha", bounds=(1e-7, 1e-1), log=True),
            'average': Categorical("average", [True, False]),
            'fit_intercept': Categorical("fit_intercept", [True]),
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    cs = ConfigurationSpace(
        space = space
    )

    l1_ratio = Float("l1_ratio", bounds=(1e-7, 1.0), log=True)
    penalty = Categorical("penalty", ["l1", "l2", "elasticnet"])
    epsilon = Float("epsilon", bounds=(1e-5, 1e-1), log=True)
    loss = Categorical("loss", ['epsilon_insensitive', 'squared_epsilon_insensitive', 'huber', 'squared_error'])
    eta0 = Float("eta0", bounds=(1e-7, 1e-1), log=True)
    learning_rate = Categorical("learning_rate", ['optimal', 'invscaling', 'constant'])
    power_t = Float("power_t", bounds=(1e-5, 1.0), log=True)

    elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
    epsilon_condition = InCondition(
        epsilon,
        loss,
        ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
    )

    eta0_in_inv_con = InCondition(eta0, learning_rate, ["invscaling", "constant"])
    power_t_condition = EqualsCondition(power_t, learning_rate, "invscaling")

    cs.add_hyperparameters(
        [l1_ratio, penalty, epsilon, loss, eta0, learning_rate, power_t]
    )
    cs.add_conditions(
        [elasticnet, epsilon_condition, power_t_condition, eta0_in_inv_con]
    )

    return cs


def get_Ridge_ConfigurationSpace(random_state):
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

def get_Lasso_ConfigurationSpace(random_state):
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

def get_ElasticNet_ConfigurationSpace(random_state):
    space = {
        'alpha': Float("alpha", bounds=(0.0, 1.0)),
        'l1_ratio': Float("l1_ratio", bounds=(0.0, 1.0)),
    }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )


def get_Lars_ConfigurationSpace(random_state):
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


def get_LassoLars_ConfigurationSpace(random_state):
    space = {
        'alpha': Float("alpha", bounds=(0.0, 1.0)),
        'eps': Float("eps", bounds=(1e-5, 1e-1), log=True),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )


def get_BaggingRegressor_ConfigurationSpace(random_state):
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

            'alpha_1': Float("alpha_1", bounds=(1e-10, 1e-3), log=True),
            'alpha_2': Float("alpha_2", bounds=(1e-10, 1e-3), log=True),
            'lambda_1': Float("lambda_1", bounds=(1e-10, 1e-3), log=True),
            'lambda_2': Float("lambda_2", bounds=(1e-10, 1e-3), log=True),
            'threshold_lambda': Integer("threshold_lambda", bounds=(1e3, 1e5)),

        }
    )

def get_TheilSenRegressor_ConfigurationSpace(random_state):
    space = {
        'n_subsamples': Integer("n_subsamples", bounds=(10, 10000)),
        'max_subpopulation': Integer("max_subpopulation", bounds=(10, 1000)),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )



def get_Perceptron_ConfigurationSpace(random_state):
    space = {
        'penalty': Categorical("penalty", [NONE_SPECIAL_STRING, 'l2', 'l1', 'elasticnet']),
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



def get_DecisionTreeRegressor_ConfigurationSpace(random_state):
    space = {
        'criterion': Categorical("criterion", ['squared_error', 'friedman_mse', 'mae']),
        # 'max_depth': Integer("max_depth", bounds=(1, n_features*2)),
        'min_samples_split': Integer("min_samples_split", bounds=(2, 21)),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 21)),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )


def get_KNeighborsRegressor_ConfigurationSpace(n_samples):
    return ConfigurationSpace(
        space = {
            'n_neighbors': Integer("n_neighbors", bounds=(1, min(100,n_samples))),
            'weights': Categorical("weights", ['uniform', 'distance']),
            'p': Integer("p", bounds=(1, 3)),
            'metric': Categorical("metric", ['minkowski', 'euclidean', 'manhattan']),
        }
    )


def get_LinearSVR_ConfigurationSpace(random_state):
    space = {
        'epsilon': Float("epsilon", bounds=(1e-4, 1.0), log=True),
        'C': Float('C',  (0.01, 1e5), log=True),
        'dual': "auto",
        'loss': Categorical("loss", ['epsilon_insensitive', 'squared_epsilon_insensitive']),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

#add coef0?
def get_SVR_ConfigurationSpace():
    space = {
            'epislon': Float("epsilon", bounds=(1e-4, 1.0), log=True),
            'shrinking': Categorical("shrinking", [True, False]),
            'C': Float('C',  (0.01, 1e5), log=True),
            'max_iter': 3000,
            'tol': 0.005,
        }
    
    cs = ConfigurationSpace(
        space = space
    )

    kernel = Categorical("kernel", ['poly', 'rbf', 'linear', 'sigmoid'])
    degree = Integer("degree", bounds=(1, 5))
    gamma = Float("gamma", bounds=(1e-5, 10.0), log=True)
    coef0 = Float("coef0", bounds=(-1, 1))
    

    degree_condition = EqualsCondition(degree, kernel, 'poly')
    gamma_condition = InCondition(gamma, kernel, ['poly', 'rbf',])
    coef0_condition = InCondition(coef0, kernel, ['poly', 'sigmoid'])

    cs.add_hyperparameters([kernel, degree, gamma, coef0])
    cs.add_conditions([degree_condition,gamma_condition])
    
    return cs




def get_XGBRegressor_ConfigurationSpace(random_state):
    space = {
        'n_estimators': 100,
        'learning_rate': Float("learning_rate", bounds=(1e-3, 1), log=True),
        'subsample': Float("subsample", bounds=(0.5, 1.0)),
        'min_child_weight': Integer("min_child_weight", bounds=(1, 21)),
        'gamma': Float("gamma", bounds=(1e-4, 20), log=True),
        'max_depth': Integer("max_depth", bounds=(3, 18)),
        'reg_alpha': Float("reg_alpha", bounds=(1e-4, 100), log=True),
        'reg_lambda': Float("reg_lambda", bounds=(1e-4, 1), log=True),
        'n_jobs': 1,
        'nthread': 1,
        'verbosity': 0,
        'objective': 'reg:squarederror',
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )


def get_AdaBoostRegressor_ConfigurationSpace(random_state):

    space = {
        'n_estimators': Integer("n_estimators", bounds=(50, 500)),
        'learning_rate': Float("learning_rate", bounds=(1e-3, 2.0), log=True),
        'loss': Categorical("loss", ['linear', 'square', 'exponential']),
    }


    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )

def get_ExtraTreesRegressor_ConfigurationSpace(random_state):
    space = {
        'n_estimators': 100,
        'criterion': Categorical("criterion", ["squared_error", "friedman_mse", "mae"]),
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
###

def get_GaussianProcessRegressor_ConfigurationSpace(n_features, random_state):
    space = {
        'n_features': n_features,
        'alpha': Float("alpha", bounds=(1e-10, 1.0), log=True),
        'thetaL': Float("thetaL", bounds=(1e-10, 1e-3), log=True),
        'thetaU': Float("thetaU", bounds=(1.0, 100000), log=True),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )

def GaussianProcessRegressor_hyperparameter_parser(params):
    kernel = sklearn.gaussian_process.kernels.RBF(
        length_scale = [1.0]*params['n_features'],
        length_scale_bounds=[(params['thetaL'], params['thetaU'])] * params['n_features'],
    )
    final_params = {"kernel": kernel, 
                    "alpha": params['alpha'],
                    "n_restarts_optimizer": 10,
                    "optimizer": "fmin_l_bfgs_b",
                    "normalize_y": True,
                    "copy_X_train": True,
                    }
    
    if "random_state" in params:
        final_params['random_state'] = params['random_state']
    
    return final_params

###
def get_GradientBoostingRegressor_ConfigurationSpace(random_state):
    early_stop = Categorical("early_stop", ["off", "valid", "train"])
    n_iter_no_change = Integer("n_iter_no_change",bounds=(1,20))
    validation_fraction = Float("validation_fraction", bounds=(0.01, 0.4))

    n_iter_no_change_cond = InCondition(n_iter_no_change, early_stop, ["valid", "train"] )
    validation_fraction_cond = EqualsCondition(validation_fraction, early_stop, "valid")

    space = {
        'loss': Categorical("loss", ['log_loss', 'exponential']),
        'learning_rate': Float("learning_rate", bounds=(1e-3, 1), log=True),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 200)),
        'min_samples_split': Integer("min_samples_split", bounds=(2, 20)),
        'subsample': Float("subsample", bounds=(0.1, 1.0)),
        'max_features': Float("max_features", bounds=(0.01, 1.00)),
        'max_leaf_nodes': Integer("max_leaf_nodes", bounds=(3, 2047)),
        'max_depth':NONE_SPECIAL_STRING, #'max_depth': Integer("max_depth", bounds=(1, 2*n_features)),
        'tol': 1e-4,
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    cs = ConfigurationSpace(
        space = space
    )
    cs.add_hyperparameters([n_iter_no_change, validation_fraction, early_stop ])
    cs.add_conditions([validation_fraction_cond, n_iter_no_change_cond])
    return cs

def GradientBoostingRegressor_hyperparameter_parser(params):

    final_params = {
        'loss': params['loss'],
        'learning_rate': params['learning_rate'],
        'min_samples_leaf': params['min_samples_leaf'],
        'min_samples_split': params['min_samples_split'],
        'max_features': params['max_features'],
        'max_leaf_nodes': params['max_leaf_nodes'],
        'max_depth': params['max_depth'],
        'tol': params['tol'],
        'subsample': params['subsample']
    }

    if 'random_state' in params:
        final_params['random_state'] = params['random_state']

    if params['early_stop'] == 'off':
        final_params['n_iter_no_change'] = None
        final_params['validation_fraction'] = None
    elif params['early_stop'] == 'valid':
        #this is required because in crossover, its possible that n_iter_no_change is not in the params
        if 'n_iter_no_change' not in params:
            final_params['n_iter_no_change'] = 10
        else:
            final_params['n_iter_no_change'] = params['n_iter_no_change']    
        if 'validation_fraction' not in params:
            final_params['validation_fraction'] = 0.1
        else:
            final_params['validation_fraction'] = params['validation_fraction']
    elif params['early_stop'] == 'train':
        if 'n_iter_no_change' not in params:
            final_params['n_iter_no_change'] = 10
        else:
            final_params['n_iter_no_change'] = params['n_iter_no_change']  
        final_params['validation_fraction'] = None


    return final_params

#only difference is l2_regularization
def get_HistGradientBoostingRegressor_ConfigurationSpace(random_state):
    early_stop = Categorical("early_stop", ["off", "valid", "train"])
    n_iter_no_change = Integer("n_iter_no_change",bounds=(1,20))
    validation_fraction = Float("validation_fraction", bounds=(0.01, 0.4))

    n_iter_no_change_cond = InCondition(n_iter_no_change, early_stop, ["valid", "train"] )
    validation_fraction_cond = EqualsCondition(validation_fraction, early_stop, "valid")

    space = {
        'loss': Categorical("loss", ['log_loss', 'exponential']),
        'learning_rate': Float("learning_rate", bounds=(1e-3, 1), log=True),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 200)),
        'max_features': Float("max_features", bounds=(0.1,1.0)), 
        'max_leaf_nodes': Integer("max_leaf_nodes", bounds=(3, 2047)),
        'max_depth':NONE_SPECIAL_STRING, #'max_depth': Integer("max_depth", bounds=(1, 2*n_features)),
        'l2_regularization': Float("l2_regularization", bounds=(1e-10, 1), log=True),
        'tol': 1e-4,
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    cs = ConfigurationSpace(
        space = space
    )
    cs.add_hyperparameters([n_iter_no_change, validation_fraction, early_stop ])
    cs.add_conditions([validation_fraction_cond, n_iter_no_change_cond])

    return cs


def HistGradientBoostingRegressor_hyperparameter_parser(params):

    final_params = {
        'learning_rate': params['learning_rate'],
        'min_samples_leaf': params['min_samples_leaf'],
        'max_features': params['max_features'],
        'max_leaf_nodes': params['max_leaf_nodes'],
        'max_depth': params['max_depth'],
        'tol': params['tol'],
        'l2_regularization': params['l2_regularization']
    }

    if 'random_state' in params:
        final_params['random_state'] = params['random_state']

    
    if params['early_stop'] == 'off':
        # final_params['n_iter_no_change'] = 0
        # final_params['validation_fraction'] = None
        final_params['early_stopping'] = False
    elif params['early_stop'] == 'valid':
        if 'n_iter_no_change' not in params:
            final_params['n_iter_no_change'] = 10
        else:
            final_params['n_iter_no_change'] = params['n_iter_no_change']    
        if 'validation_fraction' not in params:
            final_params['validation_fraction'] = 0.1
        else:
            final_params['validation_fraction'] = params['validation_fraction']
        final_params['early_stopping'] = True
    elif params['early_stop'] == 'train':
        if 'n_iter_no_change' not in params:
            final_params['n_iter_no_change'] = 10
        else:
            final_params['n_iter_no_change'] = params['n_iter_no_change'] 
        final_params['validation_fraction'] = None
        final_params['early_stopping'] = True


    return final_params


###

def get_MLPRegressor_ConfigurationSpace(random_state):
    space = {"n_iter_no_change":32}

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    cs =    ConfigurationSpace(
                space = space
            )

    n_hidden_layers = Integer("n_hidden_layers", bounds=(1, 3))
    n_nodes_per_layer = Integer("n_nodes_per_layer", bounds=(16, 512))
    activation = Categorical("activation", ['tanh', 'relu'])
    alpha = Float("alpha", bounds=(1e-7, 1e-1), log=True)
    learning_rate = Float("learning_rate", bounds=(1e-4, 1e-1), log=True)
    early_stopping = Categorical("early_stopping", [True,False])

    learning_rate_init = Float("learning_rate_init", bounds=(1e-4, 1e-1), log=True)
    learning_rate = Categorical("learning_rate", ['constant', 'invscaling', 'adaptive'])

    cs.add_hyperparameters([n_hidden_layers, n_nodes_per_layer, activation, alpha, learning_rate, early_stopping, learning_rate_init])

    return cs

def MLPRegressor_hyperparameter_parser(params):
    hyperparameters = {
        'n_iter_no_change': params['n_iter_no_change'],
        'hidden_layer_sizes' : [params['n_nodes_per_layer']]*params['n_hidden_layers'],
        'activation': params['activation'],
        'alpha': params['alpha'],
        'early_stopping': params['early_stopping'],
        'learning_rate_init': params['learning_rate_init'],
        'learning_rate': params['learning_rate'],
    }

    if 'random_state' in params:
        hyperparameters['random_state'] = params['random_state']

    return hyperparameters


def get_BaggingRegressor_ConfigurationSpace(random_state):
    space = {
            'n_estimators': Integer("n_estimators", bounds=(3, 100)),
            'max_samples': Float("max_samples", bounds=(0.1, 1.0)),
            'max_features': Float("max_features", bounds=(0.1, 1.0)),
            
            'bootstrap_features': Categorical("bootstrap_features", [True, False]),
            'n_jobs': 1,
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    bootstrap = Categorical("bootstrap", [True, False])
    oob_score = Categorical("oob_score", [True, False])

    oob_condition = EqualsCondition(oob_score, bootstrap, True)

    cs = ConfigurationSpace(
        space = space
    )

    cs.add_hyperparameters([bootstrap, oob_score])
    cs.add_conditions([oob_condition])

    return cs

def get_LGBMRegressor_ConfigurationSpace(random_state,):

    space = {
            'boosting_type': Categorical("boosting_type", ['gbdt', 'dart', 'goss']),
            'num_leaves': Integer("num_leaves", bounds=(2, 256)),
            'max_depth': Integer("max_depth", bounds=(1, 10)),
            'n_estimators': Integer("n_estimators", bounds=(10, 100)),
            'verbose':-1,
            'n_jobs': 1,
        }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space=space
    )
