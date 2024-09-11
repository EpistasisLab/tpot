from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
from ConfigSpace import EqualsCondition, OrConjunction, NotEqualsCondition, InCondition
from ..search_spaces.nodes.estimator_node import NONE_SPECIAL_STRING, TRUE_SPECIAL_STRING, FALSE_SPECIAL_STRING
import numpy as np
import sklearn


def get_LogisticRegression_ConfigurationSpace(random_state):

    dual = FALSE_SPECIAL_STRING

    space = {"solver":"saga",
                    "max_iter":1000,
                    "n_jobs":1,
                    "dual":dual,
                    }
    
    penalty = Categorical('penalty', ['l1', 'l2',"elasticnet"], default='l2')
    C = Float('C',  (0.01, 1e5), log=True)
    l1_ratio = Float('l1_ratio', (0.0, 1.0))
    class_weight = Categorical('class_weight', [NONE_SPECIAL_STRING, 'balanced'])

    l1_ratio_condition = EqualsCondition(l1_ratio, penalty, 'elasticnet')
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    

    cs = ConfigurationSpace(space)
    cs.add_hyperparameters([penalty, C, l1_ratio, class_weight])
    cs.add_conditions([l1_ratio_condition])

    return cs


def get_KNeighborsClassifier_ConfigurationSpace(n_samples):
        return ConfigurationSpace(

                space = {

                    'n_neighbors': Integer("n_neighbors", bounds=(1, min(100,n_samples)), log=True),
                    'weights': Categorical("weights", ['uniform', 'distance']),
                    'p': Integer("p", bounds=(1, 3)),
                    'metric': Categorical("metric", ['euclidean', 'minkowski']),
                    'n_jobs': 1,
                }
            ) 

def get_BaggingClassifier_ConfigurationSpace(random_state):
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

def get_DecisionTreeClassifier_ConfigurationSpace(n_featues, random_state):

    space = {
        'criterion': Categorical("criterion", ['gini', 'entropy']),
        'max_depth': Integer("max_depth", bounds=(1, min(20,2*n_featues))), #max of 20? log scale?
        'min_samples_split': Integer("min_samples_split", bounds=(1, 20)),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 20)),
        'max_features': Categorical("max_features", [NONE_SPECIAL_STRING, 'sqrt', 'log2']),
        'min_weight_fraction_leaf': 0.0,
        'class_weight' : Categorical('class_weight', [NONE_SPECIAL_STRING, 'balanced']),
    }
    

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )

#TODO Does not support predict_proba
def get_LinearSVC_ConfigurationSpace(random_state):
    space = {"dual":"auto"}
        
    penalty = Categorical('penalty', ['l1', 'l2'])
    C = Float('C',  (0.01, 1e5), log=True)
    loss = Categorical('loss', ['hinge', 'squared_hinge'])
    loss_condition = EqualsCondition(loss, penalty, 'l2')

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state


    cs = ConfigurationSpace(space)
    cs.add_hyperparameters([penalty, C, loss])
    cs.add_conditions([loss_condition])

    return cs


def get_SVC_ConfigurationSpace(random_state):

    space = {
            'max_iter': 3000,
            'probability':TRUE_SPECIAL_STRING}
        
    kernel = Categorical("kernel", ['poly', 'rbf', 'sigmoid', 'linear'])
    C = Float('C',  (0.01, 1e5), log=True)
    degree = Integer("degree", bounds=(1, 5))
    gamma = Float("gamma", bounds=(1e-5, 8), log=True)
    shrinking = Categorical("shrinking", [True, False])
    coef0 = Float("coef0", bounds=(-1, 1))
    class_weight = Categorical('class_weight', [NONE_SPECIAL_STRING, 'balanced'])

    degree_condition = EqualsCondition(degree, kernel, 'poly')
    gamma_condition = InCondition(gamma, kernel, ['rbf', 'poly'])
    coef0_condition = InCondition(coef0, kernel, ['poly', 'sigmoid'])

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state


    cs = ConfigurationSpace(space)
    cs.add_hyperparameters([kernel, C, coef0, degree, gamma, shrinking, class_weight])
    cs.add_conditions([degree_condition, gamma_condition, coef0_condition])

    return cs


def get_RandomForestClassifier_ConfigurationSpace( random_state):
    space = {
            'n_estimators': 128, #as recommended by Oshiro et al. (2012
            'max_features': Float("max_features", bounds=(0.01,1)), #log scale like autosklearn?
            'criterion': Categorical("criterion", ['gini', 'entropy']),
            'min_samples_split': Integer("min_samples_split", bounds=(2, 20)),
            'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 20)),
            'bootstrap': Categorical("bootstrap", [True, False]),
            'class_weight': Categorical("class_weight", [NONE_SPECIAL_STRING, 'balanced']),
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )


def get_XGBClassifier_ConfigurationSpace(random_state,):
    
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
        }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )

def get_LGBMClassifier_ConfigurationSpace(random_state,):

    space = {
            'boosting_type': Categorical("boosting_type", ['gbdt', 'dart', 'goss']),
            'num_leaves': Integer("num_leaves", bounds=(2, 256)),
            'max_depth': Integer("max_depth", bounds=(1, 10)),
            'n_estimators': Integer("n_estimators", bounds=(10, 100)),
            'class_weight': Categorical("class_weight", [NONE_SPECIAL_STRING, 'balanced']),
            'verbose':-1,
            'n_jobs': 1,
        }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space=space
    )


def get_ExtraTreesClassifier_ConfigurationSpace(random_state):
    space = {
            'n_estimators': 100,
            'criterion': Categorical("criterion", ["gini", "entropy"]),
            'max_features': Float("max_features", bounds=(0.01, 1.00)),
            'min_samples_split': Integer("min_samples_split", bounds=(2, 20)),
            'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 20)),
            'bootstrap': Categorical("bootstrap", [True, False]),
            'class_weight': Categorical("class_weight", [NONE_SPECIAL_STRING, 'balanced']),
            'n_jobs': 1,
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )



def get_SGDClassifier_ConfigurationSpace(random_state):
    
    space = {
            'loss': Categorical("loss", ['modified_huber']), #don't include hinge because we have LinearSVC, don't include log because we have LogisticRegression. TODO 'squared_hinge'? doesn't support predict proba
            'penalty': 'elasticnet',
            'alpha': Float("alpha", bounds=(1e-5, 0.01), log=True),
            'l1_ratio': Float("l1_ratio", bounds=(0.0, 1.0)),
            'eta0': Float("eta0", bounds=(0.01, 1.0)),
            'n_jobs': 1,
            'fit_intercept': Categorical("fit_intercept", [True]),
            'class_weight': Categorical("class_weight", [NONE_SPECIAL_STRING, 'balanced']),
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    power_t = Float("power_t", bounds=(1e-5, 100.0), log=True)
    learning_rate = Categorical("learning_rate", ['invscaling', 'constant', "optimal"])
    powertcond = EqualsCondition(power_t, learning_rate, 'invscaling')


    cs = ConfigurationSpace(
        space = space
    )

    cs.add_hyperparameters([power_t, learning_rate])
    cs.add_conditions([powertcond])

    return cs


GaussianNB_ConfigurationSpace = {}

def get_BernoulliNB_ConfigurationSpace():
    return ConfigurationSpace(
        space = {
            'alpha': Float("alpha", bounds=(1e-2, 100), log=True),
            'fit_prior': Categorical("fit_prior", [True, False]),
        }
    )


def get_MultinomialNB_ConfigurationSpace():
    return ConfigurationSpace(
        space = {
            'alpha': Float("alpha", bounds=(1e-3, 100), log=True),
            'fit_prior': Categorical("fit_prior", [True, False]),
        }
    )



def get_AdaBoostClassifier_ConfigurationSpace(random_state):
    space = {
            'n_estimators': Integer("n_estimators", bounds=(50, 500)),
            'learning_rate': Float("learning_rate", bounds=(0.01, 2), log=True),
            'algorithm': Categorical("algorithm", ['SAMME', 'SAMME.R']),
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )


def get_QuadraticDiscriminantAnalysis_ConfigurationSpace():
    return ConfigurationSpace(
        space = {
            'reg_param': Float("reg_param", bounds=(0, 1)),
        }
    )

def get_PassiveAggressiveClassifier_ConfigurationSpace(random_state):
    space = {
            'C': Float("C", bounds=(1e-5, 10), log=True),
            'loss': Categorical("loss", ['hinge', 'squared_hinge']),
            'average': Categorical("average", [True, False]),
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )
#TODO support auto shrinkage when solver is svd. may require custom node
def get_LinearDiscriminantAnalysis_ConfigurationSpace():

    solver = Categorical("solver", ['svd', 'lsqr', 'eigen'])
    shrinkage = Float("shrinkage", bounds=(0, 1))

    shrinkcond = NotEqualsCondition(shrinkage, solver, 'svd')

    cs = ConfigurationSpace()
    cs.add_hyperparameters([solver, shrinkage])
    cs.add_conditions([shrinkcond])

    return cs



#### Gradient Boosting Classifiers

def get_GradientBoostingClassifier_ConfigurationSpace(n_classes, random_state):
    early_stop = Categorical("early_stop", ["off", "valid", "train"])
    n_iter_no_change = Integer("n_iter_no_change",bounds=(1,20))
    validation_fraction = Float("validation_fraction", bounds=(0.01, 0.4))

    n_iter_no_change_cond = InCondition(n_iter_no_change, early_stop, ["valid", "train"] )
    validation_fraction_cond = EqualsCondition(validation_fraction, early_stop, "valid")

    space = {
        'learning_rate': Float("learning_rate", bounds=(1e-3, 1), log=True),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 200)),
        'min_samples_split': Integer("min_samples_split", bounds=(2, 20)),
        'subsample': Float("subsample", bounds=(0.1, 1.0)),
        'max_features': Float("max_features", bounds=(0.01, 1.00)),
        'max_leaf_nodes': Integer("max_leaf_nodes", bounds=(3, 2047)),
        'max_depth':NONE_SPECIAL_STRING,   # 'max_depth': Integer("max_depth", bounds=(1, 2*n_features)),
        'tol': 1e-4,
    }

    if n_classes == 2:
        space['loss']= Categorical("loss", ['log_loss', 'exponential'])
    else:
        space['loss'] = "log_loss"

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    cs = ConfigurationSpace(
        space = space
    )
    cs.add_hyperparameters([n_iter_no_change, validation_fraction, early_stop ])
    cs.add_conditions([validation_fraction_cond, n_iter_no_change_cond])
    return cs

def GradientBoostingClassifier_hyperparameter_parser(params):

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
        final_params['validation_fraction'] = None


    return final_params



#only difference is l2_regularization
def get_HistGradientBoostingClassifier_ConfigurationSpace(random_state):
    early_stop = Categorical("early_stop", ["off", "valid", "train"])
    n_iter_no_change = Integer("n_iter_no_change",bounds=(1,20))
    validation_fraction = Float("validation_fraction", bounds=(0.01, 0.4))

    n_iter_no_change_cond = InCondition(n_iter_no_change, early_stop, ["valid", "train"] )
    validation_fraction_cond = EqualsCondition(validation_fraction, early_stop, "valid")

    space = {
        'learning_rate': Float("learning_rate", bounds=(1e-3, 1), log=True),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 200)),
        'max_features': Float("max_features", bounds=(0.1,1.0)), 
        'max_leaf_nodes': Integer("max_leaf_nodes", bounds=(3, 2047)),
        'max_depth':NONE_SPECIAL_STRING, # 'max_depth': Integer("max_depth", bounds=(1, 2*n_features)),
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



def HistGradientBoostingClassifier_hyperparameter_parser(params):

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
        final_params['validation_fraction'] = None
        final_params['early_stopping'] = False
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

def get_MLPClassifier_ConfigurationSpace(random_state):
    space = {"n_iter_no_change":32}

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    cs =    ConfigurationSpace(
                space = space
            )

    n_hidden_layers = Integer("n_hidden_layers", bounds=(1, 3))
    n_nodes_per_layer = Integer("n_nodes_per_layer", bounds=(16, 512))
    activation = Categorical("activation", ["identity", "logistic",'tanh', 'relu'])
    alpha = Float("alpha", bounds=(1e-4, 1e-1), log=True)
    early_stopping = Categorical("early_stopping", [True,False])

    learning_rate_init = Float("learning_rate_init", bounds=(1e-4, 1e-1), log=True)
    learning_rate = Categorical("learning_rate", ['constant', 'invscaling', 'adaptive'])

    cs.add_hyperparameters([n_hidden_layers, n_nodes_per_layer, activation, alpha, learning_rate, early_stopping, learning_rate_init])

    return cs

def MLPClassifier_hyperparameter_parser(params):
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

###


###

def get_GaussianProcessClassifier_ConfigurationSpace(n_features, random_state):
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

def GaussianProcessClassifier_hyperparameter_parser(params):
    kernel = sklearn.gaussian_process.kernels.RBF(
        length_scale = [1.0]*params['n_features'],
        length_scale_bounds=[(params['thetaL'], params['thetaU'])] * params['n_features'],
    )
    final_params = {"kernel": kernel, 
                    "n_restarts_optimizer": 10,
                    "optimizer": "fmin_l_bfgs_b",
                    "copy_X_train": True,
                    }
    
    if "random_state" in params:
        final_params['random_state'] = params['random_state']
    
    return final_params