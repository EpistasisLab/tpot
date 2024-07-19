import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
import sklearn

simple_imputer_cs = ConfigurationSpace(
    space = {
        'strategy' : Categorical('strategy', 
                                 ['mean','median', 'most_frequent', 'constant']
                                 ),
        #'add_indicator' : Categorical('add_indicator', [True, False]), 
        #Removed add_indicator, it appends a mask next to the rest of the data 
        # and can cause errors. gk
    }
)

def get_IterativeImputer_config_space(n_features, random_state):
    space = { 
            'estimator' : Categorical('estimator', 
                                      ['Bayesian', 'RFR', 'Ridge', 
                                       'KNN', 'RandomForest']),
            'sample_posterior' : Categorical('sample_posterior', [True, False]),
            'initial_strategy' : Categorical('initial_strategy', 
                                             ['mean', 'median', 
                                              'most_frequent', 'constant']),
            'n_nearest_features' : Integer('n_nearest_features', 
                                           bounds=(1, n_features)),
            'imputation_order' : Categorical('imputation_order', 
                                             ['ascending', 'descending', 
                                              'roman', 'arabic', 'random']),
    }
    if random_state is not None: 
            #This is required because configspace doesn't allow None as a value
            space['random_state'] = random_state

    return ConfigurationSpace(
            space = space
            )

def get_KNNImputer_config_space(n_samples):
    space = {
            'n_neighbors': Integer('n_neighbors', bounds=(1, max(n_samples,100))),
            'weights': Categorical('weights', ['uniform', 'distance'])
    }
    return ConfigurationSpace(
          space=space
    )

def IterativeImputer_hyperparameter_parser(params):
    est = params['estimator']
    match est:
        case 'Bayesian':
                estimator = sklearn.linear_model.BayesianRidge()
        case 'RFR':
                estimator = sklearn.ensemble.RandomForestRegressor()
        case 'Ridge':
                estimator = sklearn.linear_model.Ridge()
        case 'KNN':
                estimator = sklearn.neighbors.KNeighborsRegressor()
    
    final_params = {
            'estimator' : estimator,
            'sample_posterior' : params['sample_posterior'],
            'initial_strategy' : params['initial_strategy'],
            'n_nearest_features' : params['n_nearest_features'],
            'imputation_order' : params['imputation_order'],
    }

    if "random_state" in params:
        final_params['random_state'] = params['random_state']

    return final_params