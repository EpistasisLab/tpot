import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
from ConfigSpace import EqualsCondition


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
    space = { 'initial_strategy' : Categorical('initial_strategy', 
                                             ['mean', 'median', 
                                              'most_frequent', 'constant']),
                'n_nearest_features' : Integer('n_nearest_features', 
                                           bounds=(1, n_features)),
                'imputation_order' : Categorical('imputation_order', 
                                             ['ascending', 'descending', 
                                              'roman', 'arabic', 'random']),
    }

    estimator = Categorical('estimator', ['Bayesian', 'RFR', 'Ridge', 'KNN'])  
    sample_posterior = Categorical('sample_posterior', [True, False])
    sampling_condition = EqualsCondition(sample_posterior, estimator, 'Bayesian')

    if random_state is not None: 
            #This is required because configspace doesn't allow None as a value
            space['random_state'] = random_state

    cs = ConfigurationSpace(space=space)
    cs.add_hyperparameters([estimator, sample_posterior])
    cs.add_conditions([sampling_condition])
    return cs

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
            'initial_strategy' : params['initial_strategy'],
            'n_nearest_features' : params['n_nearest_features'],
            'imputation_order' : params['imputation_order'],
    }

    if 'sample_posterior' in params:
        final_params['sample_posterior'] = params['sample_posterior']

    if 'random_state' in params:
        final_params['random_state'] = params['random_state']
        
    return final_params