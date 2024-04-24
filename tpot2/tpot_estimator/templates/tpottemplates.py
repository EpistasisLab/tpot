import tpot2
import numpy as np
import pandas as pd
from ..estimator import TPOTEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tpot2.selectors import survival_select_NSGA2, tournament_selection_dominated
#TODO These do not follow sklearn conventions of __init__

class TPOTRegressor(TPOTEstimator):
    def __init__(       self,
                        scorers=['neg_mean_squared_error'], 
                        scorers_weights=[1],
                        cv = 10, #remove this and use a value based on dataset size?
                        other_objective_functions=[], #tpot2.objectives.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights = [],
                        objective_function_names = None,
                        bigger_is_better = True,
                        categorical_features = None,
                        memory = None,
                        preprocessing = False,
                        max_time_seconds=3600, 
                        max_eval_time_seconds=60*10, 
                        n_jobs = 1,
                        validation_strategy = "none",
                        validation_fraction = .2, 
                        early_stop = None,
                        warm_start = False,
                        periodic_checkpoint_folder = None, 
                        verbose = 0,
                        memory_limit = "4GB",
                        client = None,
                        random_state=None,
                        **tpotestimator_kwargs,
        ):
        """
        See TPOTEstimator for documentation
        """

        self.scorers = scorers
        self.scorers_weights = scorers_weights
        self.cv = cv
        self.other_objective_functions = other_objective_functions
        self.other_objective_functions_weights = other_objective_functions_weights
        self.objective_function_names = objective_function_names
        self.bigger_is_better = bigger_is_better
        self.categorical_features = categorical_features
        self.memory = memory
        self.preprocessing = preprocessing
        self.max_time_seconds = max_time_seconds
        self.max_eval_time_seconds = max_eval_time_seconds
        self.n_jobs = n_jobs
        self.validation_strategy = validation_strategy
        self.validation_fraction = validation_fraction
        self.early_stop = early_stop
        self.warm_start = warm_start
        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.verbose = verbose
        self.memory_limit = memory_limit
        self.client = client
        self.random_state = random_state
        self.tpotestimator_kwargs = tpotestimator_kwargs

        self.initialized = False


    def fit(self, X, y):

        if not self.initialized:
            get_search_space_params = {"n_classes": None, 
                                        "n_samples":len(y), 
                                        "n_features":X.shape[1], 
                                        "random_state":self.random_state}

            search_space = tpot2.search_spaces.pipelines.GraphPipeline(
                root_search_space= tpot2.config.get_search_space("regressors", **get_search_space_params),
                leaf_search_space = None, 
                inner_search_space = tpot2.config.get_search_space(["selectors","transformers","regressors","scalers"],**get_search_space_params),
                max_size = 10,
            )


            super(TPOTRegressor,self).__init__(
                search_space=search_space,
                scorers=self.scorers, 
                scorers_weights=self.scorers_weights,
                cv=self.cv,
                other_objective_functions=self.other_objective_functions, #tpot2.objectives.estimator_objective_functions.number_of_nodes_objective],
                other_objective_functions_weights = self.other_objective_functions_weights,
                objective_function_names = self.objective_function_names,
                bigger_is_better = self.bigger_is_better,
                categorical_features = self.categorical_features,
                memory = self.memory,
                preprocessing = self.preprocessing,
                max_time_seconds=self.max_time_seconds, 
                max_eval_time_seconds=self.max_eval_time_seconds, 
                n_jobs=self.n_jobs,
                validation_strategy = self.validation_strategy,
                validation_fraction = self.validation_fraction, 
                early_stop = self.early_stop,
                warm_start = self.warm_start,
                periodic_checkpoint_folder = self.periodic_checkpoint_folder, 
                verbose = self.verbose,
                classification=False,
                memory_limit = self.memory_limit,
                client = self.client,
                random_state=self.random_state,
                **self.tpotestimator_kwargs)
            self.initialized = True
        
        return super().fit(X,y)


class TPOTClassifier(TPOTEstimator):
    def __init__(       self,
                        scorers=['roc_auc_ovr'], 
                        scorers_weights=[1],
                        cv = 10,
                        other_objective_functions=[], #tpot2.objectives.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights = [],
                        objective_function_names = None,
                        bigger_is_better = True,
                        categorical_features = None,
                        memory = None,
                        preprocessing = False,
                        max_time_seconds=3600, 
                        max_eval_time_seconds=60*10, 
                        n_jobs = 1,
                        validation_strategy = "none",
                        validation_fraction = .2, 
                        early_stop = None,
                        warm_start = False,
                        periodic_checkpoint_folder = None, 
                        verbose = 0,
                        memory_limit = "4GB",
                        client = None,
                        random_state=None,
                        **tpotestimator_kwargs,
                        
        ):
        """
        See TPOTEstimator for documentation
        """

        self.scorers = scorers
        self.scorers_weights = scorers_weights
        self.cv = cv
        self.other_objective_functions = other_objective_functions
        self.other_objective_functions_weights = other_objective_functions_weights
        self.objective_function_names = objective_function_names
        self.bigger_is_better = bigger_is_better
        self.categorical_features = categorical_features
        self.memory = memory
        self.preprocessing = preprocessing
        self.max_time_seconds = max_time_seconds
        self.max_eval_time_seconds = max_eval_time_seconds
        self.n_jobs = n_jobs
        self.validation_strategy = validation_strategy
        self.validation_fraction = validation_fraction
        self.early_stop = early_stop
        self.warm_start = warm_start
        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.verbose = verbose
        self.memory_limit = memory_limit
        self.client = client
        self.random_state = random_state
        self.tpotestimator_kwargs = tpotestimator_kwargs

        self.initialized = False

    def fit(self, X, y):

        if not self.initialized:

            get_search_space_params = {"n_classes": len(np.unique(y)), 
                                       "n_samples":len(y), 
                                       "n_features":X.shape[1], 
                                       "random_state":self.random_state}

            search_space = tpot2.search_spaces.pipelines.GraphPipeline(
                root_search_space= tpot2.config.get_search_space("classifiers", **get_search_space_params),
                leaf_search_space = None, 
                inner_search_space = tpot2.config.get_search_space(["selectors","transformers","classifiers", "scalers"], **get_search_space_params),
                max_size = 10,
            )


            super(TPOTClassifier,self).__init__(
                search_space=search_space,
                scorers=self.scorers, 
                scorers_weights=self.scorers_weights,
                cv = self.cv,
                other_objective_functions=self.other_objective_functions, #tpot2.objectives.estimator_objective_functions.number_of_nodes_objective],
                other_objective_functions_weights = self.other_objective_functions_weights,
                objective_function_names = self.objective_function_names,
                bigger_is_better = self.bigger_is_better,
                categorical_features = self.categorical_features,
                memory = self.memory,
                preprocessing = self.preprocessing,
                max_time_seconds=self.max_time_seconds, 
                max_eval_time_seconds=self.max_eval_time_seconds, 
                n_jobs=self.n_jobs,
                validation_strategy = self.validation_strategy,
                validation_fraction = self.validation_fraction, 
                early_stop = self.early_stop,
                warm_start = self.warm_start,
                periodic_checkpoint_folder = self.periodic_checkpoint_folder, 
                verbose = self.verbose,
                classification=True,
                memory_limit = self.memory_limit,
                client = self.client,
                random_state=self.random_state,
                **self.tpotestimator_kwargs)
            self.initialized = True
        
        return super().fit(X,y)


    def predict(self, X, **predict_params):
        check_is_fitted(self)
        #X=check_array(X)
        return self.fitted_pipeline_.predict(X,**predict_params)
