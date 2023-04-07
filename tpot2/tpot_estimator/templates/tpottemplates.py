import tpot2
import numpy as np
from ..estimator import TPOTEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
#TODO These do not follow sklearn conventions of __init__

class TPOTRegressor(TPOTEstimator):
    def __init__(       self,
                        *args,
                        scorers = ['neg_mean_squared_error'], #these get passed into CV
                        scorers_weights = [1],
                        classification = False,
                        population_size = 100,
                        generations = 100,
                        initial_population_size = None,
                        population_scaling = .8, 
                        generations_until_end_population = 1,  
                        callback: tpot2.CallBackInterface = None,
                        n_jobs=1,
                        
                        cv = 5,
                        verbose = 0, #TODO
                        other_objective_functions=[tpot2.estimator_objective_functions.average_path_length_objective], #tpot2.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights = [-1],
                        bigger_is_better = True,
                        evolver = "nsga2",
                        evolver_params = {},
                        max_depth = np.inf,
                        max_size = np.inf, 
                        max_children = np.inf,
                        root_config_dict= 'regressors',
                        inner_config_dict=["selectors", "transformers"],
                        leaf_config_dict= None,

                        subsets = None,

                        max_time_seconds=float('inf'), 
                        max_eval_time_seconds=60*10, #TODO auto set a timer? Should this be none?
                        memory_limit = '4GB',

                        n_initial_optimizations = 0,
                        optimization_cv = 3,
                        max_optimize_time_seconds=60*20,
                        optimization_steps = 10,

                        periodic_checkpoint_folder = None,

                        threshold_evaluation_early_stop = None, #[lower percentile, upper percentile]
                        threshold_evaluation_scaling = 4, # [0,inf) lower means longer to get to upper, higher reacher upper sooner
                        min_history_threshold = 20,
                        selection_evaluation_early_stop = None, #[lower percentile, upper percentile]
                        selection_evaluation_scaling = 4, # [0,inf) lower means longer to get to upper, higher reacher upper sooner

                        scorers_early_stop_tol = 0.001,
                        other_objectives_early_stop_tol =None,
                        early_stop = None,

                        warm_start = False,
                        memory = None,
                        cross_val_predict_cv = 0, #TODO: crossval predict for transformers and selectors?
                        
                        budget_range = None, #[.2, 1] [start budget, end budget]
                        budget_scaling = .8, # [0,1]
                        generations_until_end_budget = 1, #    


                        preprocessing = False,  

                        validation_strategy = "none",
                        validation_fraction = .2,

                        subset_column = None,

                        stepwise_steps = 5,
        ):
        super(TPOTRegressor,self).__init__(
            *args,
            scorers = scorers, #these get passed into CV
            scorers_weights = scorers_weights,
            classification=classification,
            population_size = population_size, #alternatively, an instance of evolver could be passed in?
            generations = generations,
            initial_population_size = initial_population_size,
            population_scaling = population_scaling,
            generations_until_end_population = generations_until_end_population,
            callback=callback,
            n_jobs=n_jobs,
            cv = cv,
            verbose = verbose, #TODO
            other_objective_functions=other_objective_functions, #tpot2.estimator_objective_functions.number_of_nodes_objective],
            other_objective_functions_weights = other_objective_functions_weights,
            bigger_is_better = bigger_is_better,
            evolver = evolver,
            evolver_params = evolver_params,
            max_depth = max_depth,
            max_size = max_size,
            max_children = max_children,
            root_config_dict= root_config_dict,
            inner_config_dict=inner_config_dict,
            leaf_config_dict= leaf_config_dict,
            subsets = subsets,
            max_time_seconds=max_time_seconds,
            memory_limit = memory_limit,
            max_eval_time_seconds=max_eval_time_seconds, #TODO auto set a timer? Should this be none?
            n_initial_optimizations = n_initial_optimizations,
            optimization_cv = optimization_cv,
            max_optimize_time_seconds=max_optimize_time_seconds,
            optimization_steps = optimization_steps,
            periodic_checkpoint_folder = periodic_checkpoint_folder,
            threshold_evaluation_early_stop = threshold_evaluation_early_stop, #[lower percentile, upper percentile]
            threshold_evaluation_scaling = threshold_evaluation_scaling, # [0,inf) lower means longer to get to upper, higher reacher upper sooner
            min_history_threshold = min_history_threshold,
            selection_evaluation_early_stop = selection_evaluation_early_stop, #[lower percentile, upper percentile]
            selection_evaluation_scaling = selection_evaluation_scaling, # [0,inf) lower means longer to get to upper, higher reacher upper sooner
            scorers_early_stop_tol = scorers_early_stop_tol,
            other_objectives_early_stop_tol =other_objectives_early_stop_tol,
            early_stop = early_stop,
            warm_start = warm_start,
            memory = memory,
            cross_val_predict_cv = cross_val_predict_cv, #TODO: crossval predict for transformers and selectors?
            budget_range = budget_range, #[.2, 1] [start budget, end budget]
            budget_scaling = budget_scaling, # [0,1]
            generations_until_end_budget = generations_until_end_budget, #
            preprocessing = preprocessing,
            validation_strategy = validation_strategy,
            validation_fraction = validation_fraction,
            subset_column = subset_column,
            stepwise_steps = stepwise_steps,
        )


class TPOTClassifier(TPOTEstimator):
    def __init__(       self,
                        scorers = ['roc_auc_ovr'], #these get passed into CV
                        scorers_weights = [1],
                        classification=True,
                                                population_size = 100,
                        generations = 100,
                        initial_population_size = None,
                        population_scaling = .8, 
                        generations_until_end_population = 1,  
                        callback: tpot2.CallBackInterface = None,
                        n_jobs=1,
                        
                        cv = 5,
                        verbose = 0, #TODO
                        other_objective_functions=[tpot2.estimator_objective_functions.average_path_length_objective], #tpot2.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights = [-1],
                        bigger_is_better = True,
                        evolver = "nsga2",
                        evolver_params = {},
                        max_depth = np.inf,
                        max_size = np.inf, 
                        max_children = np.inf,
                        root_config_dict= 'classifiers',
                        inner_config_dict=["selectors", "transformers"],
                        leaf_config_dict= None,

                        subsets = None,

                        max_time_seconds=float('inf'), 
                        max_eval_time_seconds=60*10, #TODO auto set a timer? Should this be none?
                        memory_limit = '4GB',

                        n_initial_optimizations = 0,
                        optimization_cv = 3,
                        max_optimize_time_seconds=60*20,
                        optimization_steps = 10,

                        periodic_checkpoint_folder = None,

                        threshold_evaluation_early_stop = None, #[lower percentile, upper percentile]
                        threshold_evaluation_scaling = 4, # [0,inf) lower means longer to get to upper, higher reacher upper sooner
                        min_history_threshold = 20,
                        selection_evaluation_early_stop = None, #[lower percentile, upper percentile]
                        selection_evaluation_scaling = 4, # [0,inf) lower means longer to get to upper, higher reacher upper sooner

                        scorers_early_stop_tol = 0.001,
                        other_objectives_early_stop_tol =None,
                        early_stop = None,

                        warm_start = False,
                        memory = None,
                        cross_val_predict_cv = 0, #TODO: crossval predict for transformers and selectors?
                        
                        budget_range = None, #[.2, 1] [start budget, end budget]
                        budget_scaling = .8, # [0,1]
                        generations_until_end_budget = 1, #    


                        preprocessing = False,  

                        validation_strategy = "none",
                        validation_fraction = .2,

                        subset_column = None,

                        stepwise_steps = 5,
                        client=None,
        ):
        super(TPOTClassifier,self).__init__(
            scorers = scorers, #these get passed into CV
            scorers_weights = scorers_weights,
            classification=classification,
            population_size = population_size, #alternatively, an instance of evolver could be passed in?
            generations = generations,
            initial_population_size = initial_population_size,
            population_scaling = population_scaling,
            generations_until_end_population = generations_until_end_population,
            callback=callback,
            n_jobs=n_jobs,
            cv = cv,
            verbose = verbose, #TODO
            other_objective_functions=other_objective_functions, #tpot2.estimator_objective_functions.number_of_nodes_objective],
            other_objective_functions_weights = other_objective_functions_weights,
            bigger_is_better = bigger_is_better,
            evolver = evolver,
            evolver_params = evolver_params,
            max_depth = max_depth,
            max_size = max_size,
            max_children = max_children,
            root_config_dict= root_config_dict,
            inner_config_dict=inner_config_dict,
            leaf_config_dict= leaf_config_dict,
            subsets = subsets,
            max_time_seconds=max_time_seconds,
            max_eval_time_seconds=max_eval_time_seconds, #TODO auto set a timer? Should this be none?
            memory_limit=memory_limit,
            n_initial_optimizations = n_initial_optimizations,
            optimization_cv = optimization_cv,
            max_optimize_time_seconds=max_optimize_time_seconds,
            optimization_steps = optimization_steps,
            periodic_checkpoint_folder = periodic_checkpoint_folder,
            threshold_evaluation_early_stop = threshold_evaluation_early_stop, #[lower percentile, upper percentile]
            threshold_evaluation_scaling = threshold_evaluation_scaling, # [0,inf) lower means longer to get to upper, higher reacher upper sooner
            min_history_threshold = min_history_threshold,
            selection_evaluation_early_stop = selection_evaluation_early_stop, #[lower percentile, upper percentile]
            selection_evaluation_scaling = selection_evaluation_scaling, # [0,inf) lower means longer to get to upper, higher reacher upper sooner
            scorers_early_stop_tol = scorers_early_stop_tol,
            other_objectives_early_stop_tol =other_objectives_early_stop_tol,
            early_stop = early_stop,
            warm_start = warm_start,
            memory = memory,
            cross_val_predict_cv = cross_val_predict_cv, #TODO: crossval predict for transformers and selectors?
            budget_range = budget_range, #[.2, 1] [start budget, end budget]
            budget_scaling = budget_scaling, # [0,1]
            generations_until_end_budget = generations_until_end_budget, #
            preprocessing = preprocessing,
            validation_strategy = validation_strategy,
            validation_fraction = validation_fraction,
            subset_column = subset_column,
            stepwise_steps = stepwise_steps,
            client=client,
        )


    def predict(self, X, **predict_params):
        check_is_fitted(self)
        #X = check_array(X)
        return self.fitted_pipeline_.predict(X,**predict_params)