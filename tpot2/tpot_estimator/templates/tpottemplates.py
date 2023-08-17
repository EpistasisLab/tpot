import tpot2
import numpy as np
from ..estimator import TPOTEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tpot2.selectors import survival_select_NSGA2, tournament_selection_dominated
#TODO These do not follow sklearn conventions of __init__

class TPOTRegressor(TPOTEstimator):
    def __init__(       self,
                        scorers=['neg_mean_squared_error'], 
                        scorers_weights=[1],
                        other_objective_functions=[], #tpot2.objectives.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights = [],
                        objective_function_names = None,
                        bigger_is_better = True,
                        max_size = np.inf, 
                        linear_pipeline = False,
                        root_config_dict= 'Auto',
                        inner_config_dict=["selectors", "transformers"],
                        leaf_config_dict= None,                        
                        cross_val_predict_cv = 0,
                        categorical_features = None,
                        subsets = None,
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
                        client = None
        ):
        """
        See TPOTEstimator for documentation
        """
        super(TPOTRegressor,self).__init__(
                        scorers=scorers, 
                        scorers_weights=scorers_weights,
                        cv=5,
                        other_objective_functions=other_objective_functions, #tpot2.objectives.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights = other_objective_functions_weights,
                        objective_function_names = objective_function_names,
                        bigger_is_better = bigger_is_better,
                        max_size = max_size, 
                        linear_pipeline = linear_pipeline,
                        root_config_dict = root_config_dict,
                        inner_config_dict=inner_config_dict,
                        leaf_config_dict= leaf_config_dict,                        
                        cross_val_predict_cv = cross_val_predict_cv,
                        categorical_features = categorical_features,
                        subsets = subsets,
                        memory = memory,
                        preprocessing = preprocessing,
                        max_time_seconds=max_time_seconds, 
                        max_eval_time_seconds=max_eval_time_seconds, 
                        n_jobs=n_jobs,
                        validation_strategy = validation_strategy,
                        validation_fraction = validation_fraction, 
                        early_stop = early_stop,
                        warm_start = warm_start,
                        periodic_checkpoint_folder = periodic_checkpoint_folder, 
                        verbose = verbose,
                        classification=False,
                        memory_limit = memory_limit,
                        client = client
)


class TPOTClassifier(TPOTEstimator):
    def __init__(       self,
                        scorers=['roc_auc_ovr'], 
                        scorers_weights=[1],
                        other_objective_functions=[], #tpot2.objectives.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights = [],
                        objective_function_names = None,
                        bigger_is_better = True,
                        max_size = np.inf, 
                        linear_pipeline = False,
                        root_config_dict= 'Auto',
                        inner_config_dict=["selectors", "transformers"],
                        leaf_config_dict= None,                        
                        cross_val_predict_cv = 0,
                        categorical_features = None,
                        subsets = None,
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
                        client = None
                        
        ):
        """
        See TPOTEstimator for documentation
        """
        super(TPOTClassifier,self).__init__(
                        scorers=scorers, 
                        scorers_weights=scorers_weights,
                        cv = 5,
                        other_objective_functions=other_objective_functions, #tpot2.objectives.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights = other_objective_functions_weights,
                        objective_function_names = objective_function_names,
                        bigger_is_better = bigger_is_better,
                        max_size = max_size, 
                        linear_pipeline = linear_pipeline,
                        root_config_dict = root_config_dict,
                        inner_config_dict=inner_config_dict,
                        leaf_config_dict= leaf_config_dict,                        
                        cross_val_predict_cv = cross_val_predict_cv,
                        categorical_features = categorical_features,
                        subsets = subsets,
                        memory = memory,
                        preprocessing = preprocessing,
                        max_time_seconds=max_time_seconds, 
                        max_eval_time_seconds=max_eval_time_seconds, 
                        n_jobs=n_jobs,
                        validation_strategy = validation_strategy,
                        validation_fraction = validation_fraction, 
                        early_stop = early_stop,
                        warm_start = warm_start,
                        periodic_checkpoint_folder = periodic_checkpoint_folder, 
                        verbose = verbose,
                        classification=True,
                        memory_limit = memory_limit,
                        client = client
        )


    def predict(self, X, **predict_params):
        check_is_fitted(self)
        #X=check_array(X)
        return self.fitted_pipeline_.predict(X,**predict_params)
