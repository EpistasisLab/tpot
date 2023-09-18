from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import available_if
import numpy as np
import sklearn.metrics
import tpot2.config
from sklearn.utils.validation import check_is_fitted
from tpot2.selectors import survival_select_NSGA2, tournament_selection_dominated
from sklearn.preprocessing import LabelEncoder 

import pandas as pd
from sklearn.model_selection import train_test_split
import tpot2
from dask.distributed import Client
from dask.distributed import LocalCluster
from sklearn.preprocessing import LabelEncoder
import warnings
import math
from .estimator_utils import *

from dask import config as cfg


def set_dask_settings():
    cfg.set({'distributed.scheduler.worker-ttl': None})
    cfg.set({'distributed.scheduler.allowed-failures':1})




#TODO inherit from _BaseComposition?
class TPOTEstimator(BaseEstimator):
    def __init__(self,  scorers, 
                        scorers_weights,
                        classification,
                        cv = 5,
                        other_objective_functions=[],
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
                        population_size = 50,
                        initial_population_size = None,
                        population_scaling = .5, 
                        generations_until_end_population = 1,  
                        generations = None,
                        max_time_seconds=3600, 
                        max_eval_time_seconds=60*10, 
                        validation_strategy = "none",
                        validation_fraction = .2,
                        disable_label_encoder = False,
                        
                        #early stopping parameters 
                        early_stop = None,
                        scorers_early_stop_tol = 0.001,
                        other_objectives_early_stop_tol =None,
                        threshold_evaluation_early_stop = None, 
                        threshold_evaluation_scaling = .5,
                        selection_evaluation_early_stop = None, 
                        selection_evaluation_scaling = .5, 
                        min_history_threshold = 20,
                        
                        #evolver parameters
                        survival_percentage = 1,
                        crossover_probability=.2,
                        mutate_probability=.7,
                        mutate_then_crossover_probability=.05,
                        crossover_then_mutate_probability=.05,
                        survival_selector = survival_select_NSGA2,
                        parent_selector = tournament_selection_dominated,
                        
                        #budget parameters
                        budget_range = None,
                        budget_scaling = .5,
                        generations_until_end_budget = 1,  
                        stepwise_steps = 5,
                        

                        optuna_optimize_pareto_front = False,
                        optuna_optimize_pareto_front_trials = 100,
                        optuna_optimize_pareto_front_timeout = 60*10,
                        optuna_storage = "sqlite:///optuna.db",
                        
                        #dask parameters
                        n_jobs=1,
                        memory_limit = "4GB",
                        client = None,
                        processes = True,
                        
                        #debugging and logging parameters
                        warm_start = False,
                        subset_column = None,
                        periodic_checkpoint_folder = None, 
                        callback = None,
                        
                        verbose = 0,
                        scatter = True,

                        ):
                        
        '''
        An sklearn baseestimator that uses genetic programming to optimize a pipeline.
        
        Parameters
        ----------
        
        scorers : (list, scorer)
            A scorer or list of scorers to be used in the cross-validation process. 
            see https://scikit-learn.org/stable/modules/model_evaluation.html
        
        scorers_weights : list
            A list of weights to be applied to the scorers during the optimization process.
        
        classification : bool
            If True, the problem is treated as a classification problem. If False, the problem is treated as a regression problem.
            Used to determine the CV strategy.
        
        cv : int, cross-validator
            - (int): Number of folds to use in the cross-validation process. By uses the sklearn.model_selection.KFold cross-validator for regression and StratifiedKFold for classification. In both cases, shuffled is set to True.
            - (sklearn.model_selection.BaseCrossValidator): A cross-validator to use in the cross-validation process.
                - max_depth (int): The maximum depth from any node to the root of the pipelines to be generated.
        
        other_objective_functions : list, default=[]
            A list of other objective functions to apply to the pipeline. The function takes a single parameter for the graphpipeline estimator and returns either a single score or a list of scores.
        
        other_objective_functions_weights : list, default=[]
            A list of weights to be applied to the other objective functions.
        
        objective_function_names : list, default=None
            A list of names to be applied to the objective functions. If None, will use the names of the objective functions.
        
        bigger_is_better : bool, default=True
            If True, the objective function is maximized. If False, the objective function is minimized. Use negative weights to reverse the direction.

        
        max_size : int, default=np.inf
            The maximum number of nodes of the pipelines to be generated.
        
        linear_pipeline : bool, default=False
            If True, the pipelines generated will be linear. If False, the pipelines generated will be directed acyclic graphs.
        
        root_config_dict : dict, default='auto'
            The configuration dictionary to use for the root node of the model.
            If 'auto', will use "classifiers" if classification=True, else "regressors".
            - 'selectors' : A selection of sklearn Selector methods.
            - 'classifiers' : A selection of sklearn Classifier methods.
            - 'regressors' : A selection of sklearn Regressor methods.
            - 'transformers' : A selection of sklearn Transformer methods.
            - 'arithmetic_transformer' : A selection of sklearn Arithmetic Transformer methods that replicate symbolic classification/regression operators.
            - 'passthrough' : A node that just passes though the input. Useful for passing through raw inputs into inner nodes.
            - 'feature_set_selector' : A selector that pulls out specific subsets of columns from the data. Only well defined as a leaf.
                                        Subsets are set with the subsets parameter.
            - 'skrebate' : Includes ReliefF, SURF, SURFstar, MultiSURF.
            - 'MDR' : Includes MDR.
            - 'ContinuousMDR' : Includes ContinuousMDR.
            - 'genetic encoders' : Includes Genetic Encoder methods as used in AutoQTL.
            - 'FeatureEncodingFrequencySelector': Includes FeatureEncodingFrequencySelector method as used in AutoQTL.
            - list : a list of strings out of the above options to include the corresponding methods in the configuration dictionary.
        
        inner_config_dict : dict, default=["selectors", "transformers"]
            The configuration dictionary to use for the inner nodes of the model generation.
            Default ["selectors", "transformers"]
            - 'selectors' : A selection of sklearn Selector methods.
            - 'classifiers' : A selection of sklearn Classifier methods.
            - 'regressors' : A selection of sklearn Regressor methods.
            - 'transformers' : A selection of sklearn Transformer methods.
            - 'arithmetic_transformer' : A selection of sklearn Arithmetic Transformer methods that replicate symbolic classification/regression operators.
            - 'passthrough' : A node that just passes though the input. Useful for passing through raw inputs into inner nodes.
            - 'feature_set_selector' : A selector that pulls out specific subsets of columns from the data. Only well defined as a leaf.
                                        Subsets are set with the subsets parameter.
            - 'skrebate' : Includes ReliefF, SURF, SURFstar, MultiSURF.
            - 'MDR' : Includes MDR.
            - 'ContinuousMDR' : Includes ContinuousMDR.
            - 'genetic encoders' : Includes Genetic Encoder methods as used in AutoQTL.
            - 'FeatureEncodingFrequencySelector': Includes FeatureEncodingFrequencySelector method as used in AutoQTL.
            - list : a list of strings out of the above options to include the corresponding methods in the configuration dictionary.
            - None : If None and max_depth>1, the root_config_dict will be used for the inner nodes as well.
        
        leaf_config_dict : dict, default=None 
            The configuration dictionary to use for the leaf node of the model. If set, leaf nodes must be from this dictionary.
            Otherwise leaf nodes will be generated from the root_config_dict. 
            Default None
            - 'selectors' : A selection of sklearn Selector methods.
            - 'classifiers' : A selection of sklearn Classifier methods.
            - 'regressors' : A selection of sklearn Regressor methods.
            - 'transformers' : A selection of sklearn Transformer methods.
            - 'arithmetic_transformer' : A selection of sklearn Arithmetic Transformer methods that replicate symbolic classification/regression operators.
            - 'passthrough' : A node that just passes though the input. Useful for passing through raw inputs into inner nodes.
            - 'feature_set_selector' : A selector that pulls out specific subsets of columns from the data. Only well defined as a leaf.
                                        Subsets are set with the subsets parameter.
            - 'skrebate' : Includes ReliefF, SURF, SURFstar, MultiSURF.
            - 'MDR' : Includes MDR.
            - 'ContinuousMDR' : Includes ContinuousMDR.
            - 'genetic encoders' : Includes Genetic Encoder methods as used in AutoQTL.
            - 'FeatureEncodingFrequencySelector': Includes FeatureEncodingFrequencySelector method as used in AutoQTL.
            - list : a list of strings out of the above options to include the corresponding methods in the configuration dictionary.
            - None : If None, a leaf will not be required (i.e. the pipeline can be a single root node). Leaf nodes will be generated from the inner_config_dict.
        
        cross_val_predict_cv : int, default=0
            Number of folds to use for the cross_val_predict function for inner classifiers and regressors. Estimators will still be fit on the full dataset, but the following node will get the outputs from cross_val_predict.
            
            - 0-1 : When set to 0 or 1, the cross_val_predict function will not be used. The next layer will get the outputs from fitting and transforming the full dataset.
            - >=2 : When fitting pipelines with inner classifiers or regressors, they will still be fit on the full dataset. 
                    However, the output to the next node will come from cross_val_predict with the specified number of folds.
         
        categorical_features: list or None
            Categorical columns to inpute and/or one hot encode during the preprocessing step. Used only if preprocessing is not False.
            - None : If None, TPOT2 will automatically use object columns in pandas dataframes as objects for one hot encoding in preprocessing.
            - List of categorical features. If X is a dataframe, this should be a list of column names. If X is a numpy array, this should be a list of column indices

        subsets : str or list, default=None
            Sets the subsets that the FeatureSetSeletor will select from if set as an option in one of the configuration dictionaries.
            - str : If a string, it is assumed to be a path to a csv file with the subsets. 
                The first column is assumed to be the name of the subset and the remaining columns are the features in the subset.
            - list or np.ndarray : If a list or np.ndarray, it is assumed to be a list of subsets.
            - None : If None, each column will be treated as a subset. One column will be selected per subset.
            If subsets is None, each column will be treated as a subset. One column will be selected per subset.


        memory: Memory object or string, default=None
            If supplied, pipeline will cache each transformer after calling fit. This feature
            is used to avoid computing the fit transformers within a pipeline if the parameters
            and input data are identical with another fitted pipeline during optimization process.
            - String 'auto':
                TPOT uses memory caching with a temporary directory and cleans it up upon shutdown.
            - String path of a caching directory
                TPOT uses memory caching with the provided directory and TPOT does NOT clean
                the caching directory up upon shutdown. If the directory does not exist, TPOT will
                create it.
            - Memory object:
                TPOT uses the instance of joblib.Memory for memory caching,
                and TPOT does NOT clean the caching directory up upon shutdown.
            - None:
                TPOT does not use memory caching.

        preprocessing : bool or BaseEstimator/Pipeline, 
            EXPERIMENTAL
            A pipeline that will be used to preprocess the data before CV.
            - bool : If True, will use a default preprocessing pipeline.
            - Pipeline : If an instance of a pipeline is given, will use that pipeline as the preprocessing pipeline.
        
        population_size : int, default=50
            Size of the population
        
        initial_population_size : int, default=None
            Size of the initial population. If None, population_size will be used.
        
        population_scaling : int, default=0.5
            Scaling factor to use when determining how fast we move the threshold moves from the start to end percentile.
        
        generations_until_end_population : int, default=1  
            Number of generations until the population size reaches population_size            
        
        generations : int, default=50
            Number of generations to run
            
        max_time_seconds : float, default=float("inf")
            Maximum time to run the optimization. If none or inf, will run until the end of the generations.
        
        max_eval_time_seconds : float, default=60*5
            Maximum time to evaluate a single individual. If none or inf, there will be no time limit per evaluation.
            
        validation_strategy : str, default='none'
            EXPERIMENTAL The validation strategy to use for selecting the final pipeline from the population. TPOT2 may overfit the cross validation score. A second validation set can be used to select the final pipeline.
            - 'auto' : Automatically determine the validation strategy based on the dataset shape.
            - 'reshuffled' : Use the same data for cross validation and final validation, but with different splits for the folds. This is the default for small datasets. 
            - 'split' : Use a separate validation set for final validation. Data will be split according to validation_fraction. This is the default for medium datasets. 
            - 'none' : Do not use a separate validation set for final validation. Select based on the original cross-validation score. This is the default for large datasets.

        validation_fraction : float, default=0.2
          EXPERIMENTAL The fraction of the dataset to use for the validation set when validation_strategy is 'split'. Must be between 0 and 1.
        
        disable_label_encoder : bool, default=False
            If True, TPOT will check if the target needs to be relabeled to be sequential ints from 0 to N. This is necessary for XGBoost compatibility. If the labels need to be encoded, TPOT2 will use sklearn.preprocessing.LabelEncoder to encode the labels. The encoder can be accessed via the self.label_encoder_ attribute.
            If False, no additional label encoders will be used.

        early_stop : int, default=None
            Number of generations without improvement before early stopping. All objectives must have converged within the tolerance for this to be triggered.
        
        scorers_early_stop_tol : 
            -list of floats
                list of tolerances for each scorer. If the difference between the best score and the current score is less than the tolerance, the individual is considered to have converged
                If an index of the list is None, that item will not be used for early stopping
            -int 
                If an int is given, it will be used as the tolerance for all objectives
        
        other_objectives_early_stop_tol : 
            -list of floats
                list of tolerances for each of the other objective function. If the difference between the best score and the current score is less than the tolerance, the individual is considered to have converged
                If an index of the list is None, that item will not be used for early stopping
            -int 
                If an int is given, it will be used as the tolerance for all objectives
    
        threshold_evaluation_early_stop : list [start, end], default=None
            starting and ending percentile to use as a threshold for the evaluation early stopping.
            Values between 0 and 100.
        
        threshold_evaluation_scaling : float [0,inf), default=0.5
            A scaling factor to use when determining how fast we move the threshold moves from the start to end percentile.
            Must be greater than zero. Higher numbers will move the threshold to the end faster.
        
        selection_evaluation_early_stop : list, default=None
            A lower and upper percent of the population size to select each round of CV.
            Values between 0 and 1.
        
        selection_evaluation_scaling : float, default=0.5 
            A scaling factor to use when determining how fast we move the threshold moves from the start to end percentile.
            Must be greater than zero. Higher numbers will move the threshold to the end faster.    
        
        min_history_threshold : int, default=0
            The minimum number of previous scores needed before using threshold early stopping.
        
        survival_percentage : float, default=1
            Percentage of the population size to utilize for mutation and crossover at the beginning of the generation. The rest are discarded. Individuals are selected with the selector passed into survival_selector. The value of this parameter must be between 0 and 1, inclusive. 
            For example, if the population size is 100 and the survival percentage is .5, 50 individuals will be selected with NSGA2 from the existing population. These will be used for mutation and crossover to generate the next 100 individuals for the next generation. The remainder are discarded from the live population. In the next generation, there will now be the 50 parents + the 100 individuals for a total of 150. Surivival percentage is based of the population size parameter and not the existing population size (current population size when using successive halving). Therefore, in the next generation we will still select 50 individuals from the currently existing 150.
        
        crossover_probability : float, default=.2
            Probability of generating a new individual by crossover between two individuals.
        
        mutate_probability : float, default=.7
            Probability of generating a new individual by crossover between one individuals.
        
        mutate_then_crossover_probability : float, default=.05
            Probability of generating a new individual by mutating two individuals followed by crossover.
        
        crossover_then_mutate_probability : float, default=.05
            Probability of generating a new individual by crossover between two individuals followed by a mutation of the resulting individual.
        
        survival_selector : function, default=survival_select_NSGA2
            Function to use to select individuals for survival. Must take a matrix of scores and return selected indexes.
            Used to selected population_size * survival_percentage individuals at the start of each generation to use for mutation and crossover.
        
        parent_selector : function, default=parent_select_NSGA2
            Function to use to select pairs parents for crossover and individuals for mutation. Must take a matrix of scores and return selected indexes.
        
        budget_range : list [start, end], default=None
            A starting and ending budget to use for the budget scaling.
        
        budget_scaling float : [0,1], default=0.5
            A scaling factor to use when determining how fast we move the budget from the start to end budget.
        
        generations_until_end_budget : int, default=1
            The number of generations to run before reaching the max budget.
        
        stepwise_steps : int, default=1
            The number of staircase steps to take when scaling the budget and population size.
        
            
        n_jobs : int, default=1
            Number of processes to run in parallel.
        
        memory_limit : str, default="4GB"
            Memory limit for each job. See Dask [LocalCluster documentation](https://distributed.dask.org/en/stable/api.html#distributed.Client) for more information.
        
        client : dask.distributed.Client, default=None
            A dask client to use for parallelization. If not None, this will override the n_jobs and memory_limit parameters. If None, will create a new client with num_workers=n_jobs and memory_limit=memory_limit. 
        
        processes : bool, default=True
            If True, will use multiprocessing to parallelize the optimization process. If False, will use threading.
            True seems to perform better. However, False is required for interactive debugging.
            
          
        warm_start : bool, default=False
            If True, will use the continue the evolutionary algorithm from the last generation of the previous run.
         
        subset_column : str or int, default=None
            EXPERIMENTAL The column to use for the subset selection. Must also pass in unique_subset_values to GraphIndividual to function.
        
        periodic_checkpoint_folder : str, default=None
            Folder to save the population to periodically. If None, no periodic saving will be done.
            If provided, training will resume from this checkpoint.
        
        callback : tpot2.CallBackInterface, default=None
            Callback object. Not implemented
            
        verbose : int, default=1 
            How much information to print during the optimization process. Higher values include the information from lower values.
            0. nothing
            1. progress bar
            
            3. best individual
            4. warnings
            >=5. full warnings trace
            6. evaluations progress bar. (Temporary: This used to be 2. Currently, using evaluation progress bar may prevent some instances were we terminate a generation early due to it reaching max_time_seconds in the middle of a generation OR a pipeline failed to be terminated normally and we need to manually terminate it.)
        
            
        Attributes
        ----------

        fitted_pipeline_ : GraphPipeline
            A fitted instance of the GraphPipeline that inherits from sklearn BaseEstimator. This is fitted on the full X, y passed to fit.

        evaluated_individuals : A pandas data frame containing data for all evaluated individuals in the run. 
            Columns: 
            - *objective functions : The first few columns correspond to the passed in scorers and objective functions
            - Parents : A tuple containing the indexes of the pipelines used to generate the pipeline of that row. If NaN, this pipeline was generated randomly in the initial population.
            - Variation_Function : Which variation function was used to mutate or crossover the parents. If NaN, this pipeline was generated randomly in the initial population.
            - Individual : The internal representation of the individual that is used during the evolutionary algorithm. This is not an sklearn BaseEstimator.
            - Generation : The generation the pipeline first appeared. 
            - Pareto_Front	: The nondominated front that this pipeline belongs to. 0 means that its scores is not strictly dominated by any other individual. 
                            To save on computational time, the best frontier is updated iteratively each generation. 
                            The pipelines with the 0th pareto front do represent the exact best frontier. However, the pipelines with pareto front >= 1 are only in reference to the other pipelines in the final population.
                            All other pipelines are set to NaN. 
            - Instance	: The unfitted GraphPipeline BaseEstimator. 
            - *validation objective functions : Objective function scores evaluated on the validation set.
            - Validation_Pareto_Front : The full pareto front calculated on the validation set. This is calculated for all pipelines with Pareto_Front equal to 0. Unlike the Pareto_Front which only calculates the frontier and the final population, the Validation Pareto Front is calculated for all pipelines tested on the validation set.
            
        pareto_front : The same pandas dataframe as evaluated individuals, but containing only the frontier pareto front pipelines.
        '''

        # sklearn BaseEstimator must have a corresponding attribute for each parameter.
        # These should not be modified once set.

        self.scorers = scorers
        self.scorers_weights = scorers_weights
        self.classification = classification
        self.cv = cv
        self.other_objective_functions = other_objective_functions
        self.other_objective_functions_weights = other_objective_functions_weights
        self.objective_function_names = objective_function_names
        self.bigger_is_better = bigger_is_better
        self.max_size = max_size
        self.linear_pipeline = linear_pipeline
        self.root_config_dict= root_config_dict
        self.inner_config_dict= inner_config_dict
        self.leaf_config_dict= leaf_config_dict
        self.cross_val_predict_cv = cross_val_predict_cv
        self.categorical_features = categorical_features
        self.subsets = subsets
        self.memory = memory
        self.preprocessing = preprocessing
        self.validation_strategy = validation_strategy
        self.validation_fraction = validation_fraction
        self.disable_label_encoder = disable_label_encoder
        self.population_size = population_size
        self.initial_population_size = initial_population_size
        self.population_scaling = population_scaling
        self.generations_until_end_population = generations_until_end_population
        self.generations = generations
        self.early_stop = early_stop
        self.scorers_early_stop_tol = scorers_early_stop_tol
        self.other_objectives_early_stop_tol = other_objectives_early_stop_tol
        self.max_time_seconds = max_time_seconds 
        self.max_eval_time_seconds = max_eval_time_seconds
        self.n_jobs= n_jobs
        self.memory_limit = memory_limit
        self.client = client
        self.survival_percentage = survival_percentage
        self.crossover_probability = crossover_probability
        self.mutate_probability = mutate_probability
        self.mutate_then_crossover_probability= mutate_then_crossover_probability
        self.crossover_then_mutate_probability= crossover_then_mutate_probability
        self.survival_selector=survival_selector
        self.parent_selector=parent_selector
        self.budget_range = budget_range
        self.budget_scaling = budget_scaling
        self.generations_until_end_budget = generations_until_end_budget
        self.stepwise_steps = stepwise_steps
        self.threshold_evaluation_early_stop =threshold_evaluation_early_stop
        self.threshold_evaluation_scaling =  threshold_evaluation_scaling
        self.min_history_threshold = min_history_threshold
        self.selection_evaluation_early_stop = selection_evaluation_early_stop
        self.selection_evaluation_scaling =  selection_evaluation_scaling
        self.warm_start = warm_start
        self.subset_column = subset_column
        self.verbose = verbose
        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.callback = callback
        self.processes = processes


        self.scatter = scatter

        self.optuna_optimize_pareto_front = optuna_optimize_pareto_front
        self.optuna_optimize_pareto_front_trials = optuna_optimize_pareto_front_trials
        self.optuna_optimize_pareto_front_timeout = optuna_optimize_pareto_front_timeout
        self.optuna_storage = optuna_storage

        #Initialize other used params


        if self.initial_population_size is None:
            self._initial_population_size = self.population_size
        else:
            self._initial_population_size = self.initial_population_size

        if isinstance(self.scorers, str):
            self._scorers = [self.scorers]

        elif callable(self.scorers):
            self._scorers = [self.scorers]
        else:
            self._scorers = self.scorers
        
        self._scorers = [sklearn.metrics.get_scorer(scoring) for scoring in self._scorers]
        self._scorers_early_stop_tol = self.scorers_early_stop_tol
        
        self._evolver = tpot2.evolvers.BaseEvolver
        
        self.objective_function_weights = [*scorers_weights, *other_objective_functions_weights]
        

        if self.objective_function_names is None:
            obj_names = [f.__name__ for f in other_objective_functions]
        else:
            obj_names = self.objective_function_names
        self.objective_names = [f._score_func.__name__ if hasattr(f,"_score_func") else f.__name__ for f in self._scorers] + obj_names
        
        
        if not isinstance(self.other_objectives_early_stop_tol, list):
            self._other_objectives_early_stop_tol = [self.other_objectives_early_stop_tol for _ in range(len(self.other_objective_functions))]
        else:
            self._other_objectives_early_stop_tol = self.other_objectives_early_stop_tol

        if not isinstance(self._scorers_early_stop_tol, list):
            self._scorers_early_stop_tol = [self._scorers_early_stop_tol for _ in range(len(self._scorers))]
        else:
            self._scorers_early_stop_tol = self._scorers_early_stop_tol

        self.early_stop_tol = [*self._scorers_early_stop_tol, *self._other_objectives_early_stop_tol]
        
        self._evolver_instance = None
        self.evaluated_individuals = None


        self.label_encoder_ = None


        set_dask_settings()


    def fit(self, X, y):
        if self.client is not None: #If user passed in a client manually
           _client = self.client
        else:

            if self.verbose >= 4:
                silence_logs = 30
            elif self.verbose >=5:
                silence_logs = 40
            else:
                silence_logs = 50
            cluster = LocalCluster(n_workers=self.n_jobs, #if no client is passed in and no global client exists, create our own
                    threads_per_worker=1,
                    processes=self.processes,
                    silence_logs=silence_logs,
                    memory_limit=self.memory_limit)
            _client = Client(cluster)

        if self.classification and not self.disable_label_encoder and not check_if_y_is_encoded(y):
            warnings.warn("Labels are not encoded as ints from 0 to N. For compatibility with some classifiers such as sklearn, TPOT has encoded y with the sklearn LabelEncoder. When using pipelines outside the main TPOT estimator class, you can encode the labels with est.label_encoder_")
            self.label_encoder_ = LabelEncoder()  
            y = self.label_encoder_.fit_transform(y)  

        self.evaluated_individuals = None
        #determine validation strategy
        if self.validation_strategy == 'auto':
            nrows = X.shape[0]
            ncols = X.shape[1]

            if nrows/ncols < 20:
                validation_strategy = 'reshuffled'
            elif nrows/ncols < 100:
                validation_strategy = 'split'
            else:
                validation_strategy = 'none'
        else:
            validation_strategy = self.validation_strategy

        if validation_strategy == 'split':
            if self.classification:
                X, X_val, y, y_val = train_test_split(X, y, test_size=self.validation_fraction, stratify=y, random_state=42)
            else:
                X, X_val, y, y_val = train_test_split(X, y, test_size=self.validation_fraction, random_state=42)


        X_original = X
        y_original = y
        if isinstance(self.cv, int) or isinstance(self.cv, float):
            n_folds = self.cv
        else:
            n_folds = self.cv.get_n_splits(X, y)

        if self.classification:
            X, y = remove_underrepresented_classes(X, y, n_folds)
        
        if self.preprocessing:
            #X = pd.DataFrame(X)

            #TODO: check if there are missing values in X before imputation. If not, don't include imputation in pipeline. Check if there are categorical columns. If not, don't include one hot encoding in pipeline
            if isinstance(X, pd.DataFrame): #pandas dataframe
                if self.categorical_features is not None:
                    X[self.categorical_features] = X[self.categorical_features].astype(object)
                self._preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot2.builtin_modules.ColumnSimpleImputer("categorical", strategy='most_frequent'), #impute categorical columns
                                                                            tpot2.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean'),              #impute numeric columns
                                                                            tpot2.builtin_modules.ColumnOneHotEncoder("categorical", min_frequency=0.0001))     #one hot encode categorical columns
                X = self._preprocessing_pipeline.fit_transform(X)
            else:
                if self.categorical_features is not None: #numpy array and categorical columns specified
                    self._preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot2.builtin_modules.ColumnSimpleImputer(self.categorical_features, strategy='most_frequent'),   #impute categorical columns
                                                                            tpot2.builtin_modules.ColumnSimpleImputer("all", strategy='mean'),                                      #impute remaining numeric columns
                                                                            tpot2.builtin_modules.ColumnOneHotEncoder(self.categorical_features, min_frequency=0.0001))             #one hot encode categorical columns
                else: #numpy array and no categorical columns specified, just do imputation
                    self._preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot2.builtin_modules.ColumnSimpleImputer("all", strategy='mean'))   


        else:
            self._preprocessing_pipeline = None

        #_, y = sklearn.utils.check_X_y(X, y, y_numeric=True)

        #Set up the configuation dictionaries and the search spaces



        n_samples= int(math.floor(X.shape[0]/n_folds))
        n_features=X.shape[1]

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
        else:
            self.feature_names = None

        if self.root_config_dict == 'Auto':
            if self.classification:
                n_classes = len(np.unique(y))
                root_config_dict = get_configuration_dictionary("classifiers", n_samples, n_features, self.classification, subsets=self.subsets, feature_names=self.feature_names, n_classes=n_classes)
            else:
                root_config_dict = get_configuration_dictionary("regressors", n_samples, n_features, self.classification,subsets=self.subsets, feature_names=self.feature_names)
        else:
            root_config_dict = get_configuration_dictionary(self.root_config_dict, n_samples, n_features, self.classification, subsets=self.subsets,feature_names=self.feature_names)

        inner_config_dict = get_configuration_dictionary(self.inner_config_dict, n_samples, n_features, self.classification,subsets=self.subsets, feature_names=self.feature_names)
        leaf_config_dict = get_configuration_dictionary(self.leaf_config_dict, n_samples, n_features, self.classification, subsets=self.subsets, feature_names=self.feature_names)




        #check if self.cv is a number
        if isinstance(self.cv, int) or isinstance(self.cv, float):
            if self.classification:
                self.cv_gen = sklearn.model_selection.StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
            else:
                self.cv_gen = sklearn.model_selection.KFold(n_splits=self.cv, shuffle=True, random_state=42)

        else:
            self.cv_gen = sklearn.model_selection.check_cv(self.cv, y, classifier=self.classification)
        
        def objective_function(pipeline_individual, 
                                            X, 
                                            y,
                                            is_classification=self.classification,
                                            scorers= self._scorers, 
                                            cv=self.cv_gen, 
                                            other_objective_functions=self.other_objective_functions,
                                            memory=self.memory, 
                                            cross_val_predict_cv=self.cross_val_predict_cv, 
                                            subset_column=self.subset_column, 
                                            **kwargs): 
            return objective_function_generator(
                pipeline_individual,
                X, 
                y, 
                is_classification=is_classification,
                scorers= scorers, 
                cv=cv, 
                other_objective_functions=other_objective_functions,
                memory=memory, 
                cross_val_predict_cv=cross_val_predict_cv, 
                subset_column=subset_column,
                **kwargs,
            )

        self.individual_generator_instance = tpot2.individual_representations.graph_pipeline_individual.estimator_graph_individual_generator(   
                                                            inner_config_dict=inner_config_dict,
                                                            root_config_dict=root_config_dict,
                                                            leaf_config_dict=leaf_config_dict,
                                                            max_size = self.max_size,
                                                            linear_pipeline=self.linear_pipeline,
                                                                )

        if self.threshold_evaluation_early_stop is not None or self.selection_evaluation_early_stop is not None:
            evaluation_early_stop_steps = self.cv
        else:
            evaluation_early_stop_steps = None

        if self.scatter:
            X_future = _client.scatter(X)
            y_future = _client.scatter(y)
        else:
            X_future = X
            y_future = y

        #If warm start and we have an evolver instance, use the existing one
        if not(self.warm_start and self._evolver_instance is not None):
            self._evolver_instance = self._evolver(   individual_generator=self.individual_generator_instance, 
                                            objective_functions= [objective_function],
                                            objective_function_weights = self.objective_function_weights,
                                            objective_names=self.objective_names,
                                            bigger_is_better = self.bigger_is_better,
                                            population_size= self.population_size,
                                            generations=self.generations,
                                            initial_population_size = self._initial_population_size,
                                            n_jobs=self.n_jobs,
                                            verbose = self.verbose,
                                            max_time_seconds =      self.max_time_seconds ,
                                            max_eval_time_seconds = self.max_eval_time_seconds,

                                            periodic_checkpoint_folder = self.periodic_checkpoint_folder,
                                            threshold_evaluation_early_stop = self.threshold_evaluation_early_stop,
                                            threshold_evaluation_scaling =  self.threshold_evaluation_scaling,
                                            min_history_threshold = self.min_history_threshold,

                                            selection_evaluation_early_stop = self.selection_evaluation_early_stop,
                                            selection_evaluation_scaling =  self.selection_evaluation_scaling,
                                            evaluation_early_stop_steps = evaluation_early_stop_steps,

                                            early_stop_tol = self.early_stop_tol,
                                            early_stop= self.early_stop,
                                            
                                            budget_range = self.budget_range,
                                            budget_scaling = self.budget_scaling,
                                            generations_until_end_budget = self.generations_until_end_budget,

                                            population_scaling = self.population_scaling,
                                            generations_until_end_population = self.generations_until_end_population,
                                            stepwise_steps = self.stepwise_steps,
                                            client = _client,
                                            objective_kwargs = {"X": X_future, "y": y_future},
                                            survival_selector=self.survival_selector,
                                            parent_selector=self.parent_selector,
                                            survival_percentage = self.survival_percentage,
                                            crossover_probability = self.crossover_probability,
                                            mutate_probability = self.mutate_probability,
                                            mutate_then_crossover_probability= self.mutate_then_crossover_probability,
                                            crossover_then_mutate_probability= self.crossover_then_mutate_probability,
                                            
                                            )

        
        self._evolver_instance.optimize()
        #self._evolver_instance.population.update_pareto_fronts(self.objective_names, self.objective_function_weights)
        self.make_evaluated_individuals()


        if self.optuna_optimize_pareto_front:
            pareto_front_inds = self.pareto_front['Individual'].values
            all_graphs, all_scores = tpot2.individual_representations.graph_pipeline_individual.simple_parallel_optuna(pareto_front_inds,  objective_function, self.objective_function_weights, _client, storage=self.optuna_storage, steps=self.optuna_optimize_pareto_front_trials, verbose=self.verbose, max_eval_time_seconds=self.max_eval_time_seconds, max_time_seconds=self.optuna_optimize_pareto_front_timeout, **{"X": X, "y": y})
            all_scores = tpot2.utils.eval_utils.process_scores(all_scores, len(self.objective_function_weights))
            
            if len(all_graphs) > 0:
                df = pd.DataFrame(np.column_stack((all_graphs, all_scores,np.repeat("Optuna",len(all_graphs)))), columns=["Individual"] + self.objective_names +["Parents"])
                for obj in self.objective_names:
                    df[obj] = df[obj].apply(convert_to_float)
                
                self.evaluated_individuals = pd.concat([self.evaluated_individuals, df], ignore_index=True)
            else:
                print("WARNING NO OPTUNA TRIALS COMPLETED")
        
        tpot2.utils.get_pareto_frontier(self.evaluated_individuals, column_names=self.objective_names, weights=self.objective_function_weights, invalid_values=["TIMEOUT","INVALID"])

        if validation_strategy == 'reshuffled':
            best_pareto_front_idx = list(self.pareto_front.index)
            best_pareto_front = list(self.pareto_front.loc[best_pareto_front_idx]['Individual'])
            
            #reshuffle rows
            X, y = sklearn.utils.shuffle(X, y, random_state=1)

            if self.scatter:
                X_future = _client.scatter(X)
                y_future = _client.scatter(y)
            else:
                X_future = X
                y_future = y

            val_objective_function_list = [lambda   ind, 
                                                    X, 
                                                    y, 
                                                    is_classification=self.classification,
                                                    scorers= self._scorers, 
                                                    cv=self.cv_gen, 
                                                    other_objective_functions=self.other_objective_functions, 
                                                    memory=self.memory, 
                                                    cross_val_predict_cv=self.cross_val_predict_cv, 
                                                    subset_column=self.subset_column, 
                                                    **kwargs: objective_function_generator(
                                                                                                ind,
                                                                                                X,
                                                                                                y, 
                                                                                                is_classification=is_classification,
                                                                                                scorers= scorers, 
                                                                                                cv=cv, 
                                                                                                other_objective_functions=other_objective_functions,
                                                                                                memory=memory, 
                                                                                                cross_val_predict_cv=cross_val_predict_cv, 
                                                                                                subset_column=subset_column,
                                                                                                **kwargs,
                                                                                                )]
            
            objective_kwargs = {"X": X_future, "y": y_future}
            val_scores = tpot2.utils.eval_utils.parallel_eval_objective_list(
                best_pareto_front,
                val_objective_function_list, n_jobs=self.n_jobs, verbose=self.verbose, timeout=self.max_eval_time_seconds,n_expected_columns=len(self.objective_names), client=_client, **objective_kwargs)

            val_objective_names = ['validation_'+name for name in self.objective_names]
            self.objective_names_for_selection = val_objective_names
            self.evaluated_individuals.loc[best_pareto_front_idx,val_objective_names] = val_scores

            self.evaluated_individuals["Validation_Pareto_Front"] = tpot2.utils.get_pareto_front(self.evaluated_individuals, val_objective_names, self.objective_function_weights, invalid_values=["TIMEOUT","INVALID"])

        elif validation_strategy == 'split':


            if self.scatter:            
                X_future = _client.scatter(X)
                y_future = _client.scatter(y)
                X_val_future = _client.scatter(X_val)
                y_val_future = _client.scatter(y_val)
            else:
                X_future = X
                y_future = y
                X_val_future = X_val
                y_val_future = y_val

            objective_kwargs = {"X": X_future, "y": y_future, "X_val" : X_val_future, "y_val":y_val_future }
            
            best_pareto_front_idx = list(self.pareto_front.index)
            best_pareto_front = list(self.pareto_front.loc[best_pareto_front_idx]['Individual'])
            val_objective_function_list = [lambda   ind, 
                                                    X, 
                                                    y, 
                                                    X_val, 
                                                    y_val, 
                                                    scorers= self._scorers, 
                                                    other_objective_functions=self.other_objective_functions, 
                                                    memory=self.memory, 
                                                    cross_val_predict_cv=self.cross_val_predict_cv, 
                                                    subset_column=self.subset_column, 
                                                    **kwargs: val_objective_function_generator(
                                                        ind,
                                                        X,
                                                        y,
                                                        X_val, 
                                                        y_val, 
                                                        scorers= scorers, 
                                                        other_objective_functions=other_objective_functions,
                                                        memory=memory, 
                                                        cross_val_predict_cv=cross_val_predict_cv, 
                                                        subset_column=subset_column,
                                                        **kwargs,
                                                        )]
            
            val_scores = tpot2.utils.eval_utils.parallel_eval_objective_list(
                best_pareto_front,
                val_objective_function_list, n_jobs=self.n_jobs, verbose=self.verbose, timeout=self.max_eval_time_seconds,n_expected_columns=len(self.objective_names),client=_client, **objective_kwargs)

            val_objective_names = ['validation_'+name for name in self.objective_names]
            self.objective_names_for_selection = val_objective_names
            self.evaluated_individuals.loc[best_pareto_front_idx,val_objective_names] = val_scores
            self.evaluated_individuals["Validation_Pareto_Front"] = tpot2.utils.get_pareto_front(self.evaluated_individuals, val_objective_names, self.objective_function_weights, invalid_values=["TIMEOUT","INVALID"])
        else:
            self.objective_names_for_selection = self.objective_names

        val_scores = self.evaluated_individuals[~self.evaluated_individuals[self.objective_names_for_selection].isin(["TIMEOUT","INVALID"]).any(axis=1)][self.objective_names_for_selection].astype(float)                                     
        weighted_scores = val_scores*self.objective_function_weights
        
        if self.bigger_is_better:
            best_idx = weighted_scores[self.objective_names_for_selection[0]].idxmax()
        else:
            best_idx = weighted_scores[self.objective_names_for_selection[0]].idxmin()
        
        best_individual = self.evaluated_individuals.loc[best_idx]['Individual']
        self.selected_best_score =  self.evaluated_individuals.loc[best_idx]
        

        best_individual_pipeline = best_individual.export_pipeline(memory=self.memory, cross_val_predict_cv=self.cross_val_predict_cv, subset_column=self.subset_column)

        if self.preprocessing:
            self.fitted_pipeline_ = sklearn.pipeline.make_pipeline(sklearn.base.clone(self._preprocessing_pipeline), best_individual_pipeline )
        else:
            self.fitted_pipeline_ = best_individual_pipeline 
        
        self.fitted_pipeline_.fit(X_original,y_original) #TODO use y_original as well?


        if self.client is None: #no client was passed in
            #close cluster and client
            _client.close()
            cluster.close()

        return self
        
    def _estimator_has(attr):
        '''Check if we can delegate a method to the underlying estimator.
        First, we check the first fitted final estimator if available, otherwise we
        check the unfitted final estimator.
        '''
        return  lambda self: (self.fitted_pipeline_ is not None and
            hasattr(self.fitted_pipeline_, attr)
        )



    


    @available_if(_estimator_has('predict'))
    def predict(self, X, **predict_params):
        check_is_fitted(self)
        #X = check_array(X)

        preds = self.fitted_pipeline_.predict(X,**predict_params)
        if self.classification and self.label_encoder_:
            preds = self.label_encoder_.inverse_transform(preds)

        return preds
    
    @available_if(_estimator_has('predict_proba'))
    def predict_proba(self, X, **predict_params):
        check_is_fitted(self)
        #X = check_array(X)
        return self.fitted_pipeline_.predict_proba(X,**predict_params)
    
    @available_if(_estimator_has('decision_function'))
    def decision_function(self, X, **predict_params):
        check_is_fitted(self)
        #X = check_array(X)
        return self.fitted_pipeline_.decision_function(X,**predict_params)
    
    @available_if(_estimator_has('transform'))
    def transform(self, X, **predict_params):
        check_is_fitted(self)
        #X = check_array(X)
        return self.fitted_pipeline_.transform(X,**predict_params)

    @property
    def classes_(self):
        """The classes labels. Only exist if the last step is a classifier."""
        if self.label_encoder_:
            return self.label_encoder_.classes_
        else:
            return self.fitted_pipeline_.classes_
    

    @property
    def _estimator_type(self):
        return self.fitted_pipeline_._estimator_type


    def make_evaluated_individuals(self):
        #check if _evolver_instance exists
        if self.evaluated_individuals is None:
            self.evaluated_individuals  =  self._evolver_instance.population.evaluated_individuals.copy()
            objects = list(self.evaluated_individuals.index)
            object_to_int = dict(zip(objects, range(len(objects))))
            self.evaluated_individuals = self.evaluated_individuals.set_index(self.evaluated_individuals.index.map(object_to_int))
            self.evaluated_individuals['Parents'] = self.evaluated_individuals['Parents'].apply(lambda row: convert_parents_tuples_to_integers(row, object_to_int))

            self.evaluated_individuals["Instance"] = self.evaluated_individuals["Individual"].apply(lambda ind: apply_make_pipeline(ind, preprocessing_pipeline=self._preprocessing_pipeline))

        return self.evaluated_individuals
        
    @property
    def pareto_front(self):
        #check if _evolver_instance exists
        if self.evaluated_individuals is None:
            return None
        else:
            if "Pareto_Front" not in self.evaluated_individuals:
                return self.evaluated_individuals
            else:
                return self.evaluated_individuals[self.evaluated_individuals["Pareto_Front"]==1]


