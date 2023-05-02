from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import available_if
import numpy as np
import typing
import sklearn.metrics
import tpot2.estimator_objective_functions
from functools import partial
import tpot2.config
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tpot2.parent_selectors import survival_select_NSGA2, TournamentSelection_Dominated
from sklearn.preprocessing import LabelEncoder 
from sklearn.utils.multiclass import unique_labels 
import pandas as pd
from sklearn.model_selection import train_test_split
import tpot2
import distributed
from dask.distributed import Client
from dask.distributed import LocalCluster
import math

EVOLVERS = {"nsga2":tpot2.BaseEvolver}



#TODO inherit from _BaseComposition?
class TPOTEstimator(BaseEstimator):
    def __init__(self,  scorers, 
                        scorers_weights,
                        classification,
                        cv = 5,
                        other_objective_functions=[tpot2.estimator_objective_functions.average_path_length_objective], #tpot2.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights = [-1],
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
                        validation_strategy = "none",
                        validation_fraction = .2,
                        population_size = 50,
                        initial_population_size = None,
                        population_scaling = .5, 
                        generations_until_end_population = 1,  
                        generations = 50,
                        early_stop = None,
                        scorers_early_stop_tol = 0.001,
                        other_objectives_early_stop_tol =None,
                        max_time_seconds=float('inf'), 
                        max_eval_time_seconds=60*10, 
                        n_jobs=1,
                        memory_limit = "4GB",
                        client = None,
                        survival_percentage = 1,
                        crossover_probability=.2,
                        mutate_probability=.7,
                        mutate_then_crossover_probability=.05,
                        crossover_then_mutate_probability=.05,
                        survival_selector = survival_select_NSGA2,
                        parent_selector = TournamentSelection_Dominated,
                        budget_range = None,
                        budget_scaling = .5,
                        generations_until_end_budget = 1,  
                        stepwise_steps = 5,
                        threshold_evaluation_early_stop = None, 
                        threshold_evaluation_scaling = .5,
                        min_history_threshold = 20,
                        selection_evaluation_early_stop = None, 
                        selection_evaluation_scaling = .5, 
                        n_initial_optimizations = 0,
                        optimization_cv = 3,
                        max_optimize_time_seconds=60*20,
                        optimization_steps = 10,
                        warm_start = False,
                        subset_column = None,
                        evolver = "nsga2",
                        verbose = 0,
                        periodic_checkpoint_folder = None, 
                        callback: tpot2.CallBackInterface = None,
                        processes = True,
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
        
        other_objective_functions : list, default=[tpot2.estimator_objective_functions.average_path_length_objective]
            A list of other objective functions to apply to the pipeline.
        
        other_objective_functions_weights : list, default=[-1]
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
              
        validation_strategy : str, default='none'
            EXPERIMENTAL The validation strategy to use for selecting the final pipeline from the population. TPOT2 may overfit the cross validation score. A second validation set can be used to select the final pipeline.
            - 'auto' : Automatically determine the validation strategy based on the dataset shape.
            - 'reshuffled' : Use the same data for cross validation and final validation, but with different splits for the folds. This is the default for small datasets. 
            - 'split' : Use a separate validation set for final validation. Data will be split according to validation_fraction. This is the default for medium datasets. 
            - 'none' : Do not use a separate validation set for final validation. Select based on the original cross-validation score. This is the default for large datasets.

        validation_fraction : float, default=0.2
          EXPERIMENTAL The fraction of the dataset to use for the validation set when validation_strategy is 'split'. Must be between 0 and 1.
        
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
    
        max_time_seconds : float, default=float("inf")
            Maximum time to run the optimization. If none or inf, will run until the end of the generations.
        
        max_eval_time_seconds : float, default=60*5
            Maximum time to evaluate a single individual. If none or inf, there will be no time limit per evaluation.
        
        n_jobs : int, default=1
            Number of processes to run in parallel.
        
        memory_limit : str, default="4GB"
            Memory limit for each job. See Dask [LocalCluster documentation](https://distributed.dask.org/en/stable/api.html#distributed.Client) for more information.
        
        client : dask.distributed.Client, default=None
            A dask client to use for parallelization. If not None, this will override the n_jobs and memory_limit parameters. If None, will create a new client with num_workers=n_jobs and memory_limit=memory_limit. 
        
        survival_percentage : float, default=1
            Percentage of the population size to utilize for mutation and crossover at the beginning of the generation. The rest are discarded. Individuals are selected with the selector passed into survival_selector. The value of this parameter must be between 0 and 1, inclusive. 
            For example, if the population size is 100 and the survival percentage is .5, 50 individuals will be selected with NSGA2 from the existing population. These will be used for mutation and crossover to generate the next 100 individuals for the next generation. The remainder are discarded from the live population. In the next generation, there will now be the 50 parents + the 100 individuals for a total of 150. Surivival percentage is based of the population size parameter and not the existing population size. Therefore, in the next generation we will still select 50 individuals from the currently existing 150.
        
        crossover_probability : float, default=.2
            Probability of generating a new individual by crossover between two individuals.
        
        mutate_probability : float, default=.7
            Probability of generating a new individual by crossover between one individuals.
        
        mutate_then_crossover_probability : float, default=.05
            Probability of generating a new individual by mutating two individuals followed by crossover.
        
        crossover_then_mutate_probability : float, default=.05
            Probability of generating a new individual by crossover between two individuals followed by a mutation of the resulting individual.
        
        n_parents : int, default=2
            Number of parents to use for crossover. Must be greater than 1.
        
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
        
        threshold_evaluation_early_stop : list [start, end], default=None
            starting and ending percentile to use as a threshold for the evaluation early stopping.
            Values between 0 and 100.
        
        threshold_evaluation_scaling : float [0,inf), default=0.5
            A scaling factor to use when determining how fast we move the threshold moves from the start to end percentile.
            Must be greater than zero. Higher numbers will move the threshold to the end faster.
        
        min_history_threshold : int, default=0
            The minimum number of previous scores needed before using threshold early stopping.
        
        selection_evaluation_early_stop : list, default=None
            A lower and upper percent of the population size to select each round of CV.
            Values between 0 and 1.
        
        selection_evaluation_scaling : float, default=0.5 
            A scaling factor to use when determining how fast we move the threshold moves from the start to end percentile.
            Must be greater than zero. Higher numbers will move the threshold to the end faster.
        
        n_initial_optimizations : int, default=0
            Number of individuals to optimize before starting the evolution.
        
        optimization_cv : int 
           Number of folds to use for the optuna optimization's internal cross-validation.
        
        max_optimize_time_seconds : float, default=60*5
            Maximum time to run an optimization
        
        optimization_steps : int, default=10
            Number of steps per optimization
          
        warm_start : bool, default=False
            If True, will use the continue the evolutionary algorithm from the last generation of the previous run.
         
        subset_column : str or int, default=None
            EXPERIMENTAL The column to use for the subset selection. Must also pass in unique_subset_values to GraphIndividual to function.
         
        evolver : tpot2.evolutionary_algorithms.eaNSGA2.eaNSGA2_Evolver), default=eaNSGA2_Evolver
            The evolver to use for the optimization process. See tpot2.evolutionary_algorithms
            - type : an type or subclass of a BaseEvolver
            - "nsga2" : tpot2.evolutionary_algorithms.eaNSGA2.eaNSGA2_Evolver
        
        verbose : int, default=1 
            How much information to print during the optimization process. Higher values include the information from lower values.
            0. nothing
            1. progress bar
            2. evaluations progress bar
            3. best individual
            4. warnings
            >=5. full warnings trace
        
        periodic_checkpoint_folder : str, default=None
            Folder to save the population to periodically. If None, no periodic saving will be done.
            If provided, training will resume from this checkpoint.
        
        callback : tpot2.CallBackInterface, default=None
            Callback object. Not implemented

        processes : bool, default=True
            If True, will use multiprocessing to parallelize the optimization process. If False, will use threading.
            True seems to perform better. However, False is required for interactive debugging.
            
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
        self.n_initial_optimizations  = n_initial_optimizations  
        self.optimization_cv  = optimization_cv
        self.max_optimize_time_seconds = max_optimize_time_seconds 
        self.optimization_steps = optimization_steps 
        self.warm_start = warm_start
        self.subset_column = subset_column
        self.evolver = evolver
        self.verbose = verbose
        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.callback = callback
        self.processes = processes

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
        
        if isinstance(self.evolver, str):
            self._evolver = EVOLVERS[self.evolver]
        else:
            self._evolver = self.evolver
        
       

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

        if isinstance(self.cv, int) or isinstance(self.cv, float):
            n_folds = self.cv
        else:
            n_folds = self.cv.n_splits

        n_samples= int(math.floor(X.shape[0]/n_folds))
        n_features=X.shape[1]

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
        else:
            self.feature_names = None

        if self.root_config_dict == 'Auto':
            if self.classification:
                root_config_dict = get_configuration_dictionary("classifiers", n_samples, n_features, self.classification, subsets=self.subsets, feature_names=self.feature_names)
            else:
                root_config_dict = get_configuration_dictionary("regressors", n_samples, n_features, self.classification,subsets=self.subsets, feature_names=self.feature_names)
        else:
            root_config_dict = get_configuration_dictionary(self.root_config_dict, n_samples, n_features, self.classification, subsets=self.subsets,feature_names=self.feature_names)

        inner_config_dict = get_configuration_dictionary(self.inner_config_dict, n_samples, n_features, self.classification,subsets=self.subsets, feature_names=self.feature_names)
        leaf_config_dict = get_configuration_dictionary(self.leaf_config_dict, n_samples, n_features, self.classification, subsets=self.subsets, feature_names=self.feature_names)

        if self.n_initial_optimizations > 0:
            #tmp = partial(tpot2.estimator_objective_functions.cross_val_score_objective,scorers= self._scorers, cv=self.optimization_cv, memory=self.memory, cross_val_predict_cv=self.cross_val_predict_cv, subset_column=self.subset_column )
            # optuna_objective = lambda ind,  X=X, y=y , scorers= self._scorers, cv=self.optimization_cv, memory=self.memory, cross_val_predict_cv=self.cross_val_predict_cv, subset_column=self.subset_column: tpot2.estimator_objective_functions.cross_val_score_objective(
            #     ind, 
            #     X=X, y=y, scorers= scorers, cv=cv, memory=memory, cross_val_predict_cv=cross_val_predict_cv, subset_column=subset_column )
            
            optuna_objective = partial(tpot2.estimator_objective_functions.cross_val_score_objective, X=X, y=y , scorers= self._scorers, cv=self.optimization_cv, memory=self.memory, cross_val_predict_cv=self.cross_val_predict_cv, subset_column=self.subset_column )
        else:
            optuna_objective = None


        #check if self.cv is a number
        if isinstance(self.cv, int) or isinstance(self.cv, float):
            if self.classification:
                self.cv_gen = sklearn.model_selection.StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
            else:
                self.cv_gen = sklearn.model_selection.KFold(n_splits=self.cv, shuffle=True, random_state=42)

        else:
            self.cv_gen = sklearn.model_selection.check_cv(self.cv, y, classifier=self.classification)


        self.individual_generator_instance = tpot2.estimator_graph_individual_generator(   
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


        X_future = _client.scatter(X)
        y_future = _client.scatter(y)

        #.export_pipeline(memory=self.memory, cross_val_predict_cv=self.cross_val_predict_cv, subset_column=self.subset_column),
        #tmp = partial(objective_function_generator, scorers= self._scorers, cv=self.cv_gen, other_objective_functions=self.other_objective_functions )
        self.final_object_function_list =[ lambda pipeline_individual, X, y,is_classification=self.classification,
                scorers= self._scorers, cv=self.cv_gen, other_objective_functions=self.other_objective_functions,
                 memory=self.memory, cross_val_predict_cv=self.cross_val_predict_cv, subset_column=self.subset_column, **kwargs: objective_function_generator(
                                pipeline_individual,
                                #ind,
                                X, y, 
                                is_classification=is_classification,
                                scorers= scorers, cv=cv, other_objective_functions=other_objective_functions,
                                memory=memory, cross_val_predict_cv=cross_val_predict_cv, subset_column=subset_column,
                                **kwargs,
                                )]


        #If warm start and we have an evolver instance, use the existing one
        if not(self.warm_start and self._evolver_instance is not None):
            self._evolver_instance = self._evolver(   individual_generator=self.individual_generator_instance, 
                                            objective_functions=self.final_object_function_list,
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
                                            optimization_objective=optuna_objective,
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
        #self.make_evaluated_individuals()


        self.evaluated_individuals = self._evolver_instance.population.evaluated_individuals.copy()
        self.evaluated_individuals = get_pareto_front(self.evaluated_individuals, self.objective_names, self.objective_function_weights)

        if validation_strategy == 'reshuffled':
            best_pareto_front_idx = list(self.pareto_front.index)
            best_pareto_front = self.pareto_front.loc[best_pareto_front_idx]['Instance']
            
            #reshuffle rows
            X, y = sklearn.utils.shuffle(X, y, random_state=1)
            X_future = _client.scatter(X)
            y_future = _client.scatter(y)

            val_objective_function_list = [lambda ind, X, y, is_classification=self.classification,scorers= self._scorers, cv=self.cv_gen, other_objective_functions=self.other_objective_functions, **kwargs: objective_function_generator(
                                                                                                ind,
                                                                                                X,y, 
                                                                                                is_classification=is_classification,
                                                                                                scorers= scorers, cv=cv, other_objective_functions=other_objective_functions,
                                                                                                **kwargs,
                                                                                                )]
            
            val_scores = tpot2.objectives.parallel_eval_objective_list(
                best_pareto_front,
                val_objective_function_list, n_jobs=self.n_jobs, verbose=self.verbose, timeout=self.max_eval_time_seconds,n_expected_columns=len(self.objective_names), client=_client, X= X_future, y= y_future)

            val_objective_names = ['validation_'+name for name in self.objective_names]
            self.objective_names_for_selection = val_objective_names
            self.evaluated_individuals.loc[best_pareto_front_idx,val_objective_names] = val_scores

        elif validation_strategy == 'split':

            
            X_future = _client.scatter(X)
            y_future = _client.scatter(y)
            X_val_future = _client.scatter(X_val)
            y_val_future = _client.scatter(y_val)


            best_pareto_front_idx = list(self.pareto_front.index)
            best_pareto_front = self.pareto_front.loc[best_pareto_front_idx]['Instance']
            val_objective_function_list = [lambda ind, X, y, X_val, y_val, scorers= self._scorers, other_objective_functions=self.other_objective_functions, **kwargs: val_objective_function_generator(
                ind,
                X,y,
                X_val, y_val, 
                scorers= scorers, other_objective_functions=other_objective_functions,
                **kwargs,
                )]
            
            val_scores = tpot2.objectives.parallel_eval_objective_list(
                best_pareto_front,
                val_objective_function_list, n_jobs=self.n_jobs, verbose=self.verbose, timeout=self.max_eval_time_seconds,n_expected_columns=len(self.objective_names),client=_client, X=X_future, y=y_future, X_val=X_val_future, y_val=y_val_future)

            val_objective_names = ['validation_'+name for name in self.objective_names]
            self.objective_names_for_selection = val_objective_names
            self.evaluated_individuals.loc[best_pareto_front_idx,val_objective_names] = val_scores
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
        
        self.fitted_pipeline_.fit(X_original,y) #TODO use y_original as well?
        if self.verbose >= 3:
            best_individual.plot()

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
        return self.fitted_pipeline_.predict(X,**predict_params)
    
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
        return self.fitted_pipeline_.classes_


    def make_evaluated_individuals(self):
        #check if _evolver_instance exists
        if self.evaluated_individuals is None:
            self.evaluated_individuals  =  self._evolver_instance.population.evaluated_individuals.copy()
            objects = list(self.evaluated_individuals.index)
            object_to_int = dict(zip(objects, range(len(objects))))
            self.evaluated_individuals = self.evaluated_individuals.set_index(self.evaluated_individuals.index.map(object_to_int))
            self.evaluated_individuals['Parents'] = self.evaluated_individuals['Parents'].apply(lambda row: _convert_parents_tuples_to_integers(row, object_to_int))

            self.evaluated_individuals["Instance"] = self.evaluated_individuals["Individual"].apply(lambda ind: _apply_make_pipeline(ind, preprocessing_pipeline=self._preprocessing_pipeline))

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
                return self.evaluated_individuals[self.evaluated_individuals["Pareto_Front"]==0]


def get_pareto_front(df, column_names, weights, invalid_values=["TIMEOUT","INVALID"], inplace=True, top=None):
    dftmp = df[~df[column_names].isin(invalid_values).any(axis=1)]

    if "Budget" in dftmp.columns:
        #get rows with the max budget
        dftmp = dftmp[dftmp["Budget"]==dftmp["Budget"].max()]


    indeces = dftmp[~dftmp[column_names].isna().any(axis=1)].index.values
    weighted_scores = df.loc[indeces][column_names].to_numpy()  * weights

    pareto_fronts = tpot2.parent_selectors.nondominated_sorting(weighted_scores)

    if not inplace:
        df = pd.DataFrame(index=df.index,columns=["Pareto_Front"], data=[])
    
    df["Pareto_Front"] = np.nan

    for i, front in enumerate(pareto_fronts):
        for index in front:
            df.loc[indeces[index], "Pareto_Front"] = i

    return df


def _convert_parents_tuples_to_integers(row, object_to_int):
    if type(row) == list or type(row) == np.ndarray or type(row) == tuple:
        return tuple(object_to_int[obj] for obj in row)
    else:
        return np.nan

def _apply_make_pipeline(graphindividual, preprocessing_pipeline=None):
    try: 
        if preprocessing_pipeline is None:
            return graphindividual.export_pipeline()
        else:
            return sklearn.pipeline.make_pipeline(sklearn.base.clone(preprocessing_pipeline), graphindividual.export_pipeline())
    except:
        return None

def get_configuration_dictionary(options, n_samples, n_features, classification, subsets=None, feature_names=None):
    if options is None:
        return options

    if isinstance(options, dict):
        return recursive_with_defaults(options, n_samples, n_features, classification, subsets=subsets, feature_names=feature_names)
    
    if not isinstance(options, list):
        options = [options]

    config_dict = {}

    for option in options:

        if option == "selectors":
            config_dict.update(tpot2.config.make_selector_config_dictionary(classification))

        elif option == "classifiers":
            config_dict.update(tpot2.config.make_classifier_config_dictionary(n_samples=n_samples))

        elif option == "regressors":
            config_dict.update(tpot2.config.make_regressor_config_dictionary(n_samples=n_samples))

        elif option == "transformers":
            config_dict.update(tpot2.config.make_transformer_config_dictionary(n_features=n_features))
        
        elif option == "arithmetic_transformer":
            config_dict.update(tpot2.config.make_arithmetic_transformer_config_dictionary())

        elif option == "feature_set_selector":
            config_dict.update(tpot2.config.make_FSS_config_dictionary(subsets, n_features, feature_names=feature_names))

        elif option == "skrebate":
            config_dict.update(tpot2.config.make_skrebate_config_dictionary(n_features=n_features))
        
        elif option == "MDR":
            config_dict.update(tpot2.config.make_MDR_config_dictionary())
        
        elif option == "ContinuousMDR":
            config_dict.update(tpot2.config.make_ContinuousMDR_config_dictionary())

        elif option == "FeatureEncodingFrequencySelector":
            config_dict.update(tpot2.config.make_FeatureEncodingFrequencySelector_config_dictionary())

        elif option == "genetic encoders":
            config_dict.update(tpot2.config.make_genetic_encoders_config_dictionary())

        elif option == "passthrough":
            config_dict.update(tpot2.config.make_passthrough_config_dictionary())
        

        else:
            config_dict.update(recursive_with_defaults(option, n_samples, n_features, classification, subsets=subsets, feature_names=feature_names))

    if len(config_dict) == 0:
        raise ValueError("No valid configuration options were provided. Please check the options you provided and try again.")

    return config_dict

def recursive_with_defaults(config_dict, n_samples, n_features, classification, subsets=None, feature_names=None):
    
    for key in 'leaf_config_dict', 'root_config_dict', 'inner_config_dict', 'Recursive':
        if key in config_dict:
            value = config_dict[key]
            if key=="Resursive":
                config_dict[key] = recursive_with_defaults(value,n_samples, n_features, classification, subsets=None, feature_names=None)
            else:
                config_dict[key] = get_configuration_dictionary(value, n_samples, n_features, classification, subsets, feature_names)
        
    return config_dict



def objective_function_generator(pipeline, x,y, scorers, cv, other_objective_functions, memory=None, cross_val_predict_cv=None, subset_column=None, step=None, budget=None, generation=1,is_classification=True):
    pipeline = pipeline.export_pipeline(memory=memory, cross_val_predict_cv=cross_val_predict_cv, subset_column=subset_column)
    if budget is not None and budget < 1:
        if is_classification:
            x,y = sklearn.utils.resample(x,y, stratify=y, n_samples=int(budget*len(x)), replace=False, random_state=1)
        else:
            x,y = sklearn.utils.resample(x,y, n_samples=int(budget*len(x)), replace=False, random_state=1)

    if len(scorers) > 0:
        cv_obj_scores = tpot2.estimator_objective_functions.cross_val_score_objective(sklearn.base.clone(pipeline),x,y,scorers=scorers, cv=cv , fold=step)
    else:
        cv_obj_scores = []
    
    if other_objective_functions is not None and len(other_objective_functions) >0:
        other_scores = [obj(sklearn.base.clone(pipeline)) for obj in other_objective_functions]
        #flatten
        other_scores = np.array(other_scores).flatten().tolist()
    else:
        other_scores = []
        
    return np.concatenate([cv_obj_scores,other_scores])


def val_objective_function_generator(pipeline, X_train, y_train, X_test, y_test, scorers, other_objective_functions, memory=None, cross_val_predict_cv=None, subset_column=None, ):
    #subsample the data
    pipeline = pipeline.export_pipeline(memory=memory, cross_val_predict_cv=cross_val_predict_cv, subset_column=subset_column)
    fitted_pipeline = sklearn.base.clone(pipeline)
    fitted_pipeline.fit(X_train, y_train)

    this_fold_scores = [sklearn.metrics.get_scorer(scorer)(fitted_pipeline, X_test, y_test) for scorer in scorers] 
    
    other_scores = []
    #TODO use same exported pipeline as for each objective
    if other_objective_functions is not None and len(other_objective_functions) >0:
        other_scores = [obj(sklearn.base.clone(pipeline)) for obj in other_objective_functions]
    
    return np.concatenate([this_fold_scores,other_scores])