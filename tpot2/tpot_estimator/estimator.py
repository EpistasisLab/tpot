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
from sklearn.experimental import enable_iterative_imputer

from .default_search_spaces import get_default_search_space

def set_dask_settings():
    cfg.set({'distributed.scheduler.worker-ttl': None})
    cfg.set({'distributed.scheduler.allowed-failures':1})




#TODO inherit from _BaseComposition?
class TPOTEstimator(BaseEstimator):
    def __init__(self,  
                        search_space,
                        scorers,
                        scorers_weights,
                        classification,
                        cv = 5,
                        other_objective_functions=[],
                        other_objective_functions_weights = [],
                        objective_function_names = None,
                        bigger_is_better = True,

                        export_graphpipeline = False,
                        cross_val_predict_cv = 0,
                        memory = None,

                        categorical_features = None,
                        subsets = None,
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




                        #dask parameters
                        n_jobs=1,
                        memory_limit = None,
                        client = None,
                        processes = True,

                        #debugging and logging parameters
                        warm_start = False,
                        periodic_checkpoint_folder = None,
                        callback = None,

                        verbose = 0,
                        scatter = True,

                         # random seed for random number generator (rng)
                        random_state = None,

                        ):

        '''
        An sklearn baseestimator that uses genetic programming to optimize a pipeline.

        Parameters
        ----------
        search_space : (String, tpot2.search_spaces.SklearnIndividualGenerator)
            - String : The default search space to use for the optimization. This can be either "linear" or "graph". If "linear", will use the default linear pipeline search space. If "graph", will use the default graph pipeline search space.
            - SklearnIndividualGenerator : The search space to use for the optimization. This should be an instance of a SklearnIndividualGenerator.
                The search space to use for the optimization. This should be an instance of a SklearnIndividualGenerator.
                TPOT2 has groups of search spaces found in the following folders, tpot2.search_spaces.nodes for the nodes in the pipeline and tpot2.search_spaces.pipelines for the pipeline structure.
        
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
        
        cross_val_predict_cv : int, default=0
            Number of folds to use for the cross_val_predict function for inner classifiers and regressors. Estimators will still be fit on the full dataset, but the following node will get the outputs from cross_val_predict.

            - 0-1 : When set to 0 or 1, the cross_val_predict function will not be used. The next layer will get the outputs from fitting and transforming the full dataset.
            - >=2 : When fitting pipelines with inner classifiers or regressors, they will still be fit on the full dataset.
                    However, the output to the next node will come from cross_val_predict with the specified number of folds.

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

        preprocessing : bool or BaseEstimator/Pipeline,
            EXPERIMENTAL
            A pipeline that will be used to preprocess the data before CV. Note that the parameters for these steps are not optimized. Add them to the search space to be optimized.
            - bool : If True, will use a default preprocessing pipeline which includes imputation followed by one hot encoding.
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

        scatter : bool, default=True
            If True, will scatter the data to the dask workers. If False, will not scatter the data. This can be useful for debugging.

        random_state : int, None, default=None
            A seed for reproducability of experiments. This value will be passed to numpy.random.default_rng() to create an instnce of the genrator to pass to other classes

            - int
                Will be used to create and lock in Generator instance with 'numpy.random.default_rng()'
            - None
                Will be used to create Generator for 'numpy.random.default_rng()' where a fresh, unpredictable entropy will be pulled from the OS

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

        self.search_space = search_space

        self.export_graphpipeline = export_graphpipeline
        self.cross_val_predict_cv = cross_val_predict_cv
        self.memory = memory

        if self.cross_val_predict_cv !=0 or self.memory is not None:
            if not self.export_graphpipeline:
                raise ValueError("cross_val_predict_cv and memory parameters are parameters for GraphPipeline. To enable these options export_graphpipeline to be True. Otherwise these can be passed into the relevant Search spaces as parameters.")

        self.categorical_features = categorical_features
        self.subsets = subsets
        
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
        self.verbose = verbose
        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.callback = callback
        self.processes = processes


        self.scatter = scatter



        # create random number generator based on rngseed
        self.rng = np.random.default_rng(random_state)
        # save random state passed to us for other functions that use random_state
        self.random_state = random_state
        # set the numpy seed so anything using it will be consistent as well
        np.random.seed(random_state)

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
                X, X_val, y, y_val = train_test_split(X, y, test_size=self.validation_fraction, stratify=y, random_state=self.random_state)
            else:
                X, X_val, y, y_val = train_test_split(X, y, test_size=self.validation_fraction, random_state=self.random_state)


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

            if not isinstance(self.preprocessing, bool) and isinstance(self.preprocessing, sklearn.base.BaseEstimator):
                self._preprocessing_pipeline = sklearn.base.clone(self.preprocessing)

            #TODO: check if there are missing values in X before imputation. If not, don't include imputation in pipeline. Check if there are categorical columns. If not, don't include one hot encoding in pipeline
            else: #if self.preprocessing is True or not a sklearn estimator
                
                pipeline_steps = []

                if self.categorical_features is not None: #if categorical features are specified, use those
                    pipeline_steps.append(("impute_categorical", tpot2.builtin_modules.ColumnSimpleImputer(self.categorical_features, strategy='most_frequent')))
                    pipeline_steps.append(("impute_numeric", tpot2.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean')))
                    pipeline_steps.append(("ColumnOneHotEncoder", tpot2.builtin_modules.ColumnOneHotEncoder(self.categorical_features, strategy='most_frequent')))

                else:
                    if isinstance(X, pd.DataFrame):
                        categorical_columns = X.select_dtypes(include=['object']).columns
                        if len(categorical_columns) > 0:
                            pipeline_steps.append(("impute_categorical", tpot2.builtin_modules.ColumnSimpleImputer("categorical", strategy='most_frequent')))
                            pipeline_steps.append(("impute_numeric", tpot2.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean')))
                            pipeline_steps.append(("ColumnOneHotEncoder", tpot2.builtin_modules.ColumnOneHotEncoder("categorical", strategy='most_frequent')))
                        else:
                            pipeline_steps.append(("impute_numeric", tpot2.builtin_modules.ColumnSimpleImputer("all", strategy='mean')))
                    else:
                        pipeline_steps.append(("impute_numeric", tpot2.builtin_modules.ColumnSimpleImputer("all", strategy='mean')))
                            
                self._preprocessing_pipeline = sklearn.pipeline.Pipeline(pipeline_steps)

            X = self._preprocessing_pipeline.fit_transform(X, y)
            
        else:
            self._preprocessing_pipeline = None

        #_, y = sklearn.utils.check_X_y(X, y, y_numeric=True)

        #Set up the configuation dictionaries and the search spaces

        #check if self.cv is a number
        if isinstance(self.cv, int) or isinstance(self.cv, float):
            if self.classification:
                self.cv_gen = sklearn.model_selection.StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            else:
                self.cv_gen = sklearn.model_selection.KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        else:
            self.cv_gen = sklearn.model_selection.check_cv(self.cv, y, classifier=self.classification)



        n_samples= int(math.floor(X.shape[0]/n_folds))
        n_features=X.shape[1]

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
        else:
            self.feature_names = None



        def objective_function(pipeline_individual,
                                            X,
                                            y,
                                            is_classification=self.classification,
                                            scorers= self._scorers,
                                            cv=self.cv_gen,
                                            other_objective_functions=self.other_objective_functions,
                                            export_graphpipeline=self.export_graphpipeline,
                                            memory=self.memory,
                                            cross_val_predict_cv=self.cross_val_predict_cv,
                                            **kwargs):
            return objective_function_generator(
                pipeline_individual,
                X,
                y,
                is_classification=is_classification,
                scorers= scorers,
                cv=cv,
                other_objective_functions=other_objective_functions,
                export_graphpipeline=export_graphpipeline,
                memory=memory,
                cross_val_predict_cv=cross_val_predict_cv,
                **kwargs,
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

        if self.classification:
            n_classes = len(np.unique(y))
        else:
            n_classes = None

        get_search_space_params = {"n_classes": n_classes, 
                        "n_samples":len(y), 
                        "n_features":X.shape[1], 
                        "random_state":self.random_state}

        self._search_space = get_default_search_space(self.search_space, classification=True, inner_predictors=True, **get_search_space_params)


        # TODO : Add check for empty values in X and if so, add imputation to the search space
        # make this depend on self.preprocessing
        # if check_empty_values(X):
        #     from sklearn.experimental import enable_iterative_imputer

        #     from ConfigSpace import ConfigurationSpace
        #     from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
        #     iterative_imputer_cs = ConfigurationSpace(
        #         space = {
        #             'n_nearest_features' : Categorical('n_nearest_features', [100]),
        #             'initial_strategy' : Categorical('initial_strategy', ['mean','median', 'most_frequent', ]),
        #             'add_indicator' : Categorical('add_indicator', [True, False]),
        #         }
        #     )

        #     imputation_search = tpot2.search_spaces.pipelines.ChoicePipeline([
        #         tpot2.config.get_search_space("SimpleImputer"),
        #         tpot2.search_spaces.nodes.EstimatorNode(sklearn.impute.IterativeImputer, iterative_imputer_cs)
        #     ])




        #     self.search_space_final = tpot2.search_spaces.pipelines.SequentialPipeline(search_spaces=[ imputation_search, self._search_space], memory="sklearn_pipeline_memory")
        # else:
        #     self.search_space_final = self._search_space

        self.search_space_final = self._search_space

        def ind_generator(rng):
            rng = np.random.default_rng(rng)
            while True:
                yield self.search_space_final.generate(rng)

        #If warm start and we have an evolver instance, use the existing one
        if not(self.warm_start and self._evolver_instance is not None):
            self._evolver_instance = self._evolver(   individual_generator=ind_generator(self.rng),
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

                                            rng=self.rng,
                                            )


        self._evolver_instance.optimize()
        #self._evolver_instance.population.update_pareto_fronts(self.objective_names, self.objective_function_weights)
        self.make_evaluated_individuals()




        tpot2.utils.get_pareto_frontier(self.evaluated_individuals, column_names=self.objective_names, weights=self.objective_function_weights)

        if validation_strategy == 'reshuffled':
            best_pareto_front_idx = list(self.pareto_front.index)
            best_pareto_front = list(self.pareto_front.loc[best_pareto_front_idx]['Individual'])

            #reshuffle rows
            X, y = sklearn.utils.shuffle(X, y, random_state=self.random_state)

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
                                                    export_graphpipeline=self.export_graphpipeline,
                                                    memory=self.memory,
                                                    cross_val_predict_cv=self.cross_val_predict_cv,

                                                    **kwargs: objective_function_generator(
                                                                                                ind,
                                                                                                X,
                                                                                                y,
                                                                                                is_classification=is_classification,
                                                                                                scorers= scorers,
                                                                                                cv=cv,
                                                                                                other_objective_functions=other_objective_functions,
                                                                                                export_graphpipeline=export_graphpipeline,
                                                                                                memory=memory,
                                                                                                cross_val_predict_cv=cross_val_predict_cv,
                                                                                                **kwargs,
                                                                                                )]

            objective_kwargs = {"X": X_future, "y": y_future}
            # val_scores = tpot2.utils.eval_utils.parallel_eval_objective_list(
            #     best_pareto_front,
            #     val_objective_function_list, n_jobs=self.n_jobs, verbose=self.verbose, timeout=self.max_eval_time_seconds,n_expected_columns=len(self.objective_names), client=_client, **objective_kwargs)
            val_scores, start_times, end_times, eval_errors = tpot2.utils.eval_utils.parallel_eval_objective_list2(best_pareto_front, val_objective_function_list, verbose=self.verbose, max_eval_time_seconds=self.max_eval_time_seconds, n_expected_columns=len(self.objective_names), client=_client, **objective_kwargs)



            val_objective_names = ['validation_'+name for name in self.objective_names]
            self.objective_names_for_selection = val_objective_names
            self.evaluated_individuals.loc[best_pareto_front_idx,val_objective_names] = val_scores
            self.evaluated_individuals.loc[best_pareto_front_idx,'validation_start_times'] = start_times
            self.evaluated_individuals.loc[best_pareto_front_idx,'validation_end_times'] = end_times
            self.evaluated_individuals.loc[best_pareto_front_idx,'validation_eval_errors'] = eval_errors

            self.evaluated_individuals["Validation_Pareto_Front"] = tpot2.utils.get_pareto_frontier(self.evaluated_individuals, column_names=val_objective_names, weights=self.objective_function_weights)


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
                                                    export_graphpipeline=self.export_graphpipeline,
                                                    memory=self.memory,
                                                    cross_val_predict_cv=self.cross_val_predict_cv,
                                                    **kwargs: val_objective_function_generator(
                                                        ind,
                                                        X,
                                                        y,
                                                        X_val,
                                                        y_val,
                                                        scorers= scorers,
                                                        other_objective_functions=other_objective_functions,
                                                        export_graphpipeline=export_graphpipeline,
                                                        memory=memory,
                                                        cross_val_predict_cv=cross_val_predict_cv,
                                                        **kwargs,
                                                        )]

            val_scores, start_times, end_times, eval_errors = tpot2.utils.eval_utils.parallel_eval_objective_list2(best_pareto_front, val_objective_function_list, verbose=self.verbose, max_eval_time_seconds=self.max_eval_time_seconds, n_expected_columns=len(self.objective_names), client=_client, **objective_kwargs)



            val_objective_names = ['validation_'+name for name in self.objective_names]
            self.objective_names_for_selection = val_objective_names
            self.evaluated_individuals.loc[best_pareto_front_idx,val_objective_names] = val_scores
            self.evaluated_individuals.loc[best_pareto_front_idx,'validation_start_times'] = start_times
            self.evaluated_individuals.loc[best_pareto_front_idx,'validation_end_times'] = end_times
            self.evaluated_individuals.loc[best_pareto_front_idx,'validation_eval_errors'] = eval_errors

            self.evaluated_individuals["Validation_Pareto_Front"] = tpot2.utils.get_pareto_frontier(self.evaluated_individuals, column_names=val_objective_names, weights=self.objective_function_weights)
        
        else:
            self.objective_names_for_selection = self.objective_names
        
        val_scores = self.evaluated_individuals[~self.evaluated_individuals[self.objective_names_for_selection].isna().all(1)][self.objective_names_for_selection]
        weighted_scores = val_scores*self.objective_function_weights

        if self.bigger_is_better:
            best_idx = weighted_scores[self.objective_names_for_selection[0]].idxmax()
        else:
            best_idx = weighted_scores[self.objective_names_for_selection[0]].idxmin()

        best_individual = self.evaluated_individuals.loc[best_idx]['Individual']
        self.selected_best_score =  self.evaluated_individuals.loc[best_idx]


        #TODO
        #best_individual_pipeline = best_individual.export_pipeline(memory=self.memory, cross_val_predict_cv=self.cross_val_predict_cv)
        if self.export_graphpipeline:
            best_individual_pipeline = best_individual.export_flattened_graphpipeline(memory=self.memory, cross_val_predict_cv=self.cross_val_predict_cv)
        else:
            best_individual_pipeline = best_individual.export_pipeline()

        if self.preprocessing:
            self.fitted_pipeline_ = sklearn.pipeline.make_pipeline(sklearn.base.clone(self._preprocessing_pipeline), best_individual_pipeline )
        else:
            self.fitted_pipeline_ = best_individual_pipeline

        self.fitted_pipeline_.fit(X_original,y_original) #TODO use y_original as well?


        if self.client is None: #no client was passed in
            #close cluster and client
            # _client.close()
            # cluster.close()
            try:
                _client.shutdown()
                cluster.close()
            #catch exception
            except Exception as e:
                print("Error shutting down client and cluster")
                Warning(e)

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

            self.evaluated_individuals["Instance"] = self.evaluated_individuals["Individual"].apply(lambda ind: apply_make_pipeline(ind, preprocessing_pipeline=self._preprocessing_pipeline, export_graphpipeline=self.export_graphpipeline, memory=self.memory, cross_val_predict_cv=self.cross_val_predict_cv))

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


def check_empty_values(data):
    """
    Checks for empty values in a dataset.

    Args:
        data (numpy.ndarray or pandas.DataFrame): The dataset to check.

    Returns:
        bool: True if the dataset contains empty values, False otherwise.
    """
    if isinstance(data, pd.DataFrame):
        return data.isnull().values.any()
    elif isinstance(data, np.ndarray):
        return np.isnan(data).any()
    else:
        raise ValueError("Unsupported data type")