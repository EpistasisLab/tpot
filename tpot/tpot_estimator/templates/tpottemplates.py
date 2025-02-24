"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Gabriel Ketron (gabriel.ketron@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
import tpot
import numpy as np
import pandas as pd
from ..estimator import TPOTEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tpot.selectors import survival_select_NSGA2, tournament_selection_dominated
#TODO These do not follow sklearn conventions of __init__

from ...config.template_search_spaces import get_template_search_spaces

class TPOTRegressor(TPOTEstimator):
    def __init__(       self,
                        search_space = "linear",
                        scorers=['neg_mean_squared_error'], 
                        scorers_weights=[1],
                        cv = 10, #remove this and use a value based on dataset size?
                        other_objective_functions=[], #tpot.objectives.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights = [],
                        objective_function_names = None,
                        bigger_is_better = True,
                        categorical_features = None,
                        memory = None,
                        preprocessing = False,
                        max_time_mins=60, 
                        max_eval_time_mins=10, 
                        n_jobs = 1,
                        validation_strategy = "none",
                        validation_fraction = .2, 
                        early_stop = None,
                        warm_start = False,
                        periodic_checkpoint_folder = None, 
                        verbose = 2,
                        memory_limit = None,
                        client = None,
                        random_state=None,
                        allow_inner_regressors=None,
                        **tpotestimator_kwargs,
        ):
        '''
        An sklearn baseestimator that uses genetic programming to optimize a regression pipeline.
        For more parameters, see the TPOTEstimator class.

        Parameters
        ----------

        search_space : (String, tpot.search_spaces.SearchSpace)
                        - String : The default search space to use for the optimization.
            | String     | Description      |
            | :---        |    :----:   |
            | linear  | A linear pipeline with the structure of "Selector->(transformers+Passthrough)->(classifiers/regressors+Passthrough)->final classifier/regressor." For both the transformer and inner estimator layers, TPOT may choose one or more transformers/classifiers, or it may choose none. The inner classifier/regressor layer is optional. |
            | linear-light | Same search space as linear, but without the inner classifier/regressor layer and with a reduced set of faster running estimators. |
            | graph | TPOT will optimize a pipeline in the shape of a directed acyclic graph. The nodes of the graph can include selectors, scalers, transformers, or classifiers/regressors (inner classifiers/regressors can optionally be not included). This will return a custom GraphPipeline rather than an sklearn Pipeline. More details in Tutorial 6. |
            | graph-light | Same as graph search space, but without the inner classifier/regressors and with a reduced set of faster running estimators. |
            | mdr |TPOT will search over a series of feature selectors and Multifactor Dimensionality Reduction models to find a series of operators that maximize prediction accuracy. The TPOT MDR configuration is specialized for genome-wide association studies (GWAS), and is described in detail online here.

            Note that TPOT MDR may be slow to run because the feature selection routines are computationally expensive, especially on large datasets. |
            - SearchSpace : The search space to use for the optimization. This should be an instance of a SearchSpace.
                The search space to use for the optimization. This should be an instance of a SearchSpace.
                TPOT has groups of search spaces found in the following folders, tpot.search_spaces.nodes for the nodes in the pipeline and tpot.search_spaces.pipelines for the pipeline structure.
        
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

        categorical_features : list or None
            Categorical columns to inpute and/or one hot encode during the preprocessing step. Used only if preprocessing is not False.
            
        categorical_features: list or None
            Categorical columns to inpute and/or one hot encode during the preprocessing step. Used only if preprocessing is not False.
            - None : If None, TPOT will automatically use object columns in pandas dataframes as objects for one hot encoding in preprocessing.
            - List of categorical features. If X is a dataframe, this should be a list of column names. If X is a numpy array, this should be a list of column indices


        memory: Memory object or string, default=None
            If supplied, pipeline will cache each transformer after calling fit with joblib.Memory. This feature
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
            A pipeline that will be used to preprocess the data before CV. Note that the parameters for these steps are not optimized. Add them to the search space to be optimized.
            - bool : If True, will use a default preprocessing pipeline which includes imputation followed by one hot encoding.
            - Pipeline : If an instance of a pipeline is given, will use that pipeline as the preprocessing pipeline.

        max_time_mins : float, default=float("inf")
            Maximum time to run the optimization. If none or inf, will run until the end of the generations.

        max_eval_time_mins : float, default=60*5
            Maximum time to evaluate a single individual. If none or inf, there will be no time limit per evaluation.

    
        n_jobs : int, default=1
            Number of processes to run in parallel.
            
        validation_strategy : str, default='none'
            EXPERIMENTAL The validation strategy to use for selecting the final pipeline from the population. TPOT may overfit the cross validation score. A second validation set can be used to select the final pipeline.
            - 'auto' : Automatically determine the validation strategy based on the dataset shape.
            - 'reshuffled' : Use the same data for cross validation and final validation, but with different splits for the folds. This is the default for small datasets.
            - 'split' : Use a separate validation set for final validation. Data will be split according to validation_fraction. This is the default for medium datasets.
            - 'none' : Do not use a separate validation set for final validation. Select based on the original cross-validation score. This is the default for large datasets.

        validation_fraction : float, default=0.2
          EXPERIMENTAL The fraction of the dataset to use for the validation set when validation_strategy is 'split'. Must be between 0 and 1.

        early_stop : int, default=None
            Number of generations without improvement before early stopping. All objectives must have converged within the tolerance for this to be triggered. In general a value of around 5-20 is good.

        warm_start : bool, default=False
            If True, will use the continue the evolutionary algorithm from the last generation of the previous run.

        periodic_checkpoint_folder : str, default=None
            Folder to save the population to periodically. If None, no periodic saving will be done.
            If provided, training will resume from this checkpoint.
            

        verbose : int, default=1
            How much information to print during the optimization process. Higher values include the information from lower values.
            0. nothing
            1. progress bar

            3. best individual
            4. warnings
            >=5. full warnings trace
            6. evaluations progress bar. (Temporary: This used to be 2. Currently, using evaluation progress bar may prevent some instances were we terminate a generation early due to it reaching max_time_mins in the middle of a generation OR a pipeline failed to be terminated normally and we need to manually terminate it.)


        memory_limit : str, default=None
            Memory limit for each job. See Dask [LocalCluster documentation](https://distributed.dask.org/en/stable/api.html#distributed.Client) for more information.

        client : dask.distributed.Client, default=None
            A dask client to use for parallelization. If not None, this will override the n_jobs and memory_limit parameters. If None, will create a new client with num_workers=n_jobs and memory_limit=memory_limit.

        random_state : int, None, default=None
            A seed for reproducability of experiments. This value will be passed to numpy.random.default_rng() to create an instnce of the genrator to pass to other classes

            - int
                Will be used to create and lock in Generator instance with 'numpy.random.default_rng()'
            - None
                Will be used to create Generator for 'numpy.random.default_rng()' where a fresh, unpredictable entropy will be pulled from the OS

        allow_inner_regressors : bool, default=True
            If True, the search space will include ensembled regressors.

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

        self.search_space = search_space
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
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
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
        self.allow_inner_regressors = allow_inner_regressors
        self.tpotestimator_kwargs = tpotestimator_kwargs

        self.initialized = False


    def fit(self, X, y):

        if not self.initialized:
            get_search_space_params = {"n_classes": None, 
                                        "n_samples":len(y), 
                                        "n_features":X.shape[1], 
                                        "random_state":self.random_state}

            search_space = get_template_search_spaces(self.search_space, classification=False, inner_predictors=self.allow_inner_regressors, **get_search_space_params)

            super(TPOTRegressor,self).__init__(
                search_space=search_space,
                scorers=self.scorers, 
                scorers_weights=self.scorers_weights,
                cv=self.cv,
                other_objective_functions=self.other_objective_functions, #tpot.objectives.estimator_objective_functions.number_of_nodes_objective],
                other_objective_functions_weights = self.other_objective_functions_weights,
                objective_function_names = self.objective_function_names,
                bigger_is_better = self.bigger_is_better,
                categorical_features = self.categorical_features,
                memory = self.memory,
                preprocessing = self.preprocessing,
                max_time_mins=self.max_time_mins, 
                max_eval_time_mins=self.max_eval_time_mins, 
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
                        search_space = "linear",
                        scorers=['roc_auc_ovr'], 
                        scorers_weights=[1],
                        cv = 10,
                        other_objective_functions=[], #tpot.objectives.estimator_objective_functions.number_of_nodes_objective],
                        other_objective_functions_weights = [],
                        objective_function_names = None,
                        bigger_is_better = True,
                        categorical_features = None,
                        memory = None,
                        preprocessing = False,
                        max_time_mins=60, 
                        max_eval_time_mins=10, 
                        n_jobs = 1,
                        validation_strategy = "none",
                        validation_fraction = .2, 
                        early_stop = None,
                        warm_start = False,
                        periodic_checkpoint_folder = None, 
                        verbose = 2,
                        memory_limit = None,
                        client = None,
                        random_state=None,
                        allow_inner_classifiers=None,
                        **tpotestimator_kwargs,
                        
        ):
        """
        An sklearn baseestimator that uses genetic programming to optimize a classification pipeline.
        For more parameters, see the TPOTEstimator class.

        Parameters
        ----------

        search_space : (String, tpot.search_spaces.SearchSpace)
            - String : The default search space to use for the optimization.
            | String     | Description      |
            | :---        |    :----:   |
            | linear  | A linear pipeline with the structure of "Selector->(transformers+Passthrough)->(classifiers/regressors+Passthrough)->final classifier/regressor." For both the transformer and inner estimator layers, TPOT may choose one or more transformers/classifiers, or it may choose none. The inner classifier/regressor layer is optional. |
            | linear-light | Same search space as linear, but without the inner classifier/regressor layer and with a reduced set of faster running estimators. |
            | graph | TPOT will optimize a pipeline in the shape of a directed acyclic graph. The nodes of the graph can include selectors, scalers, transformers, or classifiers/regressors (inner classifiers/regressors can optionally be not included). This will return a custom GraphPipeline rather than an sklearn Pipeline. More details in Tutorial 6. |
            | graph-light | Same as graph search space, but without the inner classifier/regressors and with a reduced set of faster running estimators. |
            | mdr |TPOT will search over a series of feature selectors and Multifactor Dimensionality Reduction models to find a series of operators that maximize prediction accuracy. The TPOT MDR configuration is specialized for genome-wide association studies (GWAS), and is described in detail online here.

            Note that TPOT MDR may be slow to run because the feature selection routines are computationally expensive, especially on large datasets. |
            - SearchSpace : The search space to use for the optimization. This should be an instance of a SearchSpace.
                The search space to use for the optimization. This should be an instance of a SearchSpace.
                TPOT has groups of search spaces found in the following folders, tpot.search_spaces.nodes for the nodes in the pipeline and tpot.search_spaces.pipelines for the pipeline structure.
        
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

        categorical_features : list or None
            Categorical columns to inpute and/or one hot encode during the preprocessing step. Used only if preprocessing is not False.
            
        categorical_features: list or None
            Categorical columns to inpute and/or one hot encode during the preprocessing step. Used only if preprocessing is not False.
            - None : If None, TPOT will automatically use object columns in pandas dataframes as objects for one hot encoding in preprocessing.
            - List of categorical features. If X is a dataframe, this should be a list of column names. If X is a numpy array, this should be a list of column indices


        memory: Memory object or string, default=None
            If supplied, pipeline will cache each transformer after calling fit with joblib.Memory. This feature
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
            A pipeline that will be used to preprocess the data before CV. Note that the parameters for these steps are not optimized. Add them to the search space to be optimized.
            - bool : If True, will use a default preprocessing pipeline which includes imputation followed by one hot encoding.
            - Pipeline : If an instance of a pipeline is given, will use that pipeline as the preprocessing pipeline.

        max_time_mins : float, default=float("inf")
            Maximum time to run the optimization. If none or inf, will run until the end of the generations.

        max_eval_time_mins : float, default=60*5
            Maximum time to evaluate a single individual. If none or inf, there will be no time limit per evaluation.

    
        n_jobs : int, default=1
            Number of processes to run in parallel.
            
        validation_strategy : str, default='none'
            EXPERIMENTAL The validation strategy to use for selecting the final pipeline from the population. TPOT may overfit the cross validation score. A second validation set can be used to select the final pipeline.
            - 'auto' : Automatically determine the validation strategy based on the dataset shape.
            - 'reshuffled' : Use the same data for cross validation and final validation, but with different splits for the folds. This is the default for small datasets.
            - 'split' : Use a separate validation set for final validation. Data will be split according to validation_fraction. This is the default for medium datasets.
            - 'none' : Do not use a separate validation set for final validation. Select based on the original cross-validation score. This is the default for large datasets.

        validation_fraction : float, default=0.2
          EXPERIMENTAL The fraction of the dataset to use for the validation set when validation_strategy is 'split'. Must be between 0 and 1.

        early_stop : int, default=None
            Number of generations without improvement before early stopping. All objectives must have converged within the tolerance for this to be triggered. In general a value of around 5-20 is good.

        warm_start : bool, default=False
            If True, will use the continue the evolutionary algorithm from the last generation of the previous run.

        periodic_checkpoint_folder : str, default=None
            Folder to save the population to periodically. If None, no periodic saving will be done.
            If provided, training will resume from this checkpoint.
            

        verbose : int, default=1
            How much information to print during the optimization process. Higher values include the information from lower values.
            0. nothing
            1. progress bar

            3. best individual
            4. warnings
            >=5. full warnings trace
            6. evaluations progress bar. (Temporary: This used to be 2. Currently, using evaluation progress bar may prevent some instances were we terminate a generation early due to it reaching max_time_mins in the middle of a generation OR a pipeline failed to be terminated normally and we need to manually terminate it.)


        memory_limit : str, default=None
            Memory limit for each job. See Dask [LocalCluster documentation](https://distributed.dask.org/en/stable/api.html#distributed.Client) for more information.

        client : dask.distributed.Client, default=None
            A dask client to use for parallelization. If not None, this will override the n_jobs and memory_limit parameters. If None, will create a new client with num_workers=n_jobs and memory_limit=memory_limit.

        random_state : int, None, default=None
            A seed for reproducability of experiments. This value will be passed to numpy.random.default_rng() to create an instnce of the genrator to pass to other classes

            - int
                Will be used to create and lock in Generator instance with 'numpy.random.default_rng()'
            - None
                Will be used to create Generator for 'numpy.random.default_rng()' where a fresh, unpredictable entropy will be pulled from the OS

        allow_inner_classifiers : bool, default=True
            If True, the search space will include ensembled classifiers. 

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
        """
        self.search_space = search_space
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
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
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
        self.allow_inner_classifiers = allow_inner_classifiers

        self.initialized = False

    def fit(self, X, y):

        if not self.initialized:

            get_search_space_params = {"n_classes": len(np.unique(y)), 
                                       "n_samples":len(y), 
                                       "n_features":X.shape[1], 
                                       "random_state":self.random_state}

            search_space = get_template_search_spaces(self.search_space, classification=True, inner_predictors=self.allow_inner_classifiers, **get_search_space_params)


            super(TPOTClassifier,self).__init__(
                search_space=search_space,
                scorers=self.scorers, 
                scorers_weights=self.scorers_weights,
                cv = self.cv,
                other_objective_functions=self.other_objective_functions, #tpot.objectives.estimator_objective_functions.number_of_nodes_objective],
                other_objective_functions_weights = self.other_objective_functions_weights,
                objective_function_names = self.objective_function_names,
                bigger_is_better = self.bigger_is_better,
                categorical_features = self.categorical_features,
                memory = self.memory,
                preprocessing = self.preprocessing,
                max_time_mins=self.max_time_mins, 
                max_eval_time_mins=self.max_eval_time_mins, 
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
