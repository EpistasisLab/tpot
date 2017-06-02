# -*- coding: utf-8 -*-

"""Copyright 2015-Present Randal S. Olson.

This file is part of the TPOT library.

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

from __future__ import print_function
import random
import inspect
import warnings
import sys
import imp
from functools import partial
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import deap
from deap import base, creator, tools, gp
from tqdm import tqdm
from copy import copy

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer

from update_checker import update_check

from ._version import __version__
from .operator_utils import TPOTOperatorClassFactory, Operator, ARGType
from .export_utils import export_pipeline, expr_to_tree, generate_pipeline_code
from .decorators import _pre_test
from .builtins import CombineDFs, StackingEstimator
from .config.classifier_light import classifier_config_dict_light
from .config.regressor_light import regressor_config_dict_light
from .config.classifier_mdr import tpot_mdr_classifier_config_dict
from .config.regressor_mdr import tpot_mdr_regressor_config_dict

from .metrics import SCORERS
from .gp_types import Output_Array
from .gp_deap import eaMuPlusLambda, mutNodeReplacement, _wrapped_cross_val_score, cxOnePoint

# hot patch for Windows: solve the problem of crashing python after Ctrl + C in Windows OS
if sys.platform.startswith('win'):
    import win32api
    try:
        import _thread
    except ImportError:
        import thread as _thread

    def handler(dwCtrlType, hook_sigint=_thread.interrupt_main):
        """SIGINT handler function."""
        if dwCtrlType == 0:  # CTRL_C_EVENT
            hook_sigint()
            return 1  # don't chain to the next handler
        return 0
    win32api.SetConsoleCtrlHandler(handler, 1)


class TPOTBase(BaseEstimator):
    """Automatically creates and optimizes machine learning pipelines using GP."""

    def __init__(self, generations=100, population_size=100, offspring_size=None,
                 mutation_rate=0.9, crossover_rate=0.1,
                 scoring=None, cv=5, subsample=1.0, n_jobs=1,
                 max_time_mins=None, max_eval_time_mins=5,
                 random_state=None, config_dict=None, warm_start=False,
                 verbosity=0, disable_update_check=False):
        """Set up the genetic programming algorithm for pipeline optimization.

        Parameters
        ----------
        generations: int, optional (default: 100)
            Number of iterations to the run pipeline optimization process.
            Generally, TPOT will work better when you give it more generations (and
            therefore time) to optimize the pipeline. TPOT will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.
        population_size: int, optional (default: 100)
            Number of individuals to retain in the GP population every generation.
            Generally, TPOT will work better when you give it more individuals
            (and therefore time) to optimize the pipeline. TPOT will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.
        offspring_size: int, optional (default: None)
            Number of offspring to produce in each GP generation.
            By default, offspring_size = population_size.
        mutation_rate: float, optional (default: 0.9)
            Mutation rate for the genetic programming algorithm in the range [0.0, 1.0].
            This parameter tells the GP algorithm how many pipelines to apply random
            changes to every generation. We recommend using the default parameter unless
            you understand how the mutation rate affects GP algorithms.
        crossover_rate: float, optional (default: 0.1)
            Crossover rate for the genetic programming algorithm in the range [0.0, 1.0].
            This parameter tells the genetic programming algorithm how many pipelines to
            "breed" every generation. We recommend using the default parameter unless you
            understand how the mutation rate affects GP algorithms.
        scoring: string or callable, optional
            Function used to evaluate the quality of a given pipeline for the
            problem. By default, accuracy is used for classification problems and
            mean squared error (MSE) for regression problems.

            Offers the same options as sklearn.model_selection.cross_val_score as well as
            a built-in score 'balanced_accuracy'. Classification metrics:

            ['accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',
            'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
            'precision', 'precision_macro', 'precision_micro', 'precision_samples',
            'precision_weighted', 'recall', 'recall_macro', 'recall_micro',
            'recall_samples', 'recall_weighted', 'roc_auc']

            Regression metrics:

            ['neg_median_absolute_error', 'neg_mean_absolute_error',
            'neg_mean_squared_error', 'r2']

            If you would like to use a custom scoring function, you can pass a callable
            function to this parameter with the signature scorer(y_true, y_pred).
            See the section on scoring functions in the documentation for more details.

            TPOT assumes that any custom scoring function with "error" or "loss" in the
            name is meant to be minimized, whereas any other functions will be maximized.
        cv: int or cross-validation generator, optional (default: 5)
            If CV is a number, then it is the number of folds to evaluate each
            pipeline over in k-fold cross-validation during the TPOT optimization
             process. If it is an object then it is an object to be used as a
             cross-validation generator.
        subsample: float, optional (default: 1.0)
            Subsample ratio of the training instance. Setting it to 0.5 means that TPOT
            randomly collects half of training samples for pipeline optimization process.
        n_jobs: int, optional (default: 1)
            Number of CPUs for evaluating pipelines in parallel during the TPOT
            optimization process. Assigning this to -1 will use as many cores as available
            on the computer.
        max_time_mins: int, optional (default: None)
            How many minutes TPOT has to optimize the pipeline.
            If provided, this setting will override the "generations" parameter and allow
            TPOT to run until it runs out of time.
        max_eval_time_mins: int, optional (default: 5)
            How many minutes TPOT has to optimize a single pipeline.
            Setting this parameter to higher values will allow TPOT to explore more
            complex pipelines, but will also allow TPOT to run longer.
        random_state: int, optional (default: None)
            Random number generator seed for TPOT. Use this parameter to make sure
            that TPOT will give you the same results each time you run it against the
            same data set with that seed.
        config_dict: a Python dictionary or string, optional (default: None)
            Python dictionary:
                A dictionary customizing the operators and parameters that
                TPOT uses in the optimization process.
                For examples, see config_regressor.py and config_classifier.py
            Path for configuration file:
                A path to a configuration file for customizing the operators and parameters that
                TPOT uses in the optimization process.
                For examples, see config_regressor.py and config_classifier.py
            String 'TPOT light':
                TPOT uses a light version of operator configuration dictionary instead of
                the default one.
            String 'TPOT MDR':
                TPOT uses a list of TPOT-MDR operator configuration dictionary instead of
                the default one.
        warm_start: bool, optional (default: False)
            Flag indicating whether the TPOT instance will reuse the population from
            previous calls to fit().
        verbosity: int, optional (default: 0)
            How much information TPOT communicates while it's running.
            0 = none, 1 = minimal, 2 = high, 3 = all.
            A setting of 2 or higher will add a progress bar during the optimization procedure.
        disable_update_check: bool, optional (default: False)
            Flag indicating whether the TPOT version checker should be disabled.

        Returns
        -------
        None

        """
        if self.__class__.__name__ == 'TPOTBase':
            raise RuntimeError('Do not instantiate the TPOTBase class directly; use TPOTRegressor or TPOTClassifier instead.')

        # Prompt the user if their version is out of date
        self.disable_update_check = disable_update_check
        if not self.disable_update_check:
            update_check('tpot', __version__)

        self._pareto_front = None
        self._optimized_pipeline = None
        self.fitted_pipeline_ = None
        self._fitted_imputer = None
        self._pop = None
        self.warm_start = warm_start
        self.population_size = population_size
        self.generations = generations
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins

        # Set offspring_size equal to population_size by default
        if offspring_size:
            self.offspring_size = offspring_size
        else:
            self.offspring_size = population_size

        self._setup_config(config_dict)

        self.operators = []
        self.arguments = []
        for key in sorted(self.config_dict.keys()):
            op_class, arg_types = TPOTOperatorClassFactory(
                key,
                self.config_dict[key],
                BaseClass=Operator,
                ArgBaseClass=ARGType
            )
            if op_class:
                self.operators.append(op_class)
                self.arguments += arg_types

        # Schedule TPOT to run for many generations if the user specifies a
        # run-time limit TPOT will automatically interrupt itself when the timer
        # runs out
        if max_time_mins is not None:
            self.generations = 1000000

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        if self.mutation_rate + self.crossover_rate > 1:
            raise ValueError(
                'The sum of the crossover and mutation probabilities must be <= 1.0.'
            )

        self.verbosity = verbosity
        self.operators_context = {
            'make_pipeline': make_pipeline,
            'make_union': make_union,
            'StackingEstimator': StackingEstimator,
            'FunctionTransformer': FunctionTransformer,
            'copy': copy
        }
        self._pbar = None

        # Dictionary of individuals that have already been evaluated in previous
        # generations
        self.evaluated_individuals_ = {}
        self.random_state = random_state

        # If the user passed a custom scoring function, store it in the sklearn
        # SCORERS dictionary
        if scoring:
            if hasattr(scoring, '__call__'):
                scoring_name = scoring.__name__
                greater_is_better = 'loss' not in scoring_name and 'error' not in scoring_name
                SCORERS[scoring_name] = make_scorer(scoring, greater_is_better=greater_is_better)
                self.scoring_function = scoring_name
            else:
                if scoring not in SCORERS:
                    raise ValueError(
                        'The scoring function {} is not available. Please '
                        'choose a valid scoring function from the TPOT '
                        'documentation.'.format(scoring)
                    )
                self.scoring_function = scoring

        self.cv = cv
        self.subsample = subsample
        if self.subsample <= 0.0 or self.subsample > 1.0:
            raise ValueError(
                'The subsample ratio of the training instance must be in the range (0.0, 1.0].'
            )
        # If the OS is windows, reset cpu number to 1 since the OS did not have multiprocessing module
        if sys.platform.startswith('win') and n_jobs != 1:
            print(
                'Warning: Although parallelization is currently supported in '
                'TPOT for Windows, pressing Ctrl+C will freeze the optimization '
                'process without saving the best pipeline! Thus, Please DO NOT '
                'press Ctrl+C during the optimization procss if n_jobs is not '
                'equal to 1. For quick test in Windows, please set n_jobs to 1 '
                'for saving the best pipeline in the middle of the optimization '
                'process via Ctrl+C.'
            )
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self._setup_pset()
        self._setup_toolbox()

    def _setup_config(self, config_dict):
        if config_dict:
            if isinstance(config_dict, dict):
                self.config_dict = config_dict
            elif config_dict == 'TPOT light':
                if self.classification:
                    self.config_dict = classifier_config_dict_light
                else:
                    self.config_dict = regressor_config_dict_light
            elif config_dict == 'TPOT MDR':
                if self.classification:
                    self.config_dict = tpot_mdr_classifier_config_dict
                else:
                    self.config_dict = tpot_mdr_regressor_config_dict
            else:
                self.config_dict = self._read_config_file(config_dict)
        else:
            self.config_dict = self.default_config_dict

    def _read_config_file(self, config_path):
        try:
            custom_config = imp.new_module('custom_config')

            with open(config_path, 'r') as config_file:
                file_string = config_file.read()
                exec(file_string, custom_config.__dict__)

            return custom_config.tpot_config
        except FileNotFoundError as e:
            raise FileNotFoundError(
                'Could not open specified TPOT operator config file: '
                '{}'.format(e.filename)
            )
        except AttributeError:
            raise AttributeError(
                'The supplied TPOT operator config file does not contain '
                'a dictionary named "tpot_config".'
            )
        except Exception as e:
            raise type(e)(
                'An error occured while attempting to read the specified '
                'custom TPOT operator configuration file.'
            )

    def _setup_pset(self):
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        self._pset = gp.PrimitiveSetTyped('MAIN', [np.ndarray], Output_Array)
        self._pset.renameArguments(ARG0='input_matrix')
        self._add_operators()
        self._add_terminals()

        if self.verbosity > 2:
            print('{} operators have been imported by TPOT.'.format(len(self.operators)))

    def _add_operators(self):
        for operator in self.operators:
            if operator.root:
                # We need to add rooted primitives twice so that they can
                # return both an Output_Array (and thus be the root of the tree),
                # and return a np.ndarray so they can exist elsewhere in the tree.
                p_types = (operator.parameter_types()[0], Output_Array)
                self._pset.addPrimitive(operator, *p_types)

            self._pset.addPrimitive(operator, *operator.parameter_types())

            # Import required modules into local namespace so that pipelines
            # may be evaluated directly
            for key in sorted(operator.import_hash.keys()):
                module_list = ', '.join(sorted(operator.import_hash[key]))

                if key.startswith('tpot.'):
                    exec('from {} import {}'.format(key[4:], module_list))
                else:
                    exec('from {} import {}'.format(key, module_list))

                for var in operator.import_hash[key]:
                    self.operators_context[var] = eval(var)

        self._pset.addPrimitive(CombineDFs(), [np.ndarray, np.ndarray], np.ndarray)

    def _add_terminals(self):
        for _type in self.arguments:
            type_values = list(_type.values)
            if 'nthread' not in _type.__name__:
                type_values += ['DEFAULT']

            for val in type_values:
                terminal_name = _type.__name__ + "=" + str(val)
                self._pset.addTerminal(val, _type, name=terminal_name)

    def _setup_toolbox(self):
        creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self._toolbox = base.Toolbox()
        self._toolbox.register('expr', self._gen_grow_safe, pset=self._pset, min_=1, max_=3)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register('compile', self._compile_to_sklearn)
        self._toolbox.register('select', tools.selNSGA2)
        self._toolbox.register('mate', self._mate_operator)
        self._toolbox.register('expr_mut', self._gen_grow_safe, min_=1, max_=4)
        self._toolbox.register('mutate', self._random_mutation_operator)

    def fit(self, features, target, sample_weight=None, groups=None):
        """Fit an optimized machine learning pipeline.

        Uses genetic programming to optimize a machine learning pipeline that
        maximizes score on the provided features and target. Performs internal
        k-fold cross-validaton to avoid overfitting on the training data.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix

            TPOT and all scikit-learn algorithms assume that the features will be numerical
            and there will be no missing values. As such, when a feature matrix is provided
            to TPOT, all missing values will automatically be replaced (i.e., imputed) using
            median value imputation.

            If you wish to use a different imputation strategy than median imputation, please
            make sure to apply imputation to your feature set prior to passing it to TPOT.
        target: array-like {n_samples}
            List of class labels for prediction
        sample_weight: array-like {n_samples}, optional
            Per-sample weights. Higher weights force TPOT to put more emphasis on those points
        groups: array-like, with shape {n_samples, }, optional
            Group labels for the samples used when performing cross-validation.
            This parameter should only be used in conjunction with sklearn's Group cross-validation
            functions, such as sklearn.model_selection.GroupKFold

        Returns
        -------
        self: object
            Returns a copy of the fitted TPOT object

        """
        features = features.astype(np.float64)

        # Resets the imputer to be fit for the new dataset
        self._fitted_imputer = None

        if np.any(np.isnan(features)):
            features = self._impute_values(features)

        self._check_dataset(features, target)

        # Randomly collect a subsample of training samples for pipeline optimization process.
        if self.subsample < 1.0:
            features, _, target, _ = train_test_split(features, target, train_size=self.subsample, random_state=self.random_state)
            # Raise a warning message if the training size is less than 1500 when subsample is not default value
            if features.shape[0] < 1500:
                print(
                    'Warning: Although subsample can accelerate pipeline optimization process, '
                    'too small training sample size may cause unpredictable effect on maximizing '
                    'score in pipeline optimization process. Increasing subsample ratio may get '
                    'a more reasonable outcome from optimization process in TPOT.'
                    )

        # Set the seed for the GP run
        if self.random_state is not None:
            random.seed(self.random_state)  # deap uses random
            np.random.seed(self.random_state)

        self._start_datetime = datetime.now()
        self._toolbox.register('evaluate', self._evaluate_individuals, features=features, target=target, sample_weight=sample_weight, groups=groups)

        # assign population, self._pop can only be not None if warm_start is enabled
        if self._pop:
            pop = self._pop
        else:
            pop = self._toolbox.population(n=self.population_size)

        def pareto_eq(ind1, ind2):
            """Determine whether two individuals are equal on the Pareto front.

            Parameters
            ----------
            ind1: DEAP individual from the GP population
                First individual to compare
            ind2: DEAP individual from the GP population
                Second individual to compare

            Returns
            ----------
            individuals_equal: bool
                Boolean indicating whether the two individuals are equal on
                the Pareto front

            """
            return np.allclose(ind1.fitness.values, ind2.fitness.values)

        # Generate new pareto front if it doesn't already exist for warm start
        if not self.warm_start or not self._pareto_front:
            self._pareto_front = tools.ParetoFront(similar=pareto_eq)

        # Start the progress bar
        if self.max_time_mins:
            total_evals = self.population_size
        else:
            total_evals = self.offspring_size * self.generations + self.population_size

        self._pbar = tqdm(total=total_evals, unit='pipeline', leave=False,
                          disable=not (self.verbosity >= 2), desc='Optimization Progress')

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pop, _ = eaMuPlusLambda(
                    population=pop,
                    toolbox=self._toolbox,
                    mu=self.population_size,
                    lambda_=self.offspring_size,
                    cxpb=self.crossover_rate,
                    mutpb=self.mutation_rate,
                    ngen=self.generations,
                    pbar=self._pbar,
                    halloffame=self._pareto_front,
                    verbose=self.verbosity,
                    max_time_mins=self.max_time_mins
                )

            # store population for the next call
            if self.warm_start:
                self._pop = pop

        # Allow for certain exceptions to signal a premature fit() cancellation
        except (KeyboardInterrupt, SystemExit):
            if self.verbosity > 0:
                self._pbar.write('')
                self._pbar.write('TPOT closed prematurely. Will use the current best pipeline.')
        finally:
            # keep trying 10 times in case weird things happened like multiple CTRL+C or exceptions
            attempts = 10
            for attempt in range(attempts):
                try:
                    # Close the progress bar
                    # Standard truthiness checks won't work for tqdm
                    if not isinstance(self._pbar, type(None)):
                        self._pbar.close()

                    # Store the pipeline with the highest internal testing score
                    if self._pareto_front:
                        self._update_top_pipeline()

                        # It won't raise error for a small test like in a unit test because a few pipeline sometimes
                        # may fail due to the training data does not fit the operator's requirement.
                        if not self._optimized_pipeline:
                            print('There was an error in the TPOT optimization '
                                  'process. This could be because the data was '
                                  'not formatted properly, or because data for '
                                  'a regression problem was provided to the '
                                  'TPOTClassifier object. Please make sure you '
                                  'passed the data to TPOT correctly.')
                        else:
                            self.fitted_pipeline_ = self._toolbox.compile(expr=self._optimized_pipeline)

                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                self.fitted_pipeline_.fit(features, target)

                            if self.verbosity in [1, 2]:
                                # Add an extra line of spacing if the progress bar was used
                                if self.verbosity >= 2:
                                    print('')
                                print('Best pipeline: {}'.format(self._optimized_pipeline))

                            # Store and fit the entire Pareto front as fitted models for convenience
                            self.pareto_front_fitted_pipelines_ = {}

                            for pipeline in self._pareto_front.items:
                                self.pareto_front_fitted_pipelines_[str(pipeline)] = self._toolbox.compile(expr=pipeline)
                                with warnings.catch_warnings():
                                    warnings.simplefilter('ignore')
                                    self.pareto_front_fitted_pipelines_[str(pipeline)].fit(features, target)
                    break

                except (KeyboardInterrupt, SystemExit, Exception) as e:
                    # raise the exception if it's our last attempt
                    if attempt == (attempts - 1):
                        raise
            return self

    def _update_top_pipeline(self):
        """Helper function to update the _optimized_pipeline field."""
        if self._pareto_front:
            top_score = -float('inf')
            for pipeline, pipeline_scores in zip(self._pareto_front.items, reversed(self._pareto_front.keys)):
                if pipeline_scores.wvalues[1] > top_score:
                    self._optimized_pipeline = pipeline
                    top_score = pipeline_scores.wvalues[1]

    def predict(self, features):
        """Use the optimized pipeline to predict the target for a feature set.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix

        Returns
        ----------
        array-like: {n_samples}
            Predicted target for the samples in the feature matrix

        """
        if not self.fitted_pipeline_:
            raise RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')

        features = features.astype(np.float64)

        if np.any(np.isnan(features)):
            features = self._impute_values(features)

        return self.fitted_pipeline_.predict(features)

    def fit_predict(self, features, target):
        """Call fit and predict in sequence.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        target: array-like {n_samples}
            List of class labels for prediction

        Returns
        ----------
        array-like: {n_samples}
            Predicted target for the provided features

        """
        self.fit(features, target)
        return self.predict(features)

    def score(self, testing_features, testing_target):
        """Returns the score on the given testing data using the user-specified scoring function.

        Parameters
        ----------
        testing_features: array-like {n_samples, n_features}
            Feature matrix of the testing set
        testing_target: array-like {n_samples}
            List of class labels for prediction in the testing set

        Returns
        -------
        accuracy_score: float
            The estimated test set accuracy

        """
        if self.fitted_pipeline_ is None:
            raise RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')

        # If the scoring function is a string, we must adjust to use the sklearn
        # scoring interface
        score = SCORERS[self.scoring_function](
            self.fitted_pipeline_,
            testing_features.astype(np.float64),
            testing_target.astype(np.float64)
        )
        return abs(score)

    def predict_proba(self, features):
        """Use the optimized pipeline to estimate the class probabilities for a feature set.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix of the testing set

        Returns
        -------
        array-like: {n_samples, n_target}
            The class probabilities of the input samples

        """
        if not self.fitted_pipeline_:
            raise RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')
        else:
            if not(hasattr(self.fitted_pipeline_, 'predict_proba')):
                raise RuntimeError('The fitted pipeline does not have the predict_proba() function.')
            return self.fitted_pipeline_.predict_proba(features.astype(np.float64))

    def set_params(self, **params):
        """Set the parameters of TPOT.

        Returns
        -------
        self
        """
        self.__init__(**params)

        return self

    def export(self, output_file_name):
        """Export the optimized pipeline as Python code.

        Parameters
        ----------
        output_file_name: string
            String containing the path and file name of the desired output file

        Returns
        -------
        None

        """
        if self._optimized_pipeline is None:
            raise RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')

        with open(output_file_name, 'w') as output_file:
            output_file.write(export_pipeline(self._optimized_pipeline, self.operators, self._pset))

    def _impute_values(self, features):
        """Impute missing values in a feature set.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            A feature matrix

        Returns
        -------
        array-like {n_samples, n_features}
        """
        if self.verbosity > 1:
            print('Imputing missing values in feature set')

        if self._fitted_imputer is None:
            self._fitted_imputer = Imputer(strategy="median", axis=1)
            self._fitted_imputer.fit(features)

        return self._fitted_imputer.transform(features)

    def _check_dataset(self, features, target):
        """Check if a dataset has a valid feature set and labels.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        target: array-like {n_samples}
            List of class labels for prediction

        Returns
        -------
        None
        """
        try:
            check_X_y(features, target, accept_sparse=False)
        except (AssertionError, ValueError):
            raise ValueError(
                'Error: Input data is not in a valid format. Please confirm '
                'that the input data is scikit-learn compatible. For example, '
                'the features must be a 2-D array and target labels must be a '
                '1-D array.'
            )

    def _compile_to_sklearn(self, expr):
        """Compile a DEAP pipeline into a sklearn pipeline.

        Parameters
        ----------
        expr: DEAP individual
            The DEAP pipeline to be compiled

        Returns
        -------
        sklearn_pipeline: sklearn.pipeline.Pipeline
        """
        sklearn_pipeline = generate_pipeline_code(expr_to_tree(expr, self._pset), self.operators)
        return eval(sklearn_pipeline, self.operators_context)

    def _set_param_recursive(self, pipeline_steps, parameter, value):
        """Recursively iterate through all objects in the pipeline and set a given parameter.

        Parameters
        ----------
        pipeline_steps: array-like
            List of (str, obj) tuples from a scikit-learn pipeline or related object
        parameter: str
            The parameter to assign a value for in each pipeline object
        value: any
            The value to assign the parameter to in each pipeline object
        Returns
        -------
        None

        """
        for (_, obj) in pipeline_steps:
            recursive_attrs = ['steps', 'transformer_list', 'estimators']

            for attr in recursive_attrs:
                if hasattr(obj, attr):
                    self._set_param_recursive(getattr(obj, attr), parameter, value)
                    break
            else:
                if hasattr(obj, parameter):
                    setattr(obj, parameter, value)


    def _evaluate_individuals(self, individuals, features, target, sample_weight=None, groups=None):
        """Determine the fit of the provided individuals.

        Parameters
        ----------
        individuals: a list of DEAP individual
            One individual is a list of pipeline operators and model parameters that can be
            compiled by DEAP into a callable function
        features: numpy.ndarray {n_samples, n_features}
            A numpy matrix containing the training and testing features for the individual's evaluation
        target: numpy.ndarray {n_samples}
            A numpy matrix containing the training and testing target for the individual's evaluation
        sample_weight: array-like {n_samples}, optional
            List of sample weights to balance (or un-balanace) the dataset target as needed
        groups: array-like {n_samples, }, optional
            Group labels for the samples used while splitting the dataset into train/test set

        Returns
        -------
        fitnesses_ordered: float
            Returns a list of tuple value indicating the individual's fitness
            according to its performance on the provided data

        """
        if self.max_time_mins:
            total_mins_elapsed = (datetime.now() - self._start_datetime).total_seconds() / 60.
            if total_mins_elapsed >= self.max_time_mins:
                raise KeyboardInterrupt('{} minutes have elapsed. TPOT will close down.'.format(total_mins_elapsed))

        # return fitness scores
        fitnesses_dict = {}
        # 4 lists of DEAP individuals, their sklearn pipelines and their operator counts for parallel computing
        eval_individuals_str = []
        sklearn_pipeline_list = []
        operator_count_list = []
        test_idx_list = []
        for indidx, individual in enumerate(individuals):
            # Disallow certain combinations of operators because they will take too long or take up too much RAM
            # This is a fairly hacky way to prevent TPOT from getting stuck on bad pipelines and should be improved in a future release
            individual = individuals[indidx]
            individual_str = str(individual)
            sklearn_pipeline_str = generate_pipeline_code(expr_to_tree(individual, self._pset), self.operators)
            if sklearn_pipeline_str.count('PolynomialFeatures') > 1:
                if self.verbosity > 2:
                    self._pbar.write('Invalid pipeline encountered. Skipping its evaluation.')
                fitnesses_dict[indidx] = (5000., -float('inf'))
                if not self._pbar.disable:
                    self._pbar.update(1)
            # Check if the individual was evaluated before
            elif individual_str in self.evaluated_individuals_:
                # Get fitness score from previous evaluation
                fitnesses_dict[indidx] = self.evaluated_individuals_[individual_str]
                if self.verbosity > 2:
                    self._pbar.write('Pipeline encountered that has previously been evaluated during the '
                                     'optimization process. Using the score from the previous evaluation.')
                if not self._pbar.disable:
                    self._pbar.update(1)
            else:
                try:
                    # Transform the tree expression into an sklearn pipeline
                    sklearn_pipeline = self._toolbox.compile(expr=individual)


                    # Fix random state when the operator allows and build sample weight dictionary
                    self._set_param_recursive(sklearn_pipeline.steps, 'random_state', 42)

                    # Count the number of pipeline operators as a measure of pipeline complexity
                    operator_count = self._operator_count(individual)

                except Exception:
                    fitnesses_dict[indidx] = (5000., -float('inf'))
                    if not self._pbar.disable:
                        self._pbar.update(1)
                    continue
                eval_individuals_str.append(individual_str)
                operator_count_list.append(operator_count)
                sklearn_pipeline_list.append(sklearn_pipeline)
                test_idx_list.append(indidx)

        # evalurate pipeline
        resulting_score_list = []
        # chunk size for pbar update
        for chunk_idx in range(0, len(sklearn_pipeline_list), self.n_jobs * 4):
            jobs = []
            for sklearn_pipeline in sklearn_pipeline_list[chunk_idx:chunk_idx + self.n_jobs * 4]:
                job = delayed(_wrapped_cross_val_score)(
                    sklearn_pipeline=sklearn_pipeline,
                    features=features,
                    target=target,
                    cv=self.cv,
                    scoring_function=self.scoring_function,
                    sample_weight=sample_weight,
                    max_eval_time_mins=self.max_eval_time_mins,
                    groups=groups
                )
                jobs.append(job)
            parallel = Parallel(n_jobs=self.n_jobs, verbose=0, pre_dispatch='2*n_jobs')
            tmp_result_score = parallel(jobs)

            # update pbar
            for val in tmp_result_score:
                if not self._pbar.disable:
                    self._pbar.update(1)
                if val == 'Timeout':
                    if self.verbosity > 2:
                        self._pbar.write('Skipped pipeline #{0} due to time out. '
                                         'Continuing to the next pipeline.'.format(self._pbar.n))
                    resulting_score_list.append(-float('inf'))
                else:
                    resulting_score_list.append(val)

        for resulting_score, operator_count, individual_str, test_idx in zip(resulting_score_list, operator_count_list, eval_individuals_str, test_idx_list):
            if type(resulting_score) in [float, np.float64, np.float32]:
                self.evaluated_individuals_[individual_str] = (max(1, operator_count), resulting_score)
                fitnesses_dict[test_idx] = self.evaluated_individuals_[individual_str]
            else:
                raise ValueError('Scoring function does not return a float.')

        fitnesses_ordered = []
        for key in sorted(fitnesses_dict.keys()):
            fitnesses_ordered.append(fitnesses_dict[key])
        return fitnesses_ordered

    @_pre_test
    def _mate_operator(self, ind1, ind2):
        return cxOnePoint(ind1, ind2)

    @_pre_test
    def _random_mutation_operator(self, individual):
        """Perform a replacement, insertion, or shrink mutation on an individual.

        Parameters
        ----------
        individual: DEAP individual
            A list of pipeline operators and model parameters that can be
            compiled by DEAP into a callable function

        Returns
        -------
        mut_ind: DEAP individual
            Returns the individual with one of the mutations applied to it

        """
        mutation_techniques = [
            partial(gp.mutInsert, pset=self._pset),
            partial(mutNodeReplacement, pset=self._pset),
            partial(gp.mutShrink)
        ]
        return np.random.choice(mutation_techniques)(individual)

    def _gen_grow_safe(self, pset, min_, max_, type_=None):
        """Generate an expression where each leaf might have a different depth between min_ and max_.

        Parameters
        ----------
        pset: PrimitiveSetTyped
            Primitive set from which primitives are selected.
        min_: int
            Minimum height of the produced trees.
        max_: int
            Maximum Height of the produced trees.
        type_: class
            The type that should return the tree when called, when
                  :obj:None (default) the type of :pset: (pset.ret)
                  is assumed.
        Returns
        -------
        individual: list
            A grown tree with leaves at possibly different depths.
        """
        def condition(height, depth, type_):
            """Stop when the depth is equal to height or when a node should be a terminal."""
            return type_ not in [np.ndarray, Output_Array] or depth == height

        return self._generate(pset, min_, max_, condition, type_)

    # Count the number of pipeline operators as a measure of pipeline complexity
    def _operator_count(self, individual):
        operator_count = 0
        for i in range(len(individual)):
            node = individual[i]
            if type(node) is deap.gp.Primitive and node.name != 'CombineDFs':
                operator_count += 1
        return operator_count

    # Generate function stolen straight from deap.gp.generate
    @_pre_test
    def _generate(self, pset, min_, max_, condition, type_=None):
        """Generate a Tree as a list of lists.

        The tree is build from the root to the leaves, and it stop growing when
        the condition is fulfilled.

        Parameters
        ----------
        pset: PrimitiveSetTyped
            Primitive set from which primitives are selected.
        min_: int
            Minimum height of the produced trees.
        max_: int
            Maximum Height of the produced trees.
        condition: function
            The condition is a function that takes two arguments,
            the height of the tree to build and the current
            depth in the tree.
        type_: class
            The type that should return the tree when called, when
            :obj:None (default) no return type is enforced.

        Returns
        -------
        individual: list
            A grown tree with leaves at possibly different depths
            dependending on the condition function.
        """
        if type_ is None:
            type_ = pset.ret
        expr = []
        height = np.random.randint(min_, max_)
        stack = [(0, type_)]
        while len(stack) != 0:
            depth, type_ = stack.pop()

            # We've added a type_ parameter to the condition function
            if condition(height, depth, type_):
                try:
                    term = np.random.choice(pset.terminals[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError(
                        'The gp.generate function tried to add '
                        'a terminal of type {}, but there is'
                        'none available. {}'.format(type_, traceback)
                        )
                if inspect.isclass(term):
                    term = term()
                expr.append(term)
            else:
                try:
                    prim = np.random.choice(pset.primitives[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError(
                        'The gp.generate function tried to add '
                        'a primitive of type {}, but there is'
                        'none available. {}'.format(type_, traceback)
                        )
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth+1, arg))
        return expr
