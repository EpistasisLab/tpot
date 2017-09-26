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
import os
import re

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
from .config.regressor_sparse import regressor_config_sparse
from .config.classifier_sparse import classifier_config_sparse

from .metrics import SCORERS
from .gp_types import Output_Array
from .gp_deap import eaMuPlusLambda, mutNodeReplacement, _wrapped_cross_val_score, cxOnePoint


# hot patch for Windows: solve the problem of crashing python after Ctrl + C in Windows OS
# https://github.com/ContinuumIO/anaconda-issues/issues/905
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
                 random_state=None, config_dict=None, population_seeds=None,
                 warm_start=False, periodic_checkpoint_folder=None, early_stop=None,
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
            String 'TPOT sparse':
                TPOT uses a configuration dictionary with a one-hot-encoder and the
                operators normally included in TPOT that also support sparse matrices.
        population_seeds: a Python list or a string, optional (Default: None)
            Python list:
                A list customizing a set of pipelines used in the first generation.
            Path for configuration file:
                A path to a configuration file for customizing a set of pipelines used in
                the first generation.
        warm_start: bool, optional (default: False)
            Flag indicating whether the TPOT instance will reuse the population from
            previous calls to fit().
        periodic_checkpoint_folder: path string, optional (default: None)
            If supplied, a folder in which tpot will periodically save the best pipeline so far while optimizing.
            Currently once per generation but not more often than once per 30 seconds.
            Useful in multiple cases:
                Sudden death before tpot could save optimized pipeline
                Track its progress
                Grab pipelines while it's still optimizing
        early_stop: int or None (default: None)
            How many generations TPOT checks whether there is no improvement in optimization process.
            End optimization process if there is no improvement in the set number of generations.
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
        self._optimized_pipeline_score = None
        self._exported_pipeline_text = ""
        self.fitted_pipeline_ = None
        self._fitted_imputer = None
        self._imputed = False
        self._pop = []
        self.warm_start = warm_start
        self.population_size = population_size
        self.generations = generations
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.max_eval_time_seconds = max(int(self.max_eval_time_mins * 60), 1)
        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.early_stop = early_stop
        self._last_optimized_pareto_front = None
        self._last_optimized_pareto_front_n_gens = 0

        # dont save periodic pipelines more often than this
        self._output_best_pipeline_period_seconds = 30

        # Try crossover and mutation at most this many times for
        # any one given individual (or pair of individuals)
        self._max_mut_loops = 50

        # Set offspring_size equal to population_size by default
        if offspring_size:
            self.offspring_size = offspring_size
        else:
            self.offspring_size = population_size

        self.config_dict_params=config_dict
        self._setup_config(self.config_dict_params)

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
        # Specifies where to output the progress messages (default: sys.stdout).
        # Maybe open this API in future version of TPOT.(io.TextIOWrapper or io.StringIO)
        self._file = sys.stdout

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
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self._setup_pset()
        self._setup_toolbox()
        self._setup_pop(population_seeds)


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
            elif config_dict == 'TPOT sparse':
                if self.classification:
                    self.config_dict = classifier_config_sparse
                else:
                    self.config_dict = regressor_config_sparse
            else:
                config = self._read_config_file(config_dict)
                if hasattr(config, 'tpot_config'):
                    self.config_dict = config.tpot_config
                else:
                    raise ValueError(
                                    'Could not find "tpot_config" in configuration file {}. '
                                    'When using a custom config file for customizing operators '
                                    'dictionary, the file must have a python dictionary with '
                                    'the standardized name of "tpot_config"'.format(config_dict)
                                    )
        else:
            self.config_dict = self.default_config_dict


    def _read_config_file(self, config_path):
        if os.path.isfile(config_path):
            try:
                custom_config = imp.new_module('custom_config')

                with open(config_path, 'r') as config_file:
                    file_string = config_file.read()
                    exec(file_string, custom_config.__dict__)
                return custom_config
            except Exception as e:
                raise ValueError(
                    'An error occured while attempting to read the specified '
                    'custom TPOT operator configuration file: {}'.format(e)
                )
        else:
            raise ValueError(
                'Could not open specified TPOT operator config file: '
                '{}'.format(config_path)
            )


    def _setup_pop(self, population_seeds):
        """If the population_seeds are specified, use them as the starting population."""
        if population_seeds:
            if not isinstance(population_seeds, list):
                config = self._read_config_file(population_seeds)
                if hasattr(config, 'population_seeds'):
                    pop_seeds = config.population_seeds
                else:
                    raise ValueError(
                                    'Could not find "population_seeds" in configuration file {}. '
                                    'When using a custom config file for customizing the seeds, '
                                    'the file must have a list of strings with the standardized '
                                    'name of "population_seeds"'.format(population_seeds)
                                    )
            else:
                pop_seeds = population_seeds

            self._pop = [creator.Individual.from_string(x, self._pset) for x in pop_seeds]


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
            self._imputed = True
            features = self._impute_values(features)
        else:
            self._imputed = False

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
        self._last_pipeline_write = self._start_datetime
        self._toolbox.register('evaluate', self._evaluate_individuals, features=features, target=target, sample_weight=sample_weight, groups=groups)

        # assign population. self._pop maybe be non-empty if the population is
        # seeded or a warm-start is being performed.
        n_left_to_generate = self.population_size - len(self._pop)
        if n_left_to_generate > 0:
            pop = self._pop + self._toolbox.population(n=n_left_to_generate)

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
                    per_generation_function=self._check_periodic_pipeline
                )

            # store population for the next call
            if self.warm_start:
                self._pop = pop

        # Allow for certain exceptions to signal a premature fit() cancellation
        except (KeyboardInterrupt, SystemExit, StopIteration) as e:
            if self.verbosity > 0:
                self._pbar.write('', file=self._file)
                self._pbar.write('{}\nTPOT closed prematurely. Will use the current best pipeline.'.format(e),
                                    file=self._file)
        finally:
            # keep trying 10 times in case weird things happened like multiple CTRL+C or exceptions
            attempts = 10
            for attempt in range(attempts):
                try:
                    # Close the progress bar
                    # Standard truthiness checks won't work for tqdm
                    if not isinstance(self._pbar, type(None)):
                        self._pbar.close()

                    self._update_top_pipeline()
                    self._summary_of_best_pipeline(features, target)
                    break

                except (KeyboardInterrupt, SystemExit, Exception) as e:
                    # raise the exception if it's our last attempt
                    if attempt == (attempts - 1):
                        raise e
            return self


    def _update_top_pipeline(self):
        """Helper function to update the _optimized_pipeline field.
        """
        # Store the pipeline with the highest internal testing score
        if self._pareto_front:
            self._optimized_pipeline_score = -float('inf')
            for pipeline, pipeline_scores in zip(self._pareto_front.items, reversed(self._pareto_front.keys)):
                if pipeline_scores.wvalues[1] > self._optimized_pipeline_score:
                    self._optimized_pipeline = pipeline
                    self._optimized_pipeline_score = pipeline_scores.wvalues[1]

            if not self._optimized_pipeline:
                raise RuntimeError('There was an error in the TPOT optimization '
                                  'process. This could be because the data was '
                                  'not formatted properly, or because data for '
                                  'a regression problem was provided to the '
                                  'TPOTClassifier object. Please make sure you '
                                  'passed the data to TPOT correctly.')
            else:
                pareto_front_wvalues = [pipeline_scores.wvalues[1] for pipeline_scores in self._pareto_front.keys]
                if not self._last_optimized_pareto_front:
                    self._last_optimized_pareto_front = pareto_front_wvalues
                elif self._last_optimized_pareto_front == pareto_front_wvalues:
                    self._last_optimized_pareto_front_n_gens += 1
                else:
                    self._last_optimized_pareto_front = pareto_front_wvalues
                    self._last_optimized_pareto_front_n_gens = 0
        else:
            # If user passes CTRL+C in initial generation, self._pareto_front (halloffame) shoule be not updated yet.
            # need raise RuntimeError because no pipeline has been optimized
            raise RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')


    def _summary_of_best_pipeline(self, features, target):
        """Print out best pipeline at the end of optimization process.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix

        target: array-like {n_samples}
            List of class labels for prediction

        Returns
        -------
        self: object
            Returns a copy of the fitted TPOT object
        """
        if not self._optimized_pipeline:
            raise RuntimeError('There was an error in the TPOT optimization '
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

                optimized_pipeline_str = self.clean_pipeline_string(self._optimized_pipeline)
                print('Best pipeline:', optimized_pipeline_str)

            # Store and fit the entire Pareto front as fitted models for convenience
            self.pareto_front_fitted_pipelines_ = {}

            for pipeline in self._pareto_front.items:
                self.pareto_front_fitted_pipelines_[str(pipeline)] = self._toolbox.compile(expr=pipeline)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.pareto_front_fitted_pipelines_[str(pipeline)].fit(features, target)


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
            self._imputed = True
            features = self._impute_values(features)
        else:
            self._imputed = False

        return self.fitted_pipeline_.predict(features)


    def fit_predict(self, features, target, sample_weight=None, groups=None):
        """Call fit and predict in sequence.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        target: array-like {n_samples}
            List of class labels for prediction
        sample_weight: array-like {n_samples}, optional
            Per-sample weights. Higher weights force TPOT to put more emphasis on those points
        groups: array-like, with shape {n_samples, }, optional
            Group labels for the samples used when performing cross-validation.
            This parameter should only be used in conjunction with sklearn's Group cross-validation
            functions, such as sklearn.model_selection.GroupKFold

        Returns
        ----------
        array-like: {n_samples}
            Predicted target for the provided features

        """
        self.fit(features, target, sample_weight=sample_weight, groups=groups)
        return self.predict(features)


    def score(self, testing_features, testing_target):
        """Return the score on the given testing data using the user-specified scoring function.

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


    def clean_pipeline_string(self, individual):
        """Provide a string of the individual without the parameter prefixes.

        Parameters
        ----------
        individual: individual
            Individual which should be represented by a pretty string

        Returns
        -------
        A string like str(individual), but with parameter prefixes removed.

        """
        dirty_string = str(individual)
        # There are many parameter prefixes in the pipeline strings, used solely for
        # making the terminal name unique, eg. LinearSVC__.
        parameter_prefixes = [(m.start(), m.end()) for m in re.finditer(', [\w]+__', dirty_string)]
        # We handle them in reverse so we do not mess up indices
        pretty = dirty_string
        for (start, end) in reversed(parameter_prefixes):
            pretty = pretty[:start+2] + pretty[end:]

        return pretty


    def _check_periodic_pipeline(self):
        """If enough time has passed, save a new optimized pipeline.

        Currently used in the per generation hook in the optimization loop.
        """
        self._update_top_pipeline()
        if self.periodic_checkpoint_folder is not None:
            total_since_last_pipeline_save = (datetime.now() - self._last_pipeline_write).total_seconds()
            if total_since_last_pipeline_save > self._output_best_pipeline_period_seconds:
                self._last_pipeline_write = datetime.now()
                self._save_periodic_pipeline()

        if self.early_stop is not None:
            if self._last_optimized_pareto_front_n_gens >= self.early_stop:
                raise StopIteration("The optimized pipeline was not improved after evaluating {} more generations. "
                                        "Will end the optimization process.\n".format(self.early_stop))


    def _save_periodic_pipeline(self):
        try:
            filename = os.path.join(self.periodic_checkpoint_folder, 'pipeline_{}.py'.format(datetime.now().strftime('%Y.%m.%d_%H-%M-%S')))
            did_export = self.export(filename, skip_if_repeated=True)
            if not did_export:
                self._update_pbar(pbar_num=0, pbar_msg='Periodic pipeline was not saved, probably saved before...')
            else:
                self._update_pbar(pbar_num=0, pbar_msg='Saving best periodic pipeline to {}'.format(filename))
        except Exception as e:
            self._update_pbar(pbar_num=0, pbar_msg='Failed saving periodic pipeline, exception:\n{}'.format(str(e)[:250]))


    def export(self, output_file_name, skip_if_repeated=False):
        """Export the optimized pipeline as Python code.

        Parameters
        ----------
        output_file_name: string
            String containing the path and file name of the desired output file
        skip_if_repeated: boolean
            If True, skip the actual writing if a pipeline
            code would be identical to the last pipeline exported

        Returns
        -------
        False if it skipped writing the pipeline to file
        True if the pipeline was actually written

        """
        if self._optimized_pipeline is None:
            raise RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')

        to_write = export_pipeline(self._optimized_pipeline, self.operators, self._pset, self._imputed, self._optimized_pipeline_score)

        # dont export a pipeline you just had
        if skip_if_repeated and (self._exported_pipeline_text == to_write):
            return False

        with open(output_file_name, 'w') as output_file:
            output_file.write(to_write)
            self._exported_pipeline_text = to_write

        return True


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
            self._fitted_imputer = Imputer(strategy="median", axis=0)
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
            if hasattr(obj, 'estimator'): # nested estimator
                est = getattr(obj, 'estimator')
                if hasattr(est, parameter):
                    setattr(est, parameter, value)
            if hasattr(obj, parameter):
                setattr(obj, parameter, value)


    def _stop_by_max_time_mins(self):
        """Stop optimization process once maximum minutes have elapsed."""
        if self.max_time_mins:
            total_mins_elapsed = (datetime.now() - self._start_datetime).total_seconds() / 60.
            if total_mins_elapsed >= self.max_time_mins:
                raise KeyboardInterrupt('{} minutes have elapsed. TPOT will close down.'.format(total_mins_elapsed))


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

        operator_counts, eval_individuals_str, sklearn_pipeline_list = self._preprocess_individuals(individuals)

        # Make the partial function that will be called below
        partial_wrapped_cross_val_score = partial(
            _wrapped_cross_val_score,
            features=features,
            target=target,
            cv=self.cv,
            scoring_function=self.scoring_function,
            sample_weight=sample_weight,
            groups=groups,
            timeout=self.max_eval_time_seconds
        )

        result_score_list = []
        # Don't use parallelization if n_jobs==1
        if self.n_jobs == 1:
            for sklearn_pipeline in sklearn_pipeline_list:
                self._stop_by_max_time_mins()
                val = partial_wrapped_cross_val_score(sklearn_pipeline=sklearn_pipeline)
                result_score_list = self._update_val(val, result_score_list)
        else:
            # chunk size for pbar update
            for chunk_idx in range(0, len(sklearn_pipeline_list), self.n_jobs * 4):
                self._stop_by_max_time_mins()
                parallel = Parallel(n_jobs=self.n_jobs, verbose=0, pre_dispatch='2*n_jobs')
                tmp_result_scores = parallel(delayed(partial_wrapped_cross_val_score)(sklearn_pipeline=sklearn_pipeline)
                                             for sklearn_pipeline in sklearn_pipeline_list[chunk_idx:chunk_idx + self.n_jobs * 4])
                # update pbar
                for val in tmp_result_scores:
                    result_score_list = self._update_val(val, result_score_list)

        self._update_evaluated_individuals_(result_score_list, eval_individuals_str, operator_counts)

        return [self.evaluated_individuals_[str(individual)] for individual in individuals]


    def _preprocess_individuals(self, individuals):
        """Preprocess DEAP individuals before pipeline evaluation.

        Parameters
        ----------
        individuals: a list of DEAP individual
            One individual is a list of pipeline operators and model parameters that can be
            compiled by DEAP into a callable function

        Returns
        -------
        operator_counts: dictionary
            a dictionary of operator counts in individuals for evaluation
        eval_individuals_str: list
            a list of string of individuals for evaluation
        sklearn_pipeline_list: list
            a list of scikit-learn pipelines converted from DEAP individuals for evaluation
        """
        # update self._pbar.total
        if not (self.max_time_mins is None) and not self._pbar.disable and self._pbar.total <= self._pbar.n:
            self._pbar.total += self.offspring_size
        # Check we do not evaluate twice the same individual in one pass.
        _, unique_individual_indices = np.unique([str(ind) for ind in individuals], return_index=True)
        unique_individuals = [ind for i, ind in enumerate(individuals) if i in unique_individual_indices]
        # update number of duplicate pipelines
        self._update_pbar(pbar_num=len(individuals)-len(unique_individuals))

        # a dictionary for storing operator counts
        operator_counts = {}
        # 2 lists of DEAP individuals' string, their sklearn pipelines for parallel computing
        eval_individuals_str = []
        sklearn_pipeline_list = []

        for individual in unique_individuals:
            # Disallow certain combinations of operators because they will take too long or take up too much RAM
            # This is a fairly hacky way to prevent TPOT from getting stuck on bad pipelines and should be improved in a future release
            individual_str = str(individual)
            sklearn_pipeline_str = generate_pipeline_code(expr_to_tree(individual, self._pset), self.operators)
            if sklearn_pipeline_str.count('PolynomialFeatures') > 1:
                self.evaluated_individuals_[individual_str] = (5000., -float('inf'))
                self._update_pbar(pbar_msg='Invalid pipeline encountered. Skipping its evaluation.')
            # Check if the individual was evaluated before
            elif individual_str in self.evaluated_individuals_:
                self._update_pbar(pbar_msg=('Pipeline encountered that has previously been evaluated during the '
                                 'optimization process. Using the score from the previous evaluation.'))
            else:
                try:
                    # Transform the tree expression into an sklearn pipeline
                    sklearn_pipeline = self._toolbox.compile(expr=individual)

                    # Fix random state when the operator allows
                    self._set_param_recursive(sklearn_pipeline.steps, 'random_state', 42)
                    # Setting the seed is needed for XGBoost support because XGBoost currently stores
                    # both a seed and random_state, and they're not synced correctly.
                    # XGBoost will raise an exception if random_state != seed.
                    if 'XGB' in sklearn_pipeline_str:
                        self._set_param_recursive(sklearn_pipeline.steps, 'seed', 42)

                    # Count the number of pipeline operators as a measure of pipeline complexity
                    operator_count = self._operator_count(individual)
                    operator_counts[individual_str] = max(1, operator_count)
                except Exception:
                    self.evaluated_individuals_[individual_str] = (5000., -float('inf'))
                    self._update_pbar()
                    continue
                eval_individuals_str.append(individual_str)
                sklearn_pipeline_list.append(sklearn_pipeline)

        return operator_counts, eval_individuals_str, sklearn_pipeline_list


    def _update_evaluated_individuals_(self, result_score_list, eval_individuals_str, operator_counts):
        """Update self.evaluated_individuals_ and error message during pipeline evaluation.

        Parameters
        ----------
        result_score_list: list
            A list of CV scores for evaluated pipelines
        eval_individuals_str: list
            A list of strings for evaluated pipelines
        operator_counts: list
            A list of operator counts for evaluated pipelines

        Returns
        -------
        None
        """
        for result_score, individual_str in zip(result_score_list, eval_individuals_str):
            if type(result_score) in [float, np.float64, np.float32]:
                self.evaluated_individuals_[individual_str] = (operator_counts[individual_str], result_score)
            else:
                raise ValueError('Scoring function does not return a float.')


    def _update_pbar(self, pbar_num=1, pbar_msg=None):
        """Update self._pbar and error message during pipeline evaluation.

        Parameters
        ----------
        pbar_num: int
            How many pipelines has been processed
        pbar_msg: None or string
            Error message

        Returns
        -------
        None
        """
        if not isinstance(self._pbar, type(None)):
            if self.verbosity > 2 and pbar_msg is not None:
                self._pbar.write(pbar_msg, file=self._file)
            if not self._pbar.disable:
                self._pbar.update(pbar_num)


    @_pre_test
    def _mate_operator(self, ind1, ind2):
        for _ in range(self._max_mut_loops):
            ind1_copy, ind2_copy = self._toolbox.clone(ind1),self._toolbox.clone(ind2)
            offspring, offspring2 = cxOnePoint(ind1_copy, ind2_copy)
            if str(offspring) not in self.evaluated_individuals_:
                # We only use the first offspring, so we do not care to check uniqueness of the second.
                break

        return offspring, offspring2


    @_pre_test
    def _random_mutation_operator(self, individual, allow_shrink=True):
        """Perform a replacement, insertion, or shrink mutation on an individual.

        Parameters
        ----------
        individual: DEAP individual
            A list of pipeline operators and model parameters that can be
            compiled by DEAP into a callable function

        allow_shrink: bool (True)
            If True the `mutShrink` operator, which randomly shrinks the pipeline,
            is allowed to be chosen as one of the random mutation operators.
            If False, `mutShrink`  will never be chosen as a mutation operator.

        Returns
        -------
        mut_ind: DEAP individual
            Returns the individual with one of the mutations applied to it

        """
        mutation_techniques = [
            partial(gp.mutInsert, pset=self._pset),
            partial(mutNodeReplacement, pset=self._pset)
        ]

        # We can't shrink pipelines with only one primitive, so we only add it if we find more primitives.
        number_of_primitives = sum([isinstance(node, deap.gp.Primitive) for node in individual])
        if number_of_primitives > 1 and allow_shrink:
            mutation_techniques.append(partial(gp.mutShrink))

        mutator = np.random.choice(mutation_techniques)

        unsuccesful_mutations = 0
        for _ in range(self._max_mut_loops):
            # We have to clone the individual because mutator operators work in-place.
            ind = self._toolbox.clone(individual)
            offspring, = mutator(ind)
            if str(offspring) not in self.evaluated_individuals_:
                break
            else:
                unsuccesful_mutations += 1

        # Sometimes you have pipelines for which every shrunk version has already been explored too.
        # To still mutate the individual, one of the two other mutators should be applied instead.
        if ((unsuccesful_mutations == 50) and
           (type(mutator) is partial and mutator.func is gp.mutShrink)):
            offspring, = self._random_mutation_operator(individual, allow_shrink=False)

        return offspring,


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


    def _operator_count(self, individual):
        """Count the number of pipeline operators as a measure of pipeline complexity.

        Parameters
        ----------
        individual: list
            A grown tree with leaves at possibly different depths
            dependending on the condition function.

        Returns
        -------
        operator_count: int
            How many operators in a pipeline
        """
        operator_count = 0
        for i in range(len(individual)):
            node = individual[i]
            if type(node) is deap.gp.Primitive and node.name != 'CombineDFs':
                operator_count += 1
        return operator_count


    def _update_val(self, val, result_score_list):
        """Update values in the list of result scores and self._pbar during pipeline evaluation.

        Parameters
        ----------
        val: float or "Timeout"
            CV scores
        result_score_list: list
            A list of CV scores

        Returns
        -------
        result_score_list: list
            A updated list of CV scores
        """
        self._update_pbar()
        if val == 'Timeout':
            self._update_pbar(pbar_msg=('Skipped pipeline #{0} due to time out. '
                             'Continuing to the next pipeline.'.format(self._pbar.n)))
            result_score_list.append(-float('inf'))
        else:
            result_score_list.append(val)
        return result_score_list

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
