# -*- coding: utf-8 -*-

"""
Copyright 2015-Present Randal S. Olson

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
import time
from functools import partial
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import deap
from deap import base, creator, tools, gp
from tqdm import tqdm
from copy import copy

from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics.scorer import make_scorer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from update_checker import update_check

from ._version import __version__
from .operator_utils import TPOTOperatorClassFactory, Operator, ARGType
from .export_utils import export_pipeline, expr_to_tree, generate_pipeline_code
from .decorators import _pre_test
from .built_in_operators import CombineDFs
from .config_classifier_light import classifier_config_dict_light
from .config_regressor_light import regressor_config_dict_light
from .config_classifier_mdr import tpot_mdr_classifier_config_dict

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
        if dwCtrlType == 0: # CTRL_C_EVENT
            hook_sigint()
            return 1 # don't chain to the next handler
        return 0
    win32api.SetConsoleCtrlHandler(handler, 1)


class TPOTBase(BaseEstimator):
    """TPOT automatically creates and optimizes machine learning pipelines using genetic programming"""

    def __init__(self, generations=100, population_size=100, offspring_size=None,
                 mutation_rate=0.9, crossover_rate=0.1,
                 scoring=None, cv=5, n_jobs=1,
                 max_time_mins=None, max_eval_time_mins=5,
                 random_state=None, config_dict=None, warm_start=False,
                 verbosity=0, disable_update_check=False):
        """Sets up the genetic programming algorithm for pipeline optimization.

        Parameters
        ----------
        generations: int (default: 100)
            Number of iterations to the run pipeline optimization process.
            Generally, TPOT will work better when you give it more generations (and
            therefore time) to optimize the pipeline. TPOT will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.
        population_size: int (default: 100)
            Number of individuals to retain in the GP population every generation.
            Generally, TPOT will work better when you give it more individuals
            (and therefore time) to optimize the pipeline. TPOT will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.
        offspring_size: int (default: None)
            Number of offspring to produce in each GP generation.
            By default, offspring_size = population_size.
        mutation_rate: float (default: 0.9)
            Mutation rate for the genetic programming algorithm in the range [0.0, 1.0].
            This parameter tells the GP algorithm how many pipelines to apply random
            changes to every generation. We recommend using the default parameter unless
            you understand how the mutation rate affects GP algorithms.
        crossover_rate: float (default: 0.1)
            Crossover rate for the genetic programming algorithm in the range [0.0, 1.0].
            This parameter tells the genetic programming algorithm how many pipelines to
            "breed" every generation. We recommend using the default parameter unless you
            understand how the mutation rate affects GP algorithms.
        scoring: function or str
            Function used to evaluate the quality of a given pipeline for the
            problem. By default, accuracy is used for classification problems and
            mean squared error (mse) for regression problems.
            TPOT assumes that this scoring function should be maximized, i.e.,
            higher is better.

            Offers the same options as sklearn.model_selection.cross_val_score as well as
            a built-in score "balanced_accuracy":

            ['accuracy', 'adjusted_rand_score', 'average_precision', 'balanced accuracy',
            'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
            'precision', 'precision_macro', 'precision_micro', 'precision_samples',
            'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro',
            'recall_samples', 'recall_weighted', 'roc_auc']
        cv: int (default: 5)
            Number of folds to evaluate each pipeline over in k-fold cross-validation
            during the TPOT optimization process.
        n_jobs: int (default: 1)
            Number of CPUs for evaluating pipelines in parallel during the TPOT
            optimization process. Assigning this to -1 will use as many cores as available
            on the computer.
        max_time_mins: int (default: None)
            How many minutes TPOT has to optimize the pipeline.
            If provided, this setting will override the "generations" parameter and allow
            TPOT to run until it runs out of time.
        max_eval_time_mins: int (default: 5)
            How many minutes TPOT has to optimize a single pipeline.
            Setting this parameter to higher values will allow TPOT to explore more
            complex pipelines, but will also allow TPOT to run longer.
        random_state: int (default: None)
            Random number generator seed for TPOT. Use this to make sure
            that TPOT will give you the same results each time you run it
            against the same data set with that seed.
        config_dict: Python dictionary or string (default: None)
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
        warm_start: bool (default: False)
            Flag indicating whether the TPOT instance will reuse the population from
            previous calls to fit().
        verbosity: int (default: 0)
            How much information TPOT communicates while it's running.
            0 = none, 1 = minimal, 2 = high, 3 = all.
            A setting of 2 or higher will add a progress bar during the optimization procedure.
        disable_update_check: bool (default: False)
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
        self._fitted_pipeline = None
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
                    raise TypeError('The TPOT MDR operator configuration file does not currently '
                    'work with TPOTRegressor. Please use TPOTClassifier instead.')
            else:
                try:
                    with open(config_dict, 'r') as input_file:
                        file_string = input_file.read()
                    self.config_dict = eval(file_string[file_string.find('{'):(file_string.rfind('}') + 1)])
                except:
                    raise TypeError('The operator configuration file is in a bad format or not available. '
                                    'Please check the configuration file before running TPOT.')
        else:
            self.config_dict = self.default_config_dict

        self.operators = []
        self.arguments = []
        for key in sorted(self.config_dict.keys()):
            op_class, arg_types = TPOTOperatorClassFactory(key, self.config_dict[key],
            BaseClass=Operator, ArgBaseClass=ARGType)
            if op_class:
                self.operators.append(op_class)
                self.arguments += arg_types

        # Schedule TPOT to run for many generations if the user specifies a run-time limit
        # TPOT will automatically interrupt itself when the timer runs out
        if not (max_time_mins is None):
            self.generations = 1000000

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        if self.mutation_rate + self.crossover_rate > 1:
            raise ValueError('The sum of the crossover and mutation probabilities must be <= 1.0.')

        self.verbosity = verbosity
        self.operators_context = {
            'make_pipeline': make_pipeline,
            'make_union': make_union,
            'VotingClassifier': VotingClassifier,
            'FunctionTransformer': FunctionTransformer,
            'copy': copy
        }



        self._pbar = None

        # Dictionary of individuals that have already been evaluated in previous generations
        self._evaluated_individuals = {}

        self.random_state = random_state

        # If the user passed a custom scoring function, store it in the sklearn SCORERS dictionary
        if scoring:
            if hasattr(scoring, '__call__'):
                scoring_name = scoring.__name__

                if 'loss' in scoring_name or 'error' in scoring_name:
                    greater_is_better = False
                else:
                    greater_is_better = True

                SCORERS[scoring_name] = make_scorer(scoring, greater_is_better=greater_is_better)
                self.scoring_function = scoring_name
            else:
                if scoring not in SCORERS:
                    raise ValueError('The scoring function {} is not available. '
                                     'Please choose a valid scoring function from the TPOT '
                                     'documentation.'.format(scoring))
                self.scoring_function = scoring

        self.cv = cv
        # If the OS is windows, reset cpu number to 1 since the OS did not have multiprocessing module
        if sys.platform.startswith('win') and n_jobs != 1:
            print('Warning: Although parallelization is currently supported in TPOT for Windows, '
                  'pressing Ctrl+C will freeze the optimization process without saving the best pipeline!'
                  'Thus, Please DO NOT press Ctrl+C during the optimization procss if n_jobs is not equal to 1.'
                  'For quick test in Windows, please set n_jobs to 1 for saving the best pipeline '
                  'in the middle of the optimization process via Ctrl+C.')
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self._setup_pset()
        self._setup_toolbox()

    def _setup_pset(self):
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        self._pset = gp.PrimitiveSetTyped('MAIN', [np.ndarray], Output_Array)

        # Rename pipeline input to "input_df"
        self._pset.renameArguments(ARG0='input_matrix')


        # Add all operators to the primitive set
        for op in self.operators:

            if op.root:
                # We need to add rooted primitives twice so that they can
                # return both an Output_Array (and thus be the root of the tree),
                # and return a np.ndarray so they can exist elsewhere in the tree.
                p_types = (op.parameter_types()[0], Output_Array)
                self._pset.addPrimitive(op, *p_types)

            self._pset.addPrimitive(op, *op.parameter_types())

            # Import required modules into local namespace so that pipelines
            # may be evaluated directly
            for key in sorted(op.import_hash.keys()):
                module_list = ', '.join(sorted(op.import_hash[key]))

                if key.startswith('tpot.'):
                    exec('from {} import {}'.format(key[4:], module_list))
                else:
                    exec('from {} import {}'.format(key, module_list))

                for var in op.import_hash[key]:
                    self.operators_context[var] = eval(var)

        self._pset.addPrimitive(CombineDFs(), [np.ndarray, np.ndarray], np.ndarray)

        # Terminals
        for _type in self.arguments:
            type_values = list(_type.values)
            if 'nthread' not in _type.__name__:
                type_values += ['DEFAULT']

            for val in type_values:
                terminal_name = _type.__name__ + "=" + str(val)
                self._pset.addTerminal(val, _type, name=terminal_name)

        if self.verbosity > 2:
            print('{} operators have been imported by TPOT.'.format(len(self.operators)))


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

    def fit(self, features, classes, sample_weight=None):
        """Fits a machine learning pipeline that maximizes classification score
        on the provided data

        Uses genetic programming to optimize a machine learning pipeline that
        maximizes classification score on the provided features and classes.
        Performs an internal stratified training/testing cross-validaton split
        to avoid overfitting on the provided data.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        classes: array-like {n_samples}
            List of class labels for prediction
        sample_weight: array-like {n_samples}
            List of sample weights

        Returns
        -------
        None

        """
        features = features.astype(np.float64)

        # Check that the input data is formatted correctly for scikit-learn
        if self.classification:
            clf = DecisionTreeClassifier(max_depth=5)
        else:
            clf = DecisionTreeRegressor(max_depth=5)

        try:
            clf = clf.fit(features, classes)
        except:
            raise ValueError('Error: Input data is not in a valid format. '
                             'Please confirm that the input data is scikit-learn compatible. '
                             'For example, the features must be a 2-D array and target labels '
                             'must be a 1-D array.')

        # Set the seed for the GP run
        if self.random_state is not None:
            random.seed(self.random_state) # deap uses random
            np.random.seed(self.random_state)

        self._start_datetime = datetime.now()

        self._toolbox.register('evaluate', self._evaluate_individuals, features=features, classes=classes, sample_weight=sample_weight)

        # assign population, self._pop can only be not None if warm_start is enabled
        if self._pop:
            pop = self._pop
        else:
            pop = self._toolbox.population(n=self.population_size)

        def pareto_eq(ind1, ind2):
            """Determines whether two individuals are equal on the Pareto front

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
                pop, _ = eaMuPlusLambda(population=pop, toolbox=self._toolbox,
                    mu=self.population_size, lambda_=self.offspring_size,
                    cxpb=self.crossover_rate, mutpb=self.mutation_rate,
                    ngen=self.generations, pbar=self._pbar, halloffame=self._pareto_front,
                    verbose=self.verbosity, max_time_mins=self.max_time_mins)

            # store population for the next call
            if self.warm_start:
                self._pop = pop

        # Allow for certain exceptions to signal a premature fit() cancellation
        except (KeyboardInterrupt, SystemExit):
            if self.verbosity > 0:
                self._pbar.write('') # just for better interface
                self._pbar.write('TPOT closed prematurely. Will use the current best pipeline.')
        finally:
            # Close the progress bar
            # Standard truthiness checks won't work for tqdm
            if not isinstance(self._pbar, type(None)):
                self._pbar.close()

            # Store the pipeline with the highest internal testing score
            if self._pareto_front:
                top_score = -float('inf')
                for pipeline, pipeline_scores in zip(self._pareto_front.items, reversed(self._pareto_front.keys)):
                    if pipeline_scores.wvalues[1] > top_score:
                        self._optimized_pipeline = pipeline
                        top_score = pipeline_scores.wvalues[1]

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
                    self._fitted_pipeline = self._toolbox.compile(expr=self._optimized_pipeline)

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        self._fitted_pipeline.fit(features, classes)

                    if self.verbosity in [1, 2]:
                        # Add an extra line of spacing if the progress bar was used
                        if self.verbosity >= 2:
                            print('')
                        print('Best pipeline: {}'.format(self._optimized_pipeline))

                    # Store and fit the entire Pareto front if sciencing
                    elif self.verbosity >= 3 and self._pareto_front:
                        self._pareto_front_fitted_pipelines = {}

                        for pipeline in self._pareto_front.items:
                            self._pareto_front_fitted_pipelines[str(pipeline)] = self._toolbox.compile(expr=pipeline)
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                self._pareto_front_fitted_pipelines[str(pipeline)].fit(features, classes)

    def predict(self, features):
        """Uses the optimized pipeline to predict the classes for a feature set

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix to predict on

        Returns
        ----------
        array-like: {n_samples}
            Predicted classes for the feature matrix

        """
        if not self._fitted_pipeline:
            raise RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')
        return self._fitted_pipeline.predict(features.astype(np.float64))

    def fit_predict(self, features, classes):
        """Convenience function that fits a pipeline then predicts on the
        provided features

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        classes: array-like {n_samples}
            List of class labels for prediction

        Returns
        ----------
        array-like: {n_samples}
            Predicted classes for the provided features

        """
        self.fit(features, classes)
        return self.predict(features)

    def score(self, testing_features, testing_classes):
        """Estimates the balanced testing accuracy of the optimized pipeline.

        Parameters
        ----------
        testing_features: array-like {n_samples, n_features}
            Feature matrix of the testing set
        testing_classes: array-like {n_samples}
            List of class labels for prediction in the testing set

        Returns
        -------
        accuracy_score: float
            The estimated test set accuracy

        """
        if self._fitted_pipeline is None:
            raise RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')

        # If the scoring function is a string, we must adjust to use the sklearn scoring interface
        return abs(SCORERS[self.scoring_function](self._fitted_pipeline,
            testing_features.astype(np.float64), testing_classes.astype(np.float64)))

    def predict_proba(self, features):
        """Uses the optimized pipeline to estimate the class probabilities for a feature set

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix of the testing set

        Returns
        -------
        array-like: {n_samples, n_classes}
            The class probabilities of the input samples

        """
        if not self._fitted_pipeline:
            raise RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')
        else:
            if not(hasattr(self._fitted_pipeline, 'predict_proba')):
                raise RuntimeError('The fitted pipeline does not have the predict_proba() function.')
            return self._fitted_pipeline.predict_proba(features.astype(np.float64))

    def set_params(self, **params):
        """Set the parameters of a TPOT instance

        Returns
        -------
        self
        """
        self.__init__(**params)

        return self

    def export(self, output_file_name):
        """Exports the current optimized pipeline as Python code

        Parameters
        ----------
        output_file_name: str
            String containing the path and file name of the desired output file

        Returns
        -------
        None

        """
        if self._optimized_pipeline is None:
            raise RuntimeError('A pipeline has not yet been optimized. Please call fit() first.')

        with open(output_file_name, 'w') as output_file:
            output_file.write(export_pipeline(self._optimized_pipeline, self.operators, self._pset))

    def _compile_to_sklearn(self, expr):
        """Compiles a DEAP pipeline into a sklearn pipeline

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
        """Recursively iterates through all objects in the pipeline and sets the given parameter to the specified value

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

    def _evaluate_individuals(self, individuals, features, classes, sample_weight = None):
        """Determines the `individual`'s fitness

        Parameters
        ----------
        individuals: a list of DEAP individual
            One individual is a list of pipeline operators and model parameters that can be
            compiled by DEAP into a callable function
        features: numpy.ndarray {n_samples, n_features}
            A numpy matrix containing the training and testing features for the
            `individual`'s evaluation
        classes: numpy.ndarray {n_samples, }
            A numpy matrix containing the training and testing classes for the
            `individual`'s evaluation

        Returns
        -------
        fitnesses_ordered: float
            Returns a list of tuple value indicating the `individual`'s fitness
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
            if individual_str.count('PolynomialFeatures') > 1:
                if self.verbosity > 2:
                    self._pbar.write('Invalid pipeline encountered. Skipping its evaluation.')
                fitnesses_dict[indidx] = (5000., -float('inf'))
                if not self._pbar.disable:
                    self._pbar.update(1)

            # Check if the individual was evaluated before
            elif individual_str in self._evaluated_individuals:
                # Get fitness score from previous evaluation
                fitnesses_dict[indidx] = self._evaluated_individuals[individual_str]
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
                    operator_count = 0
                    for i in range(len(individual)):
                        node = individual[i]
                        if ((type(node) is deap.gp.Terminal) or
                             type(node) is deap.gp.Primitive and node.name == 'CombineDFs'):
                            continue
                        operator_count += 1
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
        for chunk_idx in range(0, len(sklearn_pipeline_list),self.n_jobs*4):
            parallel = Parallel(n_jobs=self.n_jobs, verbose=0, pre_dispatch='2*n_jobs')
            tmp_result_score = parallel(delayed(_wrapped_cross_val_score)(sklearn_pipeline, features, classes,
                                         self.cv, self.scoring_function, sample_weight, self.max_eval_time_mins)
                      for sklearn_pipeline in sklearn_pipeline_list[chunk_idx:chunk_idx+self.n_jobs*4])
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
                self._evaluated_individuals[individual_str] = (max(1, operator_count), resulting_score)
                fitnesses_dict[test_idx] = self._evaluated_individuals[individual_str]
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
        """Perform a replacement, insertion, or shrink mutation on an individual

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
        """Generate an expression where each leaf might have a different depth
        between min_ and max_.

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
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
        Returns
        -------
        individual: list
            A grown tree with leaves at possibly different depths.
        """
        def condition(height, depth, type_):
            """Expression generation stops when the depth is equal to height or
            when it is randomly determined that a a node should be a terminal"""
            return type_ not in [np.ndarray, Output_Array] or depth == height

        return self._generate(pset, min_, max_, condition, type_)

    # Generate function stolen straight from deap.gp.generate
    @_pre_test
    def _generate(self, pset, min_, max_, condition, type_=None):
        """Generate a Tree as a list of list. The tree is build from the root to
        the leaves, and it stop growing when the condition is fulfilled.

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
            :obj:`None` (default) no return type is enforced.

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
                    raise IndexError("The gp.generate function tried to add "
                                      "a terminal of type '%s', but there is "
                                      "none available." % (type_,)).\
                                      with_traceback(traceback)
                if inspect.isclass(term):
                    term = term()
                expr.append(term)
            else:
                try:
                    prim = np.random.choice(pset.primitives[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add "
                                      "a primitive of type '%s', but there is "
                                      "none available." % (type_,)).\
                                      with_traceback(traceback)
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth+1, arg))

        return expr
