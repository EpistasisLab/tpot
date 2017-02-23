# -*- coding: utf-8 -*-

"""
Copyright 2016 Randal S. Olson

This file is part of the TPOT library.

The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along
with the TPOT library. If not, see http://www.gnu.org/licenses/.

"""

from __future__ import print_function
import random
import inspect
import warnings
import sys
import time
from functools import partial
from datetime import datetime
from pathos.multiprocessing import ProcessPool

import numpy as np
import deap
from deap import algorithms, base, creator, tools, gp
from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics.scorer import make_scorer

from update_checker import update_check

from ._version import __version__
from .operator_utils import TPOTOperatorClassFactory, Operator, ARGType
from .export_utils import export_pipeline, expr_to_tree, generate_pipeline_code

from .decorators import _timeout, _pre_test, TimedOutExc
from .build_in_operators import CombineDFs


from .gp_types import Bool, Output_DF
from .metrics import SCORERS
from .gp_deap import eaMuPlusLambda, mutNodeReplacement







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
# add time limit for imported function
#cross_val_score = _timeout(cross_val_score)


class TPOTBase(BaseEstimator):
    """TPOT automatically creates and optimizes machine learning pipelines using genetic programming"""

    def __init__(self, population_size=100, generations=100, offspring_size=None,
                 mutation_rate=0.9, crossover_rate=0.05,
                 scoring=None, cv=5, n_jobs=1,
                 max_time_mins=None, max_eval_time_mins=5,
                 random_state=None, operator_dict=None, verbosity=0,
                 disable_update_check=False, warm_start=False):
        """Sets up the genetic programming algorithm for pipeline optimization.

        Parameters
        ----------
        population_size: int (default: 100)
            The number of pipelines in the genetic algorithm population. Must
            be > 0.The more pipelines in the population, the slower TPOT will
            run, but it's also more likely to find better pipelines.
        offspring_size: int (default: None)
            The number of children to produce at each generation.
        generations: int (default: 100)
            The number of generations to run pipeline optimization for. Must
            be > 0. The more generations you give TPOT to run, the longer it
            takes, but it's also more likely to find better pipelines.
        mutation_rate: float (default: 0.9)
            The mutation rate for the genetic programming algorithm in the range
            [0.0, 1.0]. This tells the genetic programming algorithm how many
            pipelines to apply random changes to every generation. We don't
            recommend that you tweak this parameter unless you know what you're
            doing.
        crossover_rate: float (default: 0.05)
            The crossover rate for the genetic programming algorithm in the
            range [0.0, 1.0]. This tells the genetic programming algorithm how
            many pipelines to "breed" every generation. We don't recommend that
            you tweak this parameter unless you know what you're doing.
        scoring: function or str
            Function used to evaluate the quality of a given pipeline for the
            problem. By default, balanced class accuracy is used for
            classification problems, mean squared error for regression problems.
            TPOT assumes that this scoring function should be maximized, i.e.,
            higher is better.

            Offers the same options as sklearn.model_selection.cross_val_score:

            ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1',
            'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
            'precision', 'precision_macro', 'precision_micro', 'precision_samples',
            'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro',
            'recall_samples', 'recall_weighted', 'roc_auc']
        cv: int (default: 5)
            The number of folds to evaluate each pipeline over in k-fold
            cross-validation during the TPOT pipeline optimization process
        n_jobs: int (default: 1)
            The number of CPUs for evaluating each pipeline over
            cross-validation during the TPOT pipeline optimization process
        max_time_mins: int (default: None)
            How many minutes TPOT has to optimize the pipeline. If not None,
            this setting will override the `generations` parameter.
        max_eval_time_mins: int (default: 5)
            How many minutes TPOT has to optimize a single pipeline.
            Setting this parameter to higher values will allow TPOT to explore more complex
            pipelines but will also allow TPOT to run longer.
        random_state: int (default: None)
            The random number generator seed for TPOT. Use this to make sure
            that TPOT will give you the same results each time you run it
            against the same data set with that seed.
        operator_dict: a customized python dictionary (default: None)
            The customized python dictionary to specify the list of operators and
            their arguments. Format examples: config_regressor.py and config_classifier.py
        verbosity: int (default: 0)
            How much information TPOT communicates while it's running.
            0 = none, 1 = minimal, 2 = all
        disable_update_check: bool (default: False)
            Flag indicating whether the TPOT version checker should be disabled.
        warm_start: bool (default: False)
            Flag indicating whether TPOT will reuse models from previous calls to
            fit() for faster operation

        Returns
        -------
        None

        """
        if self.__class__.__name__ == 'TPOTBase':
            raise RuntimeError('Do not instantiate the TPOTBase class directly; '
                               'use TPOTRegressor or TPOTClassifier instead.')

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


        # set offspring_size equal to  population_size by default
        if offspring_size:
            self.offspring_size = offspring_size
        else:
            self.offspring_size = population_size

        # define operator dictionary based on files
        if operator_dict:
            self.operator_dict = operator_dict
        else:
            self.operator_dict = self.default_operator_dict

        self.operators = []
        self.arguments = []
        for key in sorted(self.operator_dict.keys()):
            op_class, arg_types = TPOTOperatorClassFactory(key, self.operator_dict[key],
            BaseClass=Operator, ArgBaseClass=ARGType)
            if op_class:
                self.operators.append(op_class)
                self.arguments += arg_types


        # Schedule TPOT to run for a very long time if the user specifies a run-time
        # limit TPOT will automatically interrupt itself when the timer runs out
        if not (max_time_mins is None):
            self.generations = 1000000

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # check if mutation_rate + crossover_rate > 1
        if self.mutation_rate + self.crossover_rate > 1:
            raise TypeError('The sum of the crossover and mutation probabilities must be smaller '
        'or equal to 1.0.')

        self.verbosity = verbosity
        self.operators_context = {
            'make_pipeline': make_pipeline,
            'make_union': make_union,
            'VotingClassifier': VotingClassifier,
            'FunctionTransformer': FunctionTransformer
        }


        self._pbar = None

        # a dictionary of individual which has already evaluated in previous generation.
        self.eval_ind = {}

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
                self.scoring_function = scoring

        self.cv = cv
        # If the OS is windows, reset cpu number to 1 since the OS did not have multiprocessing module
        if sys.platform.startswith('win') and n_jobs > 1:
            print('Warning: Parallelizing cross validation is not supported in Windows OS.',
                'Reset number of cpu to 1 during TPOT pipeline optimization process')
            self.n_jobs = 1
        else:
            self.n_jobs = n_jobs

        self._setup_pset()
        self._setup_toolbox()

    def _setup_pset(self):

        # creating dynamically create operator class

        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)


        self._pset = gp.PrimitiveSetTyped('MAIN', [np.ndarray], Output_DF)

        # Rename pipeline input to "input_df"
        self._pset.renameArguments(ARG0='input_matrix')


        # Add all operators to the primitive set
        for op in self.operators:

            if op.root:
                # We need to add rooted primitives twice so that they can
                # return both an Output_DF (and thus be the root of the tree),
                # and return a np.ndarray so they can exist elsewhere in the tree.
                p_types = (op.parameter_types()[0], Output_DF)
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
            for val in _type.values:
                self._pset.addTerminal(val, _type)

        if self.verbosity > 2:
            print('{} operators are imported.'.format(len(self.operators)))



    def _setup_toolbox(self):
        creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self._toolbox = base.Toolbox()
        self._toolbox.register('expr', self._gen_grow_safe, pset=self._pset, min_=1, max_=3)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register('compile', self._compile_to_sklearn)
        self._toolbox.register('select', tools.selNSGA2)
        self._toolbox.register('mate', _pre_test(gp.cxOnePoint))
        self._toolbox.register('expr_mut', self._gen_grow_safe, min_=1, max_=4)
        self._toolbox.register('mutate', _pre_test(self._random_mutation_operator))

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

        # Set the seed for the GP run
        if self.random_state is not None:
            random.seed(self.random_state) # deap use random
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
        # generate new pareto front if it doesn't already exist for warm start
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
                print('') # just for better interface
                print('GP closed prematurely - will use current best pipeline')
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
            raise ValueError('A pipeline has not yet been optimized. Please call fit() first.')
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
            raise ValueError('A pipeline has not yet been optimized. '
                             'Please call fit() first.')

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
            raise ValueError('A pipeline has not yet been optimized. Please call fit() first.')
        else:
            if not(hasattr(self._fitted_pipeline, 'predict_proba')):
                raise ValueError('The fitted pipeline does not have probability prediction functionality')
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
            raise ValueError('A pipeline has not yet been optimized. Please call fit() first.')

        with open(output_file_name, 'w') as output_file:
            output_file.write(export_pipeline(self._optimized_pipeline, self.operators))

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
        sklearn_pipeline = generate_pipeline_code(expr_to_tree(expr), self.operators)
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
        for (pname, obj) in pipeline_steps:
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
        if not sample_weight:
            sample_weight_dict = None

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
            if (individual_str.count('PolynomialFeatures') > 1):
                print('Invalid pipeline -- skipping its evaluation')
                fitnesses_dict[indidx] = (5000., -float('inf'))
                if not self._pbar.disable:
                    self._pbar.update(1)

            # check if the individual are evaluated before
            elif individual_str in self.eval_ind:
                # get fitness score from previous evaluation
                fitnesses_dict[indidx] = self.eval_ind[individual_str]
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
                    # add time limit for evaluation of pipeline
                    for i in range(len(individual)):
                        node = individual[i]
                        if ((type(node) is deap.gp.Terminal) or
                             type(node) is deap.gp.Primitive and node.name == 'CombineDFs'):
                            continue
                        operator_count += 1
                except:
                    fitnesses_dict[indidx] = (5000., -float('inf'))
                    if not self._pbar.disable:
                        self._pbar.update(1)
                    continue
                eval_individuals_str.append(individual_str)
                operator_count_list.append(operator_count)
                sklearn_pipeline_list.append(sklearn_pipeline)
                test_idx_list.append(indidx)

        def _set_sample_weight(pipeline_steps, sample_weight):
            """Recursively iterates through all objects in the pipeline and sets the given parameter to the specified value

            Parameters
            ----------
            pipeline_steps: array-like
                List of (str, obj) tuples from a scikit-learn pipeline or related object
            sample_weight: array-like
                List of sample weight
            Returns
            -------
            sample_weight_dict:
                A dictionary of sample_weight

            """
            sample_weight_dict = {}
            for (pname, obj) in pipeline_steps:
                if inspect.getargspec(obj.fit).args.count('sample_weight') and sample_weight:
                    step_sw = pname + '__sample_weight'
                    sample_weight_dict[step_sw] = sample_weight
            if sample_weight_dict:
                return sample_weight_dict
            else:
                return None

        @_timeout(max_eval_time_mins=self.max_eval_time_mins)
        def _wrapped_cross_val_score(sklearn_pipeline, features=features, classes=classes,
        cv=self.cv, scoring_function=self.scoring_function,sample_weight=sample_weight):
            sample_weight_dict = _set_sample_weight(sklearn_pipeline.steps, sample_weight)
            from .decorators import TimedOutExc
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cv_scores = cross_val_score(sklearn_pipeline, features, classes,
                        cv=cv, scoring=scoring_function,
                        n_jobs=1, fit_params=sample_weight_dict)
                resulting_score = np.mean(cv_scores)
            except TimedOutExc:
                resulting_score = "Timeout"
            except:
                resulting_score = -float('inf')
            return resulting_score

        if not sys.platform.startswith('win'):
            if self.n_jobs == -1:
                pool = ProcessPool()
            else:
                pool = ProcessPool(nodes=self.n_jobs)
            res_imap = pool.imap(_wrapped_cross_val_score, sklearn_pipeline_list)
            if not self._pbar.disable:
                ini_pbar_n = self._pbar.n
            # hacky way for pbar update by using imap in pathos.multiprocessing.ProcessPool
            while not self._pbar.disable:
                tmp_fitness = np.array(res_imap._items)
                num_job_done = len(tmp_fitness)
                if not self._pbar.disable and num_job_done:
                    timeout_index = list(np.where(tmp_fitness[:,1] == "Timeout")[0])
                    for idx in timeout_index:
                        if self._pbar.n - ini_pbar_n <= idx:
                            self._pbar.write("Skip pipeline #{0} due to time out. "
                            "Continuing to the next pipeline.".format(ini_pbar_n + idx + 1))
                    self._pbar.update(ini_pbar_n + num_job_done - self._pbar.n)
                if num_job_done >= len(sklearn_pipeline_list):
                    break
                else:
                    time.sleep(0.2)
            resulting_score_list = [-float('inf') if x=="Timeout" else x for x in list(res_imap)]
        else:
            resulting_score_list = []
            for sklearn_pipeline in sklearn_pipeline_list:
                try:
                    resulting_score = _wrapped_cross_val_score(sklearn_pipeline)
                except TimedOutExc:
                    resulting_score = -float('inf')
                    if not self._pbar.disable:
                        self._pbar.write("Skip pipeline #{0} due to time out. "
                        "Continuing to the next pipeline.".format(self._pbar.n + 1))
                resulting_score_list.append(resulting_score)
                if not self._pbar.disable:
                    self._pbar.update(1)

        for resulting_score, operator_count, individual_str, test_idx in zip(resulting_score_list, operator_count_list, eval_individuals_str, test_idx_list):
            if type(resulting_score) in [float, np.float64, np.float32]:
                self.eval_ind[individual_str] = (max(1, operator_count), resulting_score)
                fitnesses_dict[test_idx] = self.eval_ind[individual_str]
            else:
                raise ValueError('Scoring function does not return a float')

        fitnesses_ordered = []
        for key in sorted(fitnesses_dict.keys()):
            fitnesses_ordered.append(fitnesses_dict[key])
        return fitnesses_ordered


    def _random_mutation_operator(self, individual):
        """Perform a replacement, insert, or shrink mutation on an individual

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
        # debug usage
        #print(str(individual))
        old_ind = str(individual)
        mut_ind = (str(individual),)
        mutation_techniques = [
            partial(gp.mutInsert, pset=self._pset),
            partial(mutNodeReplacement, pset=self._pset),
            partial(gp.mutShrink)
        ]
        mut_ind = np.random.choice(mutation_techniques)(individual)
        return mut_ind


    def _gen_grow_safe(self, pset, min_, max_, type_=None):
        """Generate an expression where each leaf might have a different depth
        between *min* and *max*.

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
            return type_ not in [np.ndarray, Output_DF] or depth == height

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
