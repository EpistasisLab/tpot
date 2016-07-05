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
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
the TPOT library. If not, see http://www.gnu.org/licenses/.

"""

from __future__ import print_function
import argparse
import random
import inspect
import sys
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split

from update_checker import update_check

from ._version import __version__
from .export_utils import unroll_nested_fuction_calls, generate_import_code, replace_function_calls
from .decorators import _gp_new_generation
from .operators import *
from .helpers import Bool

import deap
from deap import algorithms, base, creator, tools, gp

from tqdm import tqdm


class TPOT(object):
    """TPOT automatically creates and optimizes machine learning pipelines using genetic programming."""

    def __init__(self, population_size=100, generations=100,
                 mutation_rate=0.9, crossover_rate=0.05,
                 random_state=0, verbosity=0, scoring_function=None,
                 disable_update_check=False):
        """Sets up the genetic programming algorithm for pipeline optimization.

        Parameters
        ----------
        population_size: int (default: 100)
            The number of pipelines in the genetic algorithm population. Must be > 0.
            The more pipelines in the population, the slower TPOT will run, but it's also more likely to find better pipelines.
        generations: int (default: 100)
            The number of generations to run pipeline optimization for. Must be > 0.
            The more generations you give TPOT to run, the longer it takes, but it's also more likely to find better pipelines.
        mutation_rate: float (default: 0.9)
            The mutation rate for the genetic programming algorithm in the range [0.0, 1.0].
            This tells the genetic programming algorithm how many pipelines to apply random changes to every generation.
            We don't recommend that you tweak this parameter unless you know what you're doing.
        crossover_rate: float (default: 0.05)
            The crossover rate for the genetic programming algorithm in the range [0.0, 1.0].
            This tells the genetic programming algorithm how many pipelines to "breed" every generation.
            We don't recommend that you tweak this parameter unless you know what you're doing.
        random_state: int (default: 0)
            The random number generator seed for TPOT. Use this to make sure that TPOT will give you the same results each time
            you run it against the same data set with that seed.
        verbosity: int (default: 0)
            How much information TPOT communicates while it's running. 0 = none, 1 = minimal, 2 = all
        scoring_function: function (default: balanced accuracy)
            Function used to evaluate the goodness of a given pipeline for the classification problem. By default, balanced class accuracy is used.
        disable_update_check: bool (default: False)
            Flag indicating whether the TPOT version checker should be disabled.

        Returns
        -------
        None

        """
        # Save params to be recalled later by get_params()
        self.params = locals()  # Must be placed before any local variable definitions
        self.params.pop('self')

        # Prompt the user if their version is out of date
        if not disable_update_check:
            update_check('tpot', __version__)

        self.hof = None
        self._optimized_pipeline = None
        self._training_features = None
        self._training_classes = None
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.verbosity = verbosity

        self.pbar = None
        self.gp_generation = 0

        if random_state > 0:
            random.seed(random_state)
            np.random.seed(random_state)
            Operator.default_arguments['random_state'] = random_state

        self._setup_pset()
        self._setup_toolbox()

        if not scoring_function:
            self.scoring_function = self._balanced_accuracy
        else:
            self.scoring_function = scoring_function

    def _setup_pset(self):
        self._pset = gp.PrimitiveSetTyped('MAIN', [pd.DataFrame], pd.DataFrame)

        # Rename pipeline input to "input_df"
        self._pset.renameArguments(ARG0='input_df')

        # Add all operators to the primitive set
        for op in Operator.inheritors():
            self._pset.addPrimitive(op, *op.parameter_types())

        # Add the combine_dfs operator
        self._pset.addPrimitive(self._combine_dfs, [pd.DataFrame, pd.DataFrame], pd.DataFrame)

        # Terminals
        int_terminals = np.concatenate((np.arange(0, 51, 1),
            np.arange(60, 110, 10)))

        for val in int_terminals:
            self._pset.addTerminal(val, int)

        float_terminals = np.concatenate(([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            np.linspace(0., 1., 101),
            np.linspace(2., 50., 49),
            np.linspace(60., 100., 5)))

        for val in float_terminals:
            self._pset.addTerminal(val, float)

        self._pset.addTerminal(True, Bool)
        self._pset.addTerminal(False, Bool)

    def _setup_toolbox(self):
        creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self._toolbox = base.Toolbox()
        self._toolbox.register('expr', self._gen_grow_safe, pset=self._pset, min_=1, max_=3)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register('compile', gp.compile, pset=self._pset)
        self._toolbox.register('select', self._combined_selection_operator)
        self._toolbox.register('mate', gp.cxOnePoint)
        self._toolbox.register('expr_mut', self._gen_grow_safe, min_=0, max_=3)
        self._toolbox.register('mutate', self._random_mutation_operator)

    def fit(self, features, classes):
        """Fits a machine learning pipeline that maximizes classification accuracy on the provided data

        Uses genetic programming to optimize a machine learning pipeline that
        maximizes classification accuracy on the provided `features` and `classes`.
        Performs an internal stratified training/testing cross-validaton split to avoid
        overfitting on the provided data.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        classes: array-like {n_samples}
            List of class labels for prediction

        Returns
        -------
        None

        """
        try:
            # Store the training features and classes for later use
            self._training_features = features
            self._training_classes = classes

            training_testing_data = pd.DataFrame(data=features)
            training_testing_data['class'] = classes

            new_col_names = {}
            for column in training_testing_data.columns.values:
                if type(column) != str:
                    new_col_names[column] = str(column).zfill(10)
            training_testing_data.rename(columns=new_col_names, inplace=True)

            # Randomize the order of the columns so there is no potential bias introduced by the initial order
            # of the columns, e.g., the most predictive features at the beginning or end.
            data_columns = list(training_testing_data.columns.values)
            np.random.shuffle(data_columns)
            training_testing_data = training_testing_data[data_columns]

            training_indices, testing_indices = train_test_split(training_testing_data.index,
                                                                 stratify=training_testing_data['class'].values,
                                                                 train_size=0.75,
                                                                 test_size=0.25)

            training_testing_data.loc[training_indices, 'group'] = 'training'
            training_testing_data.loc[testing_indices, 'group'] = 'testing'

            # Default guess: the most frequent class in the training data
            most_frequent_training_class = Counter(training_testing_data.loc[training_indices, 'class'].values).most_common(1)[0][0]
            training_testing_data.loc[:, 'guess'] = most_frequent_training_class

            self._toolbox.register('evaluate', self._evaluate_individual, training_testing_data=training_testing_data)

            pop = self._toolbox.population(n=self.population_size)

            def pareto_eq(ind1, ind2):
                """Function used to determine whether two individuals are equal on the Pareto front

                Parameters
                ----------
                ind1: DEAP individual from the GP population
                    First individual to compare
                ind2: DEAP individual from the GP population
                    Second individual to compare

                Returns
                ----------
                individuals_equal: bool
                    Boolean indicating whether the two individuals are equal on the Pareto front

                """
                return np.all(ind1.fitness.values == ind2.fitness.values)

            self.hof = tools.ParetoFront(similar=pareto_eq)

            verbose = (self.verbosity == 2)

            # Start the progress bar
            num_evaluations = self.population_size * (self.generations + 1)
            self.pbar = tqdm(total=num_evaluations, unit='pipeline', leave=False,
                             disable=(not verbose), desc='GP Progress')

            pop, _ = algorithms.eaSimple(population=pop, toolbox=self._toolbox, cxpb=self.crossover_rate,
                                     mutpb=self.mutation_rate, ngen=self.generations,
                                     halloffame=self.hof, verbose=False)

        # Allow for certain exceptions to signal a premature fit() cancellation
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            # Close the progress bar
            if not isinstance(self.pbar, type(None)):  # Standard truthiness checks won't work for tqdm
                self.pbar.close()

            # Reset gp_generation counter to restore initial state
            self.gp_generation = 0

            # Store the pipeline with the highest internal testing accuracy
            if self.hof:
                top_score = 0.
                for pipeline in self.hof:
                    pipeline_score = self._evaluate_individual(pipeline, training_testing_data)[1]
                    if pipeline_score > top_score:
                        top_score = pipeline_score
                        self._optimized_pipeline = pipeline

            if self.verbosity >= 1 and self._optimized_pipeline:
                if verbose:  # Add an extra line of spacing if the progress bar was used
                    print()

                print('Best pipeline: {}'.format(self._optimized_pipeline))

    def predict(self, testing_features):
        """Uses the optimized pipeline to predict the classes for a feature set.

        Parameters
        ----------
        testing_features: array-like {n_samples, n_features}
            Feature matrix of the testing set

        Returns
        ----------
        array-like: {n_samples}
            Predicted classes for the testing set

        """
        if self._optimized_pipeline is None:
            raise ValueError('A pipeline has not yet been optimized. Please call fit() first.')

        training_data = pd.DataFrame(self._training_features)
        training_data['class'] = self._training_classes
        training_data['group'] = 'training'

        testing_data = pd.DataFrame(testing_features)
        testing_data['class'] = 0
        testing_data['group'] = 'testing'

        training_testing_data = pd.concat([training_data, testing_data])

        # Default guess: the most frequent class in the training data
        most_frequent_training_class = Counter(self._training_classes).most_common(1)[0][0]
        training_testing_data.loc[:, 'guess'] = most_frequent_training_class

        new_col_names = {}
        for column in training_testing_data.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        training_testing_data.rename(columns=new_col_names, inplace=True)

        # Transform the tree expression in a callable function
        func = self._toolbox.compile(expr=self._optimized_pipeline)

        result = func(training_testing_data)

        return result.loc[result['group'] == 'testing', 'guess'].values

    def fit_predict(self, features, classes):
        """Convenience function that fits a pipeline then predicts on the provided features

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
        """Estimates the testing accuracy of the optimized pipeline.

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
        if self._optimized_pipeline is None:
            raise ValueError('A pipeline has not yet been optimized. Please call fit() first.')

        training_data = pd.DataFrame(self._training_features)
        training_data['class'] = self._training_classes
        training_data['group'] = 'training'

        testing_data = pd.DataFrame(testing_features)
        testing_data['class'] = testing_classes
        testing_data['group'] = 'testing'

        training_testing_data = pd.concat([training_data, testing_data])

        # Default guess: the most frequent class in the training data
        most_frequent_training_class = Counter(self._training_classes).most_common(1)[0][0]
        training_testing_data.loc[:, 'guess'] = most_frequent_training_class

        new_col_names = {}
        for column in training_testing_data.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        training_testing_data.rename(columns=new_col_names, inplace=True)

        return self._evaluate_individual(self._optimized_pipeline, training_testing_data)[1]

    def get_params(self, deep=None):
        """Get parameters for this estimator

        This function is necessary for TPOT to work as a drop-in estimator in,
        e.g., sklearn.cross_validation.cross_val_score

        Parameters
        ----------
        deep: unused
            Only implemented to maintain interface for sklearn

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """

        return self.params

    def export(self, output_file_name):
        """Exports the current optimized pipeline as Python code

        Parameters
        ----------
        output_file_name: string
            String containing the path and file name of the desired output file

        Returns
        -------
        None

        """
        if self._optimized_pipeline is None:
            raise ValueError('A pipeline has not yet been optimized. Please call fit() first.')

        exported_pipeline = self._optimized_pipeline

        # Unroll the nested function calls into serial code. Check export_utils.py for details.
        pipeline_list = unroll_nested_fuction_calls(exported_pipeline)

        # Have the exported code import all of the necessary modules and functions
        pipeline_text = generate_import_code(pipeline_list)

        # Replace the function calls with their corresponding Python code. Check export_utils.py for details.
        pipeline_text += replace_function_calls(pipeline_list)

        with open(output_file_name, 'w') as output_file:
            output_file.write(pipeline_text)

    @staticmethod
    def _combine_dfs(input_df1, input_df2):
        """Function to combine two DataFrames

        Parameters
        ----------
        input_df1: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to combine
        input_df2: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to combine

        Returns
        -------
        combined_df: pandas.DataFrame {n_samples, n_both_features+['guess', 'group', 'class']}
            Returns a DataFrame containing the features of both input_df1 and input_df2

        """
        return input_df1.join(input_df2[[column for column in input_df2.columns.values if column not in input_df1.columns.values]]).copy()

    def _evaluate_individual(self, individual, training_testing_data):
        """Determines the `individual`'s fitness according to its performance on the provided data

        Parameters
        ----------
        individual: DEAP individual
            A list of pipeline operators and model parameters that can be compiled by DEAP into a callable function
        training_testing_data: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class']}
            A DataFrame containing the training and testing data for the `individual`'s evaluation

        Returns
        -------
        fitness: float
            Returns a float value indicating the `individual`'s fitness according to its performance on the provided data

        """
        try:
            # Transform the tree expression in a callable function
            func = self._toolbox.compile(expr=individual)

            # Count the number of pipeline operators as a measure of pipeline complexity
            operator_count = 0
            for i in range(len(individual)):
                node = individual[i]
                if type(node) is deap.gp.Terminal:
                    continue
                if type(node) is deap.gp.Primitive and node.name == '_combine_dfs':
                    continue

                operator_count += 1

            result = func(training_testing_data)
            result = result[result['group'] == 'testing']
            resulting_score = self.scoring_function(result)

        except MemoryError:
            # Throw out GP expressions that are too large to be compiled in Python
            return 5000., 0.
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # Catch-all: Do not allow one pipeline that crashes to cause TPOT to crash
            # Instead, assign the crashing pipeline a poor fitness
            return 5000., 0.
        finally:
            if not self.pbar.disable:
                self.pbar.update(1)  # One more pipeline evaluated

        if isinstance(resulting_score, float) or isinstance(resulting_score, np.float64) or isinstance(resulting_score, np.float32):
            return max(1, operator_count), resulting_score
        else:
            raise ValueError('Scoring function does not return a float')

    def _balanced_accuracy(self, result):
        """Default scoring function: balanced class accuracy

        Parameters
        ----------
        result: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class']}
            A DataFrame containing a pipeline's predictions and the corresponding classes for the testing data

        Returns
        -------
        fitness: float
            Returns a float value indicating the `individual`'s balanced accuracy on the testing data

        """
        all_classes = list(set(result['class'].values))
        all_class_accuracies = []
        for this_class in all_classes:
            sens_columns = (result['guess'] == this_class) & (result['class'] == this_class)
            sens_count = float(len(result[result['class'] == this_class]))
            this_class_sensitivity = len(result[sens_columns]) / sens_count

            spec_columns = (result['guess'] != this_class) & (result['class'] != this_class)
            spec_count = float(len(result[result['class'] != this_class]))

            this_class_specificity = len(result[spec_columns]) / spec_count

            this_class_accuracy = (this_class_sensitivity + this_class_specificity) / 2.
            all_class_accuracies.append(this_class_accuracy)

        balanced_accuracy = np.mean(all_class_accuracies)

        return balanced_accuracy

    @_gp_new_generation
    def _combined_selection_operator(self, individuals, k):
        """Perform NSGA2 selection on the population according to their Pareto fitness

        Parameters
        ----------
        individuals: list
            A list of individuals to perform selection on
        k: int
            The number of individuals to return from the selection phase

        Returns
        -------
        fitness: list
            Returns a list of individuals that were selected

        """
        return tools.selNSGA2(individuals, int(k / 5.)) * 5

    def _random_mutation_operator(self, individual):
        """Perform a replacement, insert, or shrink mutation on an individual

        Parameters
        ----------
        individual: DEAP individual
            A list of pipeline operators and model parameters that can be compiled by DEAP into a callable function

        Returns
        -------
        fitness: list
            Returns the individual with one of the mutations applied to it

        """
        mutation_techniques = [
            partial(gp.mutUniform, expr=self._toolbox.expr_mut, pset=self._pset),
            partial(gp.mutInsert, pset=self._pset),
            partial(gp.mutShrink)
        ]
        return np.random.choice(mutation_techniques)(individual)

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
            """Expression generation stops when the depth is equal to height
            or when it is randomly determined that a a node should be a terminal.
            """
            return type_ != pd.DataFrame or depth == height

        return self._generate(pset, min_, max_, condition, type_)

    # Generate function stolen straight from deap.gp.generate
    def _generate(self, pset, min_, max_, condition, type_=None):
        """Generate a Tree as a list of list. The tree is build
        from the root to the leaves, and it stop growing when the
        condition is fulfilled.

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
        height = random.randint(min_, max_)
        stack = [(0, type_)]
        while len(stack) != 0:
            depth, type_ = stack.pop()

            # We've added a type_ parameter to the condition function
            if condition(height, depth, type_):
                try:
                    term = random.choice(pset.terminals[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add "
                                      "a terminal of type '%s', but there is "
                                      "none available." % (type_,)).with_traceback(traceback)
                if inspect.isclass(term):
                    term = term()
                expr.append(term)
            else:
                try:
                    prim = random.choice(pset.primitives[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError("The gp.generate function tried to add "
                                      "a primitive of type '%s', but there is "
                                      "none available." % (type_,)).with_traceback(traceback)
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth+1, arg))
        return expr


def positive_integer(value):
    """Ensures that the provided value is a positive integer; throws an exception otherwise

    Parameters
    ----------
    value: int
        The number to evaluate

    Returns
    -------
    value: int
        Returns a positive integer
    """
    try:
        value = int(value)
    except Exception:
        raise argparse.ArgumentTypeError('Invalid int value: \'{}\''.format(value))
    if value < 0:
        raise argparse.ArgumentTypeError('Invalid positive int value: \'{}\''.format(value))
    return value


def float_range(value):
    """Ensures that the provided value is a float integer in the range (0., 1.); throws an exception otherwise

    Parameters
    ----------
    value: float
        The number to evaluate

    Returns
    -------
    value: float
        Returns a float in the range (0., 1.)
    """
    try:
        value = float(value)
    except:
        raise argparse.ArgumentTypeError('Invalid float value: \'{}\''.format(value))
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError('Invalid float value: \'{}\''.format(value))
    return value


def main():
    """Main function that is called when TPOT is run on the command line"""
    parser = argparse.ArgumentParser(description='A Python tool that automatically creates and '
                                                 'optimizes machine learning pipelines using genetic programming.',
                                     add_help=False)

    parser.add_argument('INPUT_FILE', type=str, help='Data file to optimize the pipeline on; ensure that the class label column is labeled as "class".')

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    parser.add_argument('-is', action='store', dest='INPUT_SEPARATOR', default='\t',
                        type=str, help='Character used to separate columns in the input file.')

    parser.add_argument('-o', action='store', dest='OUTPUT_FILE', default='',
                        type=str, help='File to export the final optimized pipeline.')

    parser.add_argument('-g', action='store', dest='GENERATIONS', default=100,
                        type=positive_integer, help='Number of generations to run pipeline optimization over.\nGenerally, TPOT will work better when '
                                                    'you give it more generations (and therefore time) to optimize over. TPOT will evaluate '
                                                    'GENERATIONS x POPULATION_SIZE number of pipelines in total.')

    parser.add_argument('-p', action='store', dest='POPULATION_SIZE', default=100,
                        type=positive_integer, help='Number of individuals in the GP population.\nGenerally, TPOT will work better when you give it '
                                                    ' more individuals (and therefore time) to optimize over. TPOT will evaluate '
                                                    'GENERATIONS x POPULATION_SIZE number of pipelines in total.')

    parser.add_argument('-mr', action='store', dest='MUTATION_RATE', default=0.9,
                        type=float_range, help='GP mutation rate in the range [0.0, 1.0]. We recommend using the default parameter unless you '
                                               'understand how the mutation rate affects GP algorithms.')

    parser.add_argument('-xr', action='store', dest='CROSSOVER_RATE', default=0.05,
                        type=float_range, help='GP crossover rate in the range [0.0, 1.0]. We recommend using the default parameter unless you '
                                               'understand how the crossover rate affects GP algorithms.')

    parser.add_argument('-s', action='store', dest='RANDOM_STATE', default=0,
                        type=int, help='Random number generator seed for reproducibility. Set this seed if you want your TPOT run to be reproducible '
                                       'with the same seed and data set in the future.')

    parser.add_argument('-v', action='store', dest='VERBOSITY', default=1, choices=[0, 1, 2],
                        type=int, help='How much information TPOT communicates while it is running: 0 = none, 1 = minimal, 2 = all.')

    parser.add_argument('--no-update-check', action='store_true', dest='DISABLE_UPDATE_CHECK', default=False,
                        help='Flag indicating whether the TPOT version checker should be disabled.')

    parser.add_argument('--version', action='version', version='TPOT {version}'.format(version=__version__),
                        help='Show TPOT\'s version number and exit.')

    args = parser.parse_args()

    if args.VERBOSITY >= 2:
        print('\nTPOT settings:')
        for arg in sorted(args.__dict__):
            if arg == 'DISABLE_UPDATE_CHECK':
                continue
            print('{}\t=\t{}'.format(arg, args.__dict__[arg]))
        print('')

    input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR)

    if 'Class' in input_data.columns.values:
        input_data.rename(columns={'Class': 'class'}, inplace=True)

    RANDOM_STATE = args.RANDOM_STATE if args.RANDOM_STATE > 0 else None

    training_indices, testing_indices = train_test_split(input_data.index,
                                                         stratify=input_data['class'].values,
                                                         train_size=0.75,
                                                         test_size=0.25,
                                                         random_state=RANDOM_STATE)

    training_features = input_data.loc[training_indices].drop('class', axis=1).values
    training_classes = input_data.loc[training_indices, 'class'].values

    testing_features = input_data.loc[testing_indices].drop('class', axis=1).values
    testing_classes = input_data.loc[testing_indices, 'class'].values

    tpot = TPOT(generations=args.GENERATIONS, population_size=args.POPULATION_SIZE,
                mutation_rate=args.MUTATION_RATE, crossover_rate=args.CROSSOVER_RATE,
                random_state=args.RANDOM_STATE, verbosity=args.VERBOSITY,
                disable_update_check=args.DISABLE_UPDATE_CHECK)

    tpot.fit(training_features, training_classes)

    if args.VERBOSITY >= 1:
        print('\nTraining accuracy: {}'.format(tpot.score(training_features, training_classes)))
        print('Holdout accuracy: {}'.format(tpot.score(testing_features, testing_classes)))

    if args.OUTPUT_FILE != '':
        tpot.export(args.OUTPUT_FILE)


if __name__ == '__main__':
    main()
