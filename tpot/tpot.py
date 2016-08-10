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
import argparse
import random
import inspect
import sys
from functools import partial
from collections import Counter

import numpy as np
import deap
from deap import algorithms, base, creator, tools, gp
from tqdm import tqdm
from sklearn.cross_validation import train_test_split
from sklearn import metrics

import warnings
from update_checker import update_check

from ._version import __version__
from .export_utils import export_pipeline
from .decorators import _gp_new_generation
from . import operators
from .types import Bool, Output_DF
from .indices import GUESS_COL, GROUP_COL, CLASS_COL, TESTING_GROUP


class TPOT(object):
    """TPOT automatically creates and optimizes machine learning pipelines using
    genetic programming
    """

    def __init__(self, population_size=100, generations=100,
                 mutation_rate=0.9, crossover_rate=0.05,
                 random_state=None, verbosity=0, scoring_function=None,
                 scoring_kwargs={}, disable_update_check=False):
        """Sets up the genetic programming algorithm for pipeline optimization.

        Parameters
        ----------
        population_size: int (default: 100)
            The number of pipelines in the genetic algorithm population. Must
            be > 0.The more pipelines in the population, the slower TPOT will
            run, but it's also more likely to find better pipelines.
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
        random_state: int (default: 0)
            The random number generator seed for TPOT. Use this to make sure
            that TPOT will give you the same results each time you run it
            against the same data set with that seed.
        verbosity: int (default: 0)
            How much information TPOT communicates while it's running.
            0 = none, 1 = minimal, 2 = all
        scoring_function: function (default: balanced accuracy)
            Function used to evaluate the goodness of a given pipeline for the
            classification problem. By default, balanced class accuracy is used.
        disable_update_check: bool (default: False)
            Flag indicating whether the TPOT version checker should be disabled.

        Returns
        -------
        None

        """
        # Save params to be recalled later by get_params()
        self.params = locals()  # Must be before any local variable definitions
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

        if random_state:
            random.seed(random_state)
            np.random.seed(random_state)

        self.score_sign = 1
        self.clf_eval_func = False
        if scoring_function is None:
            self.scoring_function = self._balanced_accuracy
            self.scoring_kwargs = {}
        else:
            self.scoring_function = scoring_function
            self.scoring_kwargs = scoring_kwargs

        # If the scoring function has loss in the name, maximize the negative of the fitness score
        if 'loss' in self.scoring_function.__name__:
            self.score_sign = -1
        self.clf_eval_func = self._parse_scoring_docstring(self.scoring_function)
    
        self._setup_pset()
        self._setup_toolbox()
    
    def _setup_pset(self):
        self._pset = gp.PrimitiveSetTyped('MAIN', [np.ndarray], Output_DF)

        # Rename pipeline input to "input_df"
        self._pset.renameArguments(ARG0='input_matrix')

        # Add all operators to the primitive set
        for op in operators.Operator.inheritors():
            if op.root:
                # We need to add rooted primitives twice so that they can
                # return both an Output_DF (and thus be the root of the tree),
                # and return a np.ndarray so they can exist elsewhere in the
                # tree.
                p_types = ([str] +  op.parameter_types()[0], Output_DF)
                self._pset.addPrimitive(op, *p_types)

            self._pset.addPrimitive(op, [str] +  op.parameter_types()[0], op.parameter_types()[1])

        self._pset.addPrimitive(operators.CombineDFs(),
            [np.ndarray, np.ndarray], np.ndarray)

        # Terminals
        int_terminals = np.concatenate((
            np.arange(0, 51, 1),
            np.arange(60, 110, 10))
        )

        for val in int_terminals:
            self._pset.addTerminal(val, int)

        float_terminals = np.concatenate((
            [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            np.linspace(0., 1., 101),
            np.linspace(2., 50., 49),
            np.linspace(60., 100., 5))
        )

        for val in float_terminals:
            self._pset.addTerminal(val, float)

        self._pset.addTerminal(True, Bool)
        self._pset.addTerminal(False, Bool)
        self._pset.addTerminal(self.clf_eval_func, str)

    def _setup_toolbox(self):
        creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
        creator.create('Individual',
            gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self._toolbox = base.Toolbox()
        self._toolbox.register('expr',
            self._gen_grow_safe, pset=self._pset, min_=1, max_=3)
        self._toolbox.register('individual',
            tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population',
            tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register('compile', gp.compile, pset=self._pset)
        self._toolbox.register('select', self._combined_selection_operator)
        self._toolbox.register('mate', gp.cxOnePoint)
        self._toolbox.register('expr_mut', self._gen_grow_safe, min_=1, max_=3)
        self._toolbox.register('mutate', self._random_mutation_operator)


    def fit(self, features, classes):
        """Fits a machine learning pipeline that maximizes classification
        accuracy on the provided data

        Uses genetic programming to optimize a machine learning pipeline that
        maximizes classification accuracy on the provided features and classes.
        Performs an internal stratified training/testing cross-validaton split
        to avoid overfitting on the provided data.

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

            # Randomize the order of the columns so there is no potential bias
            # introduced by the initial order of the columns, e.g., the most
            # predictive features at the beginning or end.
            np.take(features, np.random.permutation(features.shape[1]), axis=1, out=features)

            training_features, testing_features, training_classes, testing_classes = \
                train_test_split(features, classes, train_size=0.75, test_size=0.25)

            # Training group is 0. testing group is 1.
            training_data = np.insert(training_features, 0, training_classes, axis=1)  # Insert the classes
            training_data = np.insert(training_data, 0, np.zeros((training_data.shape[0],)), axis=1)  # Insert the group
            testing_data = np.insert(testing_features, 0, np.zeros((testing_features.shape[0],)), axis=1)  # Insert the classes
            testing_data = np.insert(testing_data, 0, np.ones((testing_data.shape[0],)), axis=1)  # Insert the group

            # Insert guess
            most_frequent_class = Counter(training_classes).most_common(1)[0][0]
            data = np.concatenate([training_data, testing_data])
            data = np.insert(data, 0, np.array([most_frequent_class] * data.shape[0]), axis=1)



            self._toolbox.register('evaluate', self._evaluate_individual, data=data)

            pop = self._toolbox.population(n=self.population_size)

            def pareto_eq(ind1, ind2):
                """Function used to determine whether two individuals are equal
                on the Pareto front

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
                return np.all(ind1.fitness.values == ind2.fitness.values)

            self.hof = tools.ParetoFront(similar=pareto_eq)

            verbose = (self.verbosity == 2)

            # Start the progress bar
            num_evaluations = self.population_size * (self.generations + 1)
            self.pbar = tqdm(total=num_evaluations, unit='pipeline', leave=False,
                             disable=(not verbose), desc='GP Progress')

            pop, _ = algorithms.eaSimple(
                population=pop, toolbox=self._toolbox, cxpb=self.crossover_rate,
                mutpb=self.mutation_rate, ngen=self.generations,
                halloffame=self.hof, verbose=False)

        # Allow for certain exceptions to signal a premature fit() cancellation
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            # Close the progress bar
            # Standard truthiness checks won't work for tqdm
            if not isinstance(self.pbar, type(None)):
                self.pbar.close()

            # Reset gp_generation counter to restore initial state
            self.gp_generation = 0

            # Store the pipeline with the highest internal testing accuracy
            if self.hof:
                if self.score_sign == -1:
                    top_score = -5000.
                else:
                    top_score = 0.
                for pipeline in self.hof:
                    pipeline_score = self._evaluate_individual(pipeline, data)[1]
                    if pipeline_score > top_score:
                        top_score = pipeline_score
                        self._optimized_pipeline = pipeline

            if self.verbosity >= 1 and self._optimized_pipeline:
                # Add an extra line of spacing if the progress bar was used
                if verbose:
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
            raise ValueError(('A pipeline has not yet been optimized.'
                              'Please call fit() first.'))

        training_data = np.insert(self._training_features, 0, self._training_classes, axis=1)  # Insert the classes
        training_data = np.insert(training_data, 0, np.zeros((training_data.shape[0],)), axis=1)  # Insert the group
        testing_data = np.insert(testing_features, 0, np.zeros((testing_features.shape[0],)), axis=1)  # Insert the classes
        testing_data = np.insert(testing_data, 0, np.ones((testing_data.shape[0],)), axis=1)  # Insert the group

        most_frequent_class = Counter(self._training_classes).most_common(1)[0][0]
        data = np.concatenate([training_data, testing_data])
        data = np.insert(data, 0, np.array([most_frequent_class] * data.shape[0]), axis=1)

        # Transform the tree expression in a callable function
        func = self._toolbox.compile(expr=self._optimized_pipeline)
        result = func(data)

        return result[result[:, GROUP_COL] == TESTING_GROUP][:, CLASS_COL]

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
            raise ValueError(('A pipeline has not yet been optimized.'
                              'Please call fit() first.'))

        training_data = np.insert(self._training_features, 0, self._training_classes, axis=1)  # Insert the classes
        training_data = np.insert(training_data, 0, np.zeros((training_data.shape[0],)), axis=1)  # Insert the group
        testing_data = np.insert(testing_features, 0, testing_classes, axis=1)  # Insert the classes
        testing_data = np.insert(testing_data, 0, np.ones((testing_data.shape[0],)), axis=1)  # Insert the group

        most_frequent_class = Counter(self._training_classes).most_common(1)[0][0]
        data = np.concatenate([training_data, testing_data])
        data = np.insert(data, 0, np.array([most_frequent_class] * data.shape[0]), axis=1)

        # Default guess: the most frequent class in the training data
        most_frequent_class = Counter(self._training_classes).most_common(1)[0][0]
        data = np.concatenate([training_data, testing_data])
        data = np.insert(data, 0, np.array([most_frequent_class] * data.shape[0]), axis=1)

        return self._evaluate_individual(self._optimized_pipeline, data)[1]

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
        output_file_name: str
            String containing the path and file name of the desired output file

        Returns
        -------
        None

        """
        if self._optimized_pipeline is None:
            raise ValueError(('A pipeline has not yet been optimized.'
                              'Please call fit() first.'))

        with open(output_file_name, 'w') as output_file:
            output_file.write(export_pipeline(self._optimized_pipeline))

    def _parse_scoring_docstring(self, scoring_function):
        """Function used to determine what function a classifier will need to use to supply to the scoring function

        Parameters
        ----------
        scoring_function: method: [true labels], [predicted labels OR prediction probabilities OR decision function output] -> float
            Classification metric used to evaluate a pipeline's fitness

        Returns
        -------
        clf_eval_func: string - {'predict', 'predict_proba', 'decision_function'}
            String representation of what each classifier will need to supply to the scoring function 

        """
        
        possible_sklearn_metrics = [name for name, val in metrics.__dict__.items()]

        if str(scoring_function.__name__ ) in possible_sklearn_metrics:
            docstring = str(scoring_function.__doc__)
            if 'decision_function' in docstring:
                return 'decision_function'
            elif 'predict_proba' in docstring:
                return 'predict_proba'
            elif 'predicted labels' in docstring.lower() or 'estimated targets' in docstring.lower():
                return 'predict'

        return 'predict'

    def _evaluate_individual(self, individual, data):
        """Determines the `individual`'s fitness according to its performance on
        the provided data

        Parameters
        ----------
        individual: DEAP individual
            A list of pipeline operators and model parameters that can be
            compiled by DEAP into a callable function
        data: numpy.ndarray {n_samples, n_features}
            A numpy matrix containing the training and testing data for the
            `individual`'s evaluation

        Returns
        -------
        fitness: float
            Returns a float value indicating the `individual`'s fitness
            according to its performance on the provided data

        """

        try:
            # Transform the tree expression in a callable function
            func = self._toolbox.compile(expr=individual)

            # Count the number of pipeline operators as a measure of pipeline
            # complexity
            operator_count = 0
            for i in range(len(individual)):
                node = individual[i]
                if type(node) is deap.gp.Terminal:
                    continue
                if type(node) is deap.gp.Primitive and node.name == 'CombineDFs':
                    continue
                operator_count += 1

            result = func(data)
            result = result[result[:, GROUP_COL] == TESTING_GROUP]
            #If the scoring function requires we use predict_proba or decision_function then unpickle the data we pickled when training the classifier
            if self.clf_eval_func == 'predict_proba' or self.clf_eval_func == 'decision_function':
                resulting_score = self.scoring_function(result[:, CLASS_COL], [np.loads(x) for x in result[:, GUESS_COL]], **self.scoring_kwargs)
            else:
                resulting_score = self.scoring_function(result[:, CLASS_COL], result[:, GUESS_COL], **self.scoring_kwargs)

        except MemoryError:
            # Throw out GP expressions that are too large to be compiled
            if self.score_sign == -1:
                return 5000., -5000.
            else:
                return 5000., 0.
        except (KeyboardInterrupt, SystemExit):
            raise
        # except Exception:
            # Catch-all: Do not allow one pipeline that crashes to cause TPOT
            # to crash. Instead, assign the crashing pipeline a poor fitness
        #     if self.score_sign == -1:
        #         return 5000., -5000.
        #     else:
        #         return 5000., 0.
        #     return 5000., 0.
        finally:
            if not self.pbar.disable:
                self.pbar.update(1)  # One more pipeline evaluated
                
        if type(resulting_score) in [float, np.float64, np.float32]:
            return max(1, operator_count), resulting_score * self.score_sign
        else:
            raise ValueError('Scoring function does not return a float')

    def _balanced_accuracy(self, y_true, y_pred):
        """Default scoring function: balanced class accuracy

        Parameters
        ----------
        y_true: numpy.ndarray {n_samples, 1}
            A numpy matrix containing a pipeline's classes for the testing data

        y_pred: numpy.ndarray {n_samples, 1}
            A numpy matrix containing a pipeline's guesses for the testing data 

        Returns
        -------
        fitness: float
            Returns a float value indicating the `individual`'s balanced
            accuracy on the testing data

        """
        result = np.zeros((y_true.shape, 2))
        class_col = 0
        guess_col = 1
        result[:, class_col] = y_true
        result[:, guess_col] = y_pred

        all_classes = list(set(result[:, class_col]))
        all_class_accuracies = []
        for this_class in all_classes:
            this_class_sensitivity = \
                float(result[(result[:, guess_col] == this_class) &
                             (result[:, class_col] == this_class)].shape[0]) /\
                float(result[result[:, class_col] == this_class].shape[0])

            this_class_specificity = \
                float(result[(result[:, guess_col] != this_class) &
                             (result[:, class_col] != this_class)].shape[0]) /\
                float(result[result[:, class_col] != this_class].shape[0])

            this_class_accuracy = (this_class_sensitivity + this_class_specificity) / 2.
            all_class_accuracies.append(this_class_accuracy)

        balanced_accuracy = np.mean(all_class_accuracies)

        return balanced_accuracy

    @_gp_new_generation
    def _combined_selection_operator(self, individuals, k):
        """Perform NSGA2 selection on the population according to their Pareto
        fitness

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
            A list of pipeline operators and model parameters that can be
            compiled by DEAP into a callable function

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
            or when it is randomly determined that a a node should be a terminal
            """
            return type_ not in [np.ndarray, Output_DF] or depth == height

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


def positive_integer(value):
    """Ensures that the provided value is a positive integer; throws an
    exception otherwise

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
    """Ensures that the provided value is a float integer in the range (0., 1.)
    throws an exception otherwise

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

    parser.add_argument('-s', action='store', dest='RANDOM_STATE', default=None,
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

    input_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
    features = input_data.view((np.float64, len(input_data.dtype.names)))
    features = np.delete(features, input_data.dtype.names.index('class'))

    training_features, testing_features, training_classes, testing_classes = \
        train_test_split(features, input_data['class'], random_state=args.RANDOM_STATE)

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
