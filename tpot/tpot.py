# -*- coding: utf-8 -*-

"""
Copyright 2015 Randal S. Olson

This file is part of the TPOT library.

The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
the Twitter Bot library. If not, see http://www.gnu.org/licenses/.

"""

from __future__ import print_function
import argparse
import operator
import random
import hashlib
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit

import deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from operators import operator_registry

class TPOT(object):
    """TPOT automatically creates and optimizes Machine Learning pipelines using genetic programming.

    Attributes
    ----------
    optimized_pipeline_: object
        The optimized pipeline, available after calling `fit`

    """
    optimized_pipeline_ = None

    def __init__(self, population_size=100, generations=100,
                 mutation_rate=0.9, crossover_rate=0.05,
                 random_state=0, verbosity=0, scoring_function=None):
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
        scoring_function: function (default: None)
            Function used to evaluate the goodness of a given pipeline for the classification problem. By default, balanced class accuracy is used.

        Returns
        -------
        None

        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.verbosity = verbosity

        if random_state > 0:
            random.seed(random_state)
            np.random.seed(random_state)

        self.pset = gp.PrimitiveSetTyped('MAIN', [pd.DataFrame], pd.DataFrame)
            
        for Operator in operator_registry.values():
            v = Operator
            self.pset.addPrimitive(v.evaluate_operator, v.intypes, v.outtype, v.__class__.__name__)
        
        ####################################
        self.pset.addPrimitive(operator.add, [int, int], int)
        self.pset.addPrimitive(operator.sub, [int, int], int)
        self.pset.addPrimitive(operator.mul, [int, int], int)
        self.pset.addPrimitive(self._div, [int, int], float)
        ####################################
        
        for val in range(0, 101):
            self.pset.addTerminal(val, int)
        for val in [100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]:
            self.pset.addTerminal(val, float)

        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register('expr', gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register('individual', tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register('compile', gp.compile, pset=self.pset)
        self.toolbox.register('select', self._combined_selection_operator)
        self.toolbox.register('mate', gp.cxOnePoint)
        self.toolbox.register('expr_mut', gp.genFull, min_=0, max_=3)
        self.toolbox.register('mutate', self._random_mutation_operator)
	
        if not scoring_function:
            self.scoring_function=self._balanced_accuracy
        else:
            self.scoring_function=scoring_function


    def fit(self, features, classes, feature_names=None):
        """Uses genetic programming to optimize a Machine Learning pipeline that
           maximizes classification accuracy on the provided `features` and `classes`.
           Optionally, name the features in the data frame according to `feature_names`.
           Performs a stratified training/testing cross-validaton split to avoid
           overfitting on the provided data.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        classes: array-like {n_samples}
            List of class labels for prediction
        feature_names: array-like {n_features} (default: None)
            List of feature names as strings

        Returns
        -------
        None

        """
        try:
            training_testing_data = pd.DataFrame(data=features, columns=feature_names)
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

            training_indices, testing_indices = next(iter(StratifiedShuffleSplit(training_testing_data['class'].values,
                                                                                 n_iter=1,
                                                                                 train_size=0.75,
                                                                                 test_size=0.25)))

            training_testing_data.loc[training_indices, 'group'] = 'training'
            training_testing_data.loc[testing_indices, 'group'] = 'testing'

            # Default the basic guess to the most frequent class
            most_frequent_class = Counter(training_testing_data.loc[training_indices, 'class'].values).most_common(1)[0][0]
            training_testing_data.loc[:, 'guess'] = most_frequent_class

            self.toolbox.register('evaluate', self._evaluate_individual, training_testing_data=training_testing_data)

            pop = self.toolbox.population(n=self.population_size)
            self.hof = tools.HallOfFame(maxsize=1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register('Minimum score', np.min)
            stats.register('Average score', np.mean)
            stats.register('Maximum score', np.max)

            verbose = (self.verbosity == 2)

            pop, log = algorithms.eaSimple(population=pop, toolbox=self.toolbox, cxpb=self.crossover_rate,
                                           mutpb=self.mutation_rate, ngen=self.generations,
                                           stats=stats, halloffame=self.hof, verbose=verbose)

            self.optimized_pipeline_ = self.hof[0]

            if self.verbosity == 2:
                print('')

            if self.verbosity >= 1:
                print('Best pipeline:', self.hof[0])

        # Store the best pipeline if the optimization process is ended prematurely
        except KeyboardInterrupt:
            self.optimized_pipeline_ = self.hof[0]

    def predict(self, training_features, training_classes, testing_features):
        """Uses the optimized pipeline to predict the classes for a feature set.

        Parameters
        ----------
        training_features: array-like {n_samples, n_features}
            Feature matrix of the training set
        training_classes: array-like {n_samples}
            List of class labels for prediction in the training set
        testing_features: array-like {n_samples, n_features}
            Feature matrix of the testing set

        Returns
        ----------
        array-like: {n_samples}
            Predicted classes for the testing set

        """
        if self.optimized_pipeline_ is None:
            raise ValueError('A pipeline has not yet been optimized. Please call fit() first.')

        training_data = pd.DataFrame(training_features)
        training_data['class'] = training_classes
        training_data['group'] = 'training'

        testing_data = pd.DataFrame(testing_features)
        testing_data['class'] = 0
        testing_data['group'] = 'testing'

        training_testing_data = pd.concat([training_data, testing_data])
        most_frequent_class = Counter(training_classes).most_common(1)[0][0]
        training_testing_data.loc[:, 'guess'] = most_frequent_class

        new_col_names = {}
        for column in training_testing_data.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        training_testing_data.rename(columns=new_col_names, inplace=True)

        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=self.optimized_pipeline_)

        result = func(training_testing_data)
        return result.loc[result['group'] == 'testing', 'guess'].values

    def score(self, training_features, training_classes, testing_features, testing_classes):
        """Estimates the testing accuracy of the optimized pipeline.

        Parameters
        ----------
        training_features: array-like {n_samples, n_features}
            Feature matrix of the training set
        training_classes: array-like {n_samples}
            List of class labels for prediction in the training set
        testing_features: array-like {n_samples, n_features}
            Feature matrix of the testing set
        testing_classes: array-like {n_samples}
            List of class labels for prediction in the testing set

        Returns
        -------
        accuracy_score: float
            The estimated test set accuracy

        """
        if self.optimized_pipeline_ is None:
            raise ValueError('A pipeline has not yet been optimized. Please call fit() first.')

        training_data = pd.DataFrame(training_features)
        training_data['class'] = training_classes
        training_data['group'] = 'training'

        testing_data = pd.DataFrame(testing_features)
        testing_data['class'] = testing_classes
        testing_data['group'] = 'testing'

        training_testing_data = pd.concat([training_data, testing_data])
        most_frequent_class = Counter(training_classes).most_common(1)[0][0]
        training_testing_data.loc[:, 'guess'] = most_frequent_class

        for column in training_testing_data.columns.values:
            if type(column) != str:
                training_testing_data.rename(columns={column: str(column).zfill(10)}, inplace=True)

        return self._evaluate_individual(self.optimized_pipeline_, training_testing_data)[0]


    def _replace_mathematical_operators(self, exported_pipeline):
        """Replace all of the mathematical operators with their results in export().

        Parameters
        ----------
        exported_pipeline: 
           The current optimized pipeline

        Returns
        -------
        exported_pipeline: 
           The current optimized pipeline after replacing the mathematical operators

        """
        
        while True:
            for i in range(len(exported_pipeline) - 1, -1, -1):
                node = exported_pipeline[i]
                if type(node) is deap.gp.Primitive and node.name in ['add', 'sub', 'mul', '_div']:
                    val1 = int(exported_pipeline[i + 1].name)
                    val2 = int(exported_pipeline[i + 2].name)
                    if node.name == 'add':
                        new_val = val1 + val2
                    elif node.name == 'sub':
                        new_val = val1 - val2
                    elif node.name == 'mul':
                        new_val = val1 * val2
                    else:
                        new_val = self._div(val1, val2)

                    new_val = deap.gp.Terminal(symbolic=new_val, terminal=new_val, ret=new_val)
                    exported_pipeline = exported_pipeline[:i] + [new_val] + exported_pipeline[i + 3:]
                    break
            else:
                break

        return exported_pipeline

    def _unroll_nested_fuctions_calls(self, exported_pipeline):
        """Unroll the nested function calls into serial code in export()

        Parameters
        ----------
        exported_pipeline: 
           The current optimized pipeline

        Returns
        -------
        exported_pipeline: 
           The current optimized pipeline after unrolling the nested function calls
        pipeline_list: 
           List of operators in the current optimized pipeline

        """

        pipeline_list = []
        result_num = 1
        while True:
            for node_index in range(len(exported_pipeline) - 1, -1, -1):
                node = exported_pipeline[node_index]
                if type(node) is not deap.gp.Primitive:
                    continue

                node_params = exported_pipeline[node_index + 1:node_index + node.arity + 1]

                new_val = 'result{}'.format(result_num)
                operator_list = [new_val, node.name]
                operator_list.extend([x.name for x in node_params])
                pipeline_list.append(operator_list)
                result_num += 1
                new_val = deap.gp.Terminal(symbolic=new_val, terminal=new_val, ret=new_val)
                exported_pipeline = exported_pipeline[:node_index] + [new_val] + exported_pipeline[node_index + node.arity + 1:]
                break
            else:
                break
        return exported_pipeline, pipeline_list

    
    def export(self, output_file_name):
        """Exports the current optimized pipeline as Python code.

        Parameters
        ----------
        output_file_name: string
            String containing the path and file name of the desired output file

        Returns
        -------
        None

        """
        if self.optimized_pipeline_ is None:
            raise ValueError('A pipeline has not yet been optimized. Please call fit() first.')

        exported_pipeline = self.optimized_pipeline_

        exported_pipeline = self._replace_mathematical_operators(exported_pipeline)

        exported_pipeline, pipeline_list = self._unroll_nested_fuctions_calls(exported_pipeline)


        # Have the code import all of the necessary modules and functions
        # operator[1] is the name of the operator
        operators_used = set([operator[1] for operator in pipeline_list])
        
        pipeline_text = '''import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
'''
        modeling = ''
        for op_name, op in operator_registry.iteritems():
            if op_name in operators_used:
                pipeline_text += op.import_code + '\n'

        pipeline_text += '''
# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75, test_size=0.25)))

'''
        # Replace the function calls with their corresponding Python code
        for operator in pipeline_list:
            operator_num = int(operator[0].strip('result'))
            result_name = operator[0]
            operator_name = operator[1]
            operator_text = ''

            # Make copies of the data set for each reference to ARG0
            if operator[2] == 'ARG0':
                operator[2] = 'result{}'.format(operator_num)
                operator_text += '\n{} = tpot_data.copy()\n'.format(operator[2])

            if len(operator) > 3 and operator[3] == 'ARG0':
                operator[3] = 'result{}'.format(operator_num)
                operator_text += '\n{} = tpot_data.copy()\n'.format(operator[3])

             # Replace the TPOT functions with their corresponding Python code
            for op_name, op in operator_registry.iteritems():
                if operator_name == op_name:
                    operator_text += op.callable_code(operator_num, operator, result_name)
                    break                    

            pipeline_text += operator_text

        with open(output_file_name, 'w') as output_file:
            output_file.write(pipeline_text)

    @staticmethod
    def _div(num1, num2):
        """Divides two numbers
        
        Parameters
        ----------
        num1: int
            The dividend
        num2: int
            The divisor

        Returns
        -------
        result: float
            Returns num1 / num2, or 0 if num2 == 0
        
        """
        if num2 == 0:
            return 0.
        
        return float(num1) / float(num2)

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
            func = self.toolbox.compile(expr=individual)
        except MemoryError:
            # Throw out GP expressions that are too large to be compiled in Python
            return 0.,

        result = func(training_testing_data)
        result = result[result['group'] == 'testing']
        res = self.scoring_function(result)
        
        if isinstance(res, float) or isinstance(res, np.float64) or isinstance(res, np.float32):
            return res,
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
            this_class_accuracy = len(result[(result['guess'] == this_class) \
                & (result['class'] == this_class)])\
                / float(len(result[result['class'] == this_class]))
            all_class_accuracies.append(this_class_accuracy)

        balanced_accuracy = np.mean(all_class_accuracies)

        return balanced_accuracy

    def _combined_selection_operator(self, individuals, k):
        """Perform selection on the population according to their fitness

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
        
        # 10% of the new population are copies of the current best-performing pipeline (i.e., elitism)
        best_inds = int(0.1 * k)
        
        # The remaining 90% of the new population are selected by tournament selection
        rest_inds = k - best_inds
        return (tools.selBest(individuals, 1) * best_inds +
                tools.selDoubleTournament(individuals, k=rest_inds, fitness_size=3,
                                          parsimony_size=2, fitness_first=True))

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
        roll = random.random()
        if roll <= 0.333333:
            return gp.mutUniform(individual, expr=self.toolbox.expr_mut, pset=self.pset)
        elif roll <= 0.666666:
            return gp.mutInsert(individual, pset=self.pset)
        else:
            return gp.mutShrink(individual)


def main():
    parser = argparse.ArgumentParser(description='A Python tool that'
            ' automatically creates and optimizes Machine Learning pipelines'
            ' using genetic programming.')

    def positive_integer(value):
        try:
            value = int(value)
        except:
            raise argparse.ArgumentTypeError('invalid int value: \'{}\''.format(value))
        if value < 0:
            raise argparse.ArgumentTypeError('invalid positive int value: \'{}\''.format(value))
        return value

    def float_range(value):
        try:
            value = float(value)
        except:
            raise argparse.ArgumentTypeError('invalid float value: \'{}\''.format(value))
        if value < 0.0 or value > 1.0:
            raise argparse.ArgumentTypeError('invalid float value: \'{}\''.format(value))
        return value

    parser.add_argument('-i', action='store', dest='input_file', default=None,
                        type=str, help='Data file to optimize the pipeline on. Ensure that the class column is labeled as "class".')

    parser.add_argument('-is', action='store', dest='input_separator', default='\t',
                        type=str, help='Character used to separate columns in the input file.')

    parser.add_argument('-o', action='store', dest='output_file', default='',
                        type=str, help='File to export the final optimized pipeline.')

    parser.add_argument('-g', action='store', dest='generations', default=100,
                        type=positive_integer, help='Number of generations to run pipeline optimization for.')

    parser.add_argument('-mr', action='store', dest='mutation_rate', default=0.9,
                        type=float_range, help='Mutation rate in the range [0.0, 1.0]')

    parser.add_argument('-xr', action='store', dest='crossover_rate', default=0.05,
                        type=float_range, help='Crossover rate in the range [0.0, 1.0]')

    parser.add_argument('-p', action='store', dest='population_size', default=100,
                        type=positive_integer, help='Number of individuals in the GP population.')

    parser.add_argument('-s', action='store', dest='random_state', default=0,
                        type=int, help='Random number generator seed for reproducibility.')

    parser.add_argument('-v', action='store', dest='verbosity', default=1, choices=[0, 1, 2],
                        type=int, help='How much information TPOT communicates while it is running. 0 = none, 1 = minimal, 2 = all')

    parser.add_argument('--version', action='store_true', dest='version', default=False, help='Display the current TPOT version')

    args = parser.parse_args()
    
    if args.version:
        from _version import __version__
        print('TPOT version: {}'.format(__version__))
        return
    elif args.input_file is None:
        parser.print_help()
        print('\nError: You must specify an input file with -i')
        return

    if args.verbosity >= 2:
        print('\nTPOT settings:')
        for arg in sorted(args.__dict__):
            if arg == 'version':
                continue
            print('{}\t=\t{}'.format(arg, args.__dict__[arg]))
        print('')

    input_data = pd.read_csv(args.input_file, sep=args.input_separator)

    if 'Class' in input_data.columns.values:
        input_data.rename(columns={'Class': 'class'}, inplace=True)

    if args.random_state > 0:
        random_state = args.random_state
    else:
        random_state = None

    training_indices, testing_indices = next(iter(StratifiedShuffleSplit(input_data['class'].values,
                                                                         n_iter=1,
                                                                         train_size=0.75,
                                                                         test_size=0.25,
                                                                         random_state=random_state)))

    training_features = input_data.loc[training_indices].drop('class', axis=1).values
    training_classes = input_data.loc[training_indices, 'class'].values

    testing_features = input_data.loc[testing_indices].drop('class', axis=1).values
    testing_classes = input_data.loc[testing_indices, 'class'].values

    tpot = TPOT(generations=args.generations, population_size=args.population_size,
                mutation_rate=args.mutation_rate, crossover_rate=args.crossover_rate,
                random_state=args.random_state, verbosity=args.verbosity)

    tpot.fit(training_features, training_classes)

    if args.verbosity >= 1:
        print('\nTraining accuracy: {}'.format(tpot.score(training_features, training_classes,
                                             training_features, training_classes)))
        print('Testing accuracy: {}'.format(tpot.score(training_features, training_classes,
                                            testing_features, testing_classes)))
    
    if args.output_file != '':
        tpot.export(args.output_file)




if __name__ == '__main__':
    main()
