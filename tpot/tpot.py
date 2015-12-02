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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit

import deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

class TPOT(object):
    """TPOT automatically creates and optimizes Machine Learning pipelines using genetic programming.

    Parameters
    ----------
    population_size: int (default: 100)
        The number of pipelines in the genetic algorithm population.
        Must be > 0. The more pipelines in the population,
        the slower TPOT will run, but it's also more likely to
        find better pipelines.
    generations: int (default: 100)
        The number of generations to run pipeline optimization for. Must be > 0.
        The more generations you give TPOT to run, the longer it takes,
        but it's also more likely to find better pipelines.
    mutation_rate: float (default: 0.9)
        The mutation rate for the genetic programming algorithm
        in the range [0.0, 1.0]. This tells the genetic programming algorithm
        how many pipelines to apply random changes to every generation.
        We don't recommend that you tweak this parameter unless you
        know what you're doing.
    crossover_rate: float (default: 0.05)
        The crossover rate for the genetic programming algorithm
        in the range [0.0, 1.0]. This tells the genetic programming
        algorithm how many pipelines to "breed" every generation.
        We don't recommend that you tweak this parameter
        unless you know what you're doing.
    random_state: int (default: 0)
        The random number generator seed for TPOT.
        Use this to make sure that TPOT will give you the same results
        each time you run it against the same data set with that seed.
        No random seed if random_state=None.
    verbosity: int {0, 1, 2} (default: 0)
        How much information TPOT communicates while
        it's running. 0 = none, 1 = minimal, 2 = all

    Attributes
    ----------
    best_features_cache_: dict
        Best features, available after calling `fit`
    optimized_pipeline_: object
        The optimized pipeline, available after calling `fit`

    """
    optimized_pipeline_ = None
    best_features_cache_ = {}

    def __init__(self, population_size=100, generations=100,
                 mutation_rate=0.9, crossover_rate=0.05,
                 random_state=0, verbosity=0):
        """Sets up the genetic programming algorithm for pipeline optimization."""
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.verbosity = verbosity

        if random_state > 0:
            random.seed(random_state)
            np.random.seed(random_state)

        self.pset = gp.PrimitiveSetTyped('MAIN', [pd.DataFrame], pd.DataFrame)
        self.pset.addPrimitive(self.decision_tree, [pd.DataFrame, int, int], pd.DataFrame)
        self.pset.addPrimitive(self.random_forest, [pd.DataFrame, int, int], pd.DataFrame)
        self.pset.addPrimitive(self._combine_dfs, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
        self.pset.addPrimitive(self._subset_df, [pd.DataFrame, int, int], pd.DataFrame)
        self.pset.addPrimitive(self._dt_feature_selection, [pd.DataFrame, int], pd.DataFrame)

        self.pset.addPrimitive(operator.add, [int, int], int)
        self.pset.addPrimitive(operator.sub, [int, int], int)
        self.pset.addPrimitive(operator.mul, [int, int], int)
        for val in range(0, 101):
            self.pset.addTerminal(val, int)

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
        classes: array-like {n_classnames}
            List of class name strings
        feature_names: array-like {n_featurenames} (default: None)
            List of feature name strings

        Returns
        -------
        None

        """
        try:
            self.best_features_cache_ = {}

            training_testing_data = pd.DataFrame(data=features, columns=feature_names)
            training_testing_data['class'] = classes

            for column in training_testing_data.columns.values:
                if type(column) != str:
                    training_testing_data.rename(columns={column: str(column).zfill(5)}, inplace=True)

            # Randomize the order of the columns so there is no potential bias introduced by the initial order
            # of the columns, e.g., the most predictive features at the beginning or end.
            data_columns = list(training_testing_data.columns.values)
            np.random.shuffle(data_columns)
            training_testing_data = training_testing_data[data_columns]

            training_indeces, testing_indeces = next(iter(StratifiedShuffleSplit(training_testing_data['class'].values,
                                                                                 n_iter=1,
                                                                                 train_size=0.75)))

            training_testing_data.loc[training_indeces, 'group'] = 'training'
            training_testing_data.loc[testing_indeces, 'group'] = 'testing'

            # Default the basic guess to the most frequent class
            most_frequent_class = Counter(training_testing_data.loc[training_indeces, 'class'].values).most_common(1)[0][0]
            training_testing_data['guess'] = most_frequent_class

            self.toolbox.register('evaluate', self._evaluate_individual, training_testing_data=training_testing_data)

            pop = self.toolbox.population(n=self.population_size)
            self.hof = tools.HallOfFame(maxsize=1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register('Minimum accuracy', np.min)
            stats.register('Average accuracy', np.mean)
            stats.register('Maximum accuracy', np.max)

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
        training_classes: array-like {n_classnames}
            List of class name strings in the training set
        testing_features: array-like {n_samples, n_features}
            Feature matrix of the test set

        Returns
        ----------
        array-like: {n_samples, n_features}
            Feature matrix of the `guess` values

        """
        if self.optimized_pipeline_ is None:
            raise Exception('A pipeline has not yet been optimized. Please call fit() first.')

        self.best_features_cache_ = {}

        training_data = pd.DataFrame(training_features)
        training_data['class'] = training_classes
        training_data['group'] = 'training'

        testing_data = pd.DataFrame(testing_features)
        testing_data['class'] = 0
        testing_data['group'] = 'testing'

        training_testing_data = pd.concat([training_data, testing_data])
        most_frequent_class = Counter(training_classes).most_common(1)[0][0]
        training_testing_data['guess'] = most_frequent_class

        for column in training_testing_data.columns.values:
            if type(column) != str:
                training_testing_data.rename(columns={column: str(column).zfill(5)}, inplace=True)

        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=self.optimized_pipeline_)

        result = func(training_testing_data)
        return result[result['group'] == 'testing', 'guess'].values

    def score(self, training_features, training_classes, testing_features, testing_classes):
        """Estimates the testing accuracy of the optimized pipeline.

        Parameters
        ----------
        training_features: array-like {n_samples, n_features}
            Feature matrix of the training set
        training_classes: array-like {n_classnames}
            List of class name strings in the training set
        test_features: array-like {n_samples, n_features}
            Feature matrix of the test set
        testing_classes: array-like {n_classnames}
            List of class name strings in the test set

        Returns
        -------
        accuracy_score: float
            The estimated test set accuracy

        """
        if self.optimized_pipeline_ is None:
            raise Exception('A pipeline has not yet been optimized. Please call fit() first.')

        self.best_features_cache_ = {}

        training_data = pd.DataFrame(training_features)
        training_data['class'] = training_classes
        training_data['group'] = 'training'

        testing_data = pd.DataFrame(testing_features)
        testing_data['class'] = testing_classes
        testing_data['group'] = 'testing'

        training_testing_data = pd.concat([training_data, testing_data])
        most_frequent_class = Counter(training_classes).most_common(1)[0][0]
        training_testing_data['guess'] = most_frequent_class

        for column in training_testing_data.columns.values:
            if type(column) != str:
                training_testing_data.rename(columns={column: str(column).zfill(5)}, inplace=True)

        return self._evaluate_individual(self.optimized_pipeline_, training_testing_data)[0]

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
        if self.optimized_pipeline_ == None:
            raise Exception('A pipeline has not yet been optimized. Please call fit() first.')
        
        exported_pipeline = self.optimized_pipeline_
        
        # Replace all of the mathematical operators with their results
        while True:
            for i in range(len(exported_pipeline) - 1, -1, -1):
                node = exported_pipeline[i]
                if type(node) == deap.gp.Primitive and node.name in ['add', 'sub', 'mul']:
                    val1 = int(exported_pipeline[i + 1].name)
                    val2 = int(exported_pipeline[i + 2].name)
                    if node.name == 'add':
                        new_val = val1 + val2
                    elif node.name == 'sub':
                        new_val = val1 - val2
                    else:
                        new_val = val1 * val2

                    new_val = deap.gp.Terminal(symbolic=new_val, terminal=new_val, ret=new_val)
                    exported_pipeline = exported_pipeline[:i] + [new_val] + exported_pipeline[i + 3:]
                    break
            else:
                break

        # Unroll the nested function calls into serial code
        pipeline_list = []
        result_num = 1
        while True:
            for node_index in range(len(exported_pipeline) - 1, -1, -1):
                node = exported_pipeline[node_index]
                if type(node) != deap.gp.Primitive:
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
    
        # Replace the function calls with their corresponding Python code
        pipeline_text = '''from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indeces, testing_indeces = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75)))

'''
        for operator in pipeline_list:
            operator_num = int(operator[0].strip('result'))
            result_name = operator[0]
            operator_name = operator[1]
            operator_text = ''
        
            # Make copies of the data set for each reference to ARG0
            if operator[2] == 'ARG0':
                operator[2] = 'result{}'.format(operator_num)
                operator_text += '\n{} = tpot_data.copy()\n'.format(operator[2])

            if operator[3] == 'ARG0':
                operator[3] = 'result{}'.format(operator_num)
                operator_text += '\n{} = tpot_data.copy()\n'.format(operator[3])


            # Replace the TPOT functions with their corresponding Python code
            if operator_name == 'decision_tree':
                max_features = int(operator[3])
                max_depth = int(operator[4])
            
                if max_features < 1:
                    max_features = '\'auto\''
                elif max_features == 1:
                    max_features = None
                else:
                    max_features = 'min({}, len({}.columns) - 1)'.format(max_features, operator[2])

                if max_depth < 1:
                    max_depth = None

                operator_text += '\n# Perform classification with a decision tree classifier'
                operator_text += '\ndtc{} = DecisionTreeClassifier(max_features={}, max_depth={})\n'.format(operator_num, max_features, max_depth)
                operator_text += '''dtc{0}.fit({1}.loc[training_indeces].drop('class', axis=1).values, {1}.loc[training_indeces]['class'].values)\n'''.format(operator_num, operator[2])
                if result_name != operator[2]:
                    operator_text += '{} = {}\n'.format(result_name, operator[2])
                operator_text += '''{0}['dtc{1}-classification'] = dtc{1}.predict({0}.drop('class', axis=1).values)\n'''.format(result_name, operator_num)
        
            elif operator_name == 'random_forest':
                num_trees = int(operator[3])
                max_features = int(operator[4])
            
                if num_trees < 1:
                    num_trees = 1
                elif num_trees > 500:
                    num_trees = 500

                if max_features < 1:
                    max_features = '\'auto\''
                elif max_features == 1:
                    max_features = 'None'
                else:
                    max_features = 'min({}, len({}.columns) - 1)'.format(max_features, operator[2])
            
                operator_text += '\n# Perform classification with a random forest classifier'
                operator_text += '\nrfc{} = RandomForestClassifier(n_estimators={}, max_features={})\n'.format(operator_num, num_trees, max_features)
                operator_text += '''rfc{0}.fit({1}.loc[training_indeces].drop('class', axis=1).values, {1}.loc[training_indeces]['class'].values)\n'''.format(operator_num, operator[2])
                if result_name != operator[2]:
                    operator_text += '{} = {}\n'.format(result_name, operator[2])
                operator_text += '''{0}['rfc{1}-classification'] = rfc{1}.predict({0}.drop('class', axis=1).values)\n'''.format(result_name, operator_num)
        
            elif operator_name == '_combine_dfs':
                operator_text += '\n# Combine two DataFrames'
                operator_text += '\n{2} = {0}.join({1}[[column for column in {1}.columns.values if column not in {0}.columns.values]])\n'.format(operator[2], operator[3], result_name)
        
            elif operator_name == '_subset_df':
                start = int(operator[3])
                stop = int(operator[4])
                if stop <= start:
                    stop = start + 1
                
                operator_text += '\n# Subset the data columns'
                operator_text += '\nsubset_df1 = {0}[sorted({0}.columns.values)[{1}:{2}]]\n'.format(operator[2], start, stop)
                operator_text += '''subset_df2 = {0}[[column for column in ['class'] if column not in subset_df1.columns.values]]\n'''.format(operator[2])
                operator_text += '{} = subset_df1.join(subset_df2)\n'.format(result_name)
        
            elif operator_name == '_dt_feature_selection':
                operator_text += '''
# Decision-tree based feature selection
training_features = {0}.loc[training_indeces].drop('class', axis=1)
training_class_vals = {0}.loc[training_indeces, 'class'].values

pair_scores = dict()
for features in combinations(training_features.columns.values, 2):
    dtc = DecisionTreeClassifier()
    training_feature_vals = training_features[list(features)].values
    dtc.fit(training_feature_vals, training_class_vals)
    pair_scores[features] = (dtc.score(training_feature_vals, training_class_vals), list(features))

best_pairs = []
for pair in sorted(pair_scores, key=pair_scores.get, reverse=True)[:{1}]:
    best_pairs.extend(list(pair))
best_pairs = sorted(list(set(best_pairs)))

{2} = {0}[sorted(list(set(best_pairs + ['class'])))]
'''.format(operator[2], operator[3], result_name)
        
            pipeline_text += operator_text
        
        with open(output_file_name, 'w') as output_file:
            output_file.write(pipeline_text)

    @staticmethod
    def decision_tree(input_df, max_features, max_depth):
        """Fits a decision tree classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the decision tree
        max_features: int
            Number of features used to fit the decision tree
        max_depth: int
            Maximum depth of the decision tree

        Returns
        -------
        input_df:  {n_samples, n_features+['guess']}
            Returns a modified input DataFrame with the guess column changed.

        """
        if max_features < 1:
            max_features = 'auto'
        elif max_features == 1:
            max_features = None
        elif max_features > len(input_df.columns) - 3:
            max_features = len(input_df.columns) - 3

        if max_depth < 1:
            max_depth = None

        input_df = input_df.copy()

        if len(input_df.columns) == 3:
            return input_df

        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1).values
        training_classes = input_df.loc[input_df['group'] == 'training', 'class'].values

        dtc = DecisionTreeClassifier(max_features=max_features,
                                     max_depth=max_depth,
                                     random_state=42)

        dtc.fit(training_features, training_classes)

        all_features = input_df.drop(['class', 'group', 'guess'], axis=1).values
        input_df['guess'] = dtc.predict(all_features)

        # Also store the guesses as a synthetic feature
        sf_hash = '-'.join(sorted(input_df.columns.values))
        sf_hash += 'DT-{}-{}'.format(max_features, max_depth)
        sf_identifier = 'SyntheticFeature-{}'.format(hashlib.sha224(sf_hash.encode('UTF-8')).hexdigest())
        input_df[sf_identifier] = input_df['guess'].values

        return input_df

    @staticmethod
    def random_forest(input_df, num_trees, max_features):
        """Fits a random forest classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the decision tree
        num_trees: int
            Number of trees in the random forest
        max_features: int
            Number of features used to fit the decision tree

        Returns
        -------
        input_df:  {n_samples, n_features+['guess']}
            Returns a modified input DataFrame with the guess column changed.

        """
        if num_trees < 1:
            num_trees = 1
        elif num_trees > 500:
            num_trees = 500

        if max_features < 1:
            max_features = 'auto'
        elif max_features == 1:
            max_features = None
        elif max_features > len(input_df.columns) - 3:
            max_features = len(input_df.columns) - 3

        input_df = input_df.copy()

        if len(input_df.columns) == 3:
            return input_df

        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1).values
        training_classes = input_df.loc[input_df['group'] == 'training', 'class'].values

        rfc = RandomForestClassifier(n_estimators=num_trees,
                                     max_features=max_features,
                                     random_state=42,
                                     n_jobs=-1)
        rfc.fit(training_features, training_classes)

        all_features = input_df.drop(['class', 'group', 'guess'], axis=1).values
        input_df['guess'] = rfc.predict(all_features)

        # Also store the guesses as a synthetic feature
        sf_hash = '-'.join(sorted(input_df.columns.values))
        sf_hash += 'RF-{}-{}'.format(num_trees, max_features)
        sf_identifier = 'SyntheticFeature-{}'.format(hashlib.sha224(sf_hash.encode('UTF-8')).hexdigest())
        input_df[sf_identifier] = input_df['guess'].values

        return input_df

    @staticmethod
    def _combine_dfs(input_df1, input_df2):
        """Function to combine two DataFrames"""
        return input_df1.join(input_df2[[column for column in input_df2.columns.values if column not in input_df1.columns.values]]).copy()

    @staticmethod
    def _subset_df(input_df, start, stop):
        """Subset the provided DataFrame down to the columns between the `start` and `stop` column indeces."""
        if stop <= start:
            stop = start + 1

        subset_df1 = input_df[sorted(input_df.columns.values)[start:stop]]
        subset_df2 = input_df[[column for column in ['guess', 'class', 'group'] if column not in subset_df1.columns.values]]
        return subset_df1.join(subset_df2).copy()

    def _dt_feature_selection(self, input_df, num_pairs):
        """Uses decision trees to discover the best pair(s) of features to keep."""
        num_pairs = min(max(1, num_pairs), 50)

        # If this set of features has already been analyzed, use the cache.
        # Since the smart subset can be costly, this will save a lot of computation time.
        input_df_columns_hash = hashlib.sha224('-'.join(sorted(input_df.columns.values)).encode('UTF-8')).hexdigest()
        if input_df_columns_hash in self.best_features_cache_:
            best_pairs = []
            for pair in self.best_features_cache_[input_df_columns_hash][:num_pairs]:
                best_pairs += list(pair)
            return input_df[sorted(list(set(best_pairs + ['guess', 'class', 'group'])))].copy()

        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values

        pair_scores = {}

        for features in combinations(training_features.columns.values, 2):
            dtc = DecisionTreeClassifier(random_state=42)
            training_feature_vals = training_features[list(features)].values
            dtc.fit(training_feature_vals, training_class_vals)
            pair_scores[features] = (dtc.score(training_feature_vals, training_class_vals), list(features))

        if len(pair_scores) == 0:
            return input_df[['guess', 'class', 'group']].copy()

        # Keep the best features cache within a reasonable size
        if len(self.best_features_cache_) > 1000:
            del self.best_features_cache_[list(self.best_features_cache_.keys())[0]]

        # Keep `num_pairs` best pairs of features
        best_pairs = []
        for pair in sorted(pair_scores, key=pair_scores.get, reverse=True)[:num_pairs]:
            best_pairs.extend(list(pair))
        best_pairs = sorted(list(set(best_pairs)))

        # Store the best 50 pairs of features in the cache
        self.best_features_cache_[input_df_columns_hash] = [list(pair) for pair in sorted(pair_scores, key=pair_scores.get, reverse=True)[:50]]

        return input_df[sorted(list(set(best_pairs + ['guess', 'class', 'group'])))].copy()

    def _evaluate_individual(self, individual, training_testing_data):
        """Determines the `individual`'s classification balanced accuracy on the provided data."""
        try:
            # Transform the tree expression in a callable function
            func = self.toolbox.compile(expr=individual)
        except MemoryError:
            # Throw out GP expressions that are too large to be compiled in Python
            return 0.,

        result = func(training_testing_data)
        result = result[result['group'] == 'testing']

        all_classes = list(set(result['class'].values))
        all_class_accuracies = []
        for this_class in all_classes:
            this_class_accuracy = len(result[(result['guess'] == this_class) \
                    & (result['class'] == this_class)])\
                    / float(len(result[result['class'] == this_class]))
            all_class_accuracies.append(this_class_accuracy)

        balanced_accuracy = np.mean(all_class_accuracies)

        return balanced_accuracy,

    def _combined_selection_operator(self, individuals, k):
        """Regular selection + elitism."""
        best_inds = int(0.1 * k)
        rest_inds = k - best_inds
        return (tools.selBest(individuals, 1) * best_inds +
                tools.selDoubleTournament(individuals, k=rest_inds, fitness_size=3,
                                          parsimony_size=2, fitness_first=True))

    def _random_mutation_operator(self, individual):
        """Randomly picks a replacement, insert, or shrink mutation."""
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

    parser.add_argument('-i', action='store', dest='input_file', required=True,
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

    args = parser.parse_args()

    if args.verbosity >= 2:
        print('\nTPOT settings:')
        for arg in sorted(args.__dict__):
            print('{}\t=\t{}'.format(arg, args.__dict__[arg]))

    input_data = pd.read_csv(args.input_file, sep=args.input_separator)

    if 'Class' in input_data.columns.values:
        input_data.rename(columns={'Class': 'class'}, inplace=True)

    if args.random_state > 0:
        random_state = args.random_state
    else:
        random_state = None

    training_indeces, testing_indeces = next(iter(StratifiedShuffleSplit(input_data['class'].values,
                                                                         n_iter=1,
                                                                         train_size=0.75,
                                                                         random_state=random_state)))

    training_features = input_data.loc[training_indeces].drop('class', axis=1).values
    training_classes = input_data.loc[training_indeces, 'class'].values

    testing_features = input_data.loc[testing_indeces].drop('class', axis=1).values
    testing_classes = input_data.loc[testing_indeces, 'class'].values

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
