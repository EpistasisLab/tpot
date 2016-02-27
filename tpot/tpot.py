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
import operator
import random
import hashlib
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectPercentile, RFE, SelectFwe
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import StratifiedShuffleSplit
from xgboost import XGBClassifier
import warnings

from .export_utils import *

import deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

class TPOT(object):
    """TPOT automatically creates and optimizes machine learning pipelines using genetic programming."""

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
        scoring_function: function (default: balanced accuracy)
            Function used to evaluate the goodness of a given pipeline for the classification problem. By default, balanced class accuracy is used.

        Returns
        -------
        None

        """
        self._optimized_pipeline = None
        self._training_features = None
        self._training_classes = None
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.verbosity = verbosity

        if random_state > 0:
            random.seed(random_state)
            np.random.seed(random_state)

        self._pset = gp.PrimitiveSetTyped('MAIN', [pd.DataFrame], pd.DataFrame)

        # Machine learning model operators
        self._pset.addPrimitive(self._decision_tree, [pd.DataFrame, int, int], pd.DataFrame)
        self._pset.addPrimitive(self._random_forest, [pd.DataFrame, int, int], pd.DataFrame)
        self._pset.addPrimitive(self._logistic_regression, [pd.DataFrame, float], pd.DataFrame)
        # Temporarily remove SVC -- badly overfits on multiclass data sets
        #self._pset.addPrimitive(self._svc, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._knnc, [pd.DataFrame, int], pd.DataFrame)
        self._pset.addPrimitive(self._xgradient_boosting, [pd.DataFrame, float, int, int], pd.DataFrame)

        # Feature preprocessing operators
        self._pset.addPrimitive(self._combine_dfs, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
        self._pset.addPrimitive(self._variance_threshold, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._standard_scaler, [pd.DataFrame], pd.DataFrame)
        self._pset.addPrimitive(self._robust_scaler, [pd.DataFrame], pd.DataFrame)
        self._pset.addPrimitive(self._polynomial_features, [pd.DataFrame], pd.DataFrame)
        self._pset.addPrimitive(self._pca, [pd.DataFrame, int, int], pd.DataFrame)

        # Feature selection operators
        self._pset.addPrimitive(self._select_kbest, [pd.DataFrame, int], pd.DataFrame)
        self._pset.addPrimitive(self._select_fwe, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._select_percentile, [pd.DataFrame, int], pd.DataFrame)
        self._pset.addPrimitive(self._rfe, [pd.DataFrame, int, float], pd.DataFrame)

        # Mathematical operators
        self._pset.addPrimitive(operator.add, [int, int], int)
        self._pset.addPrimitive(operator.sub, [int, int], int)
        self._pset.addPrimitive(operator.mul, [int, int], int)
        self._pset.addPrimitive(self._div, [int, int], float)
        for val in range(0, 101):
            self._pset.addTerminal(val, int)
        for val in [100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]:
            self._pset.addTerminal(val, float)

        creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMulti)

        self._toolbox = base.Toolbox()
        self._toolbox.register('expr', gp.genHalfAndHalf, pset=self._pset, min_=1, max_=3)
        self._toolbox.register('individual', tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register('compile', gp.compile, pset=self._pset)
        self._toolbox.register('select', self._combined_selection_operator)
        self._toolbox.register('mate', gp.cxOnePoint)
        self._toolbox.register('expr_mut', gp.genFull, min_=0, max_=3)
        self._toolbox.register('mutate', self._random_mutation_operator)

        if not scoring_function:
            self.scoring_function = self._balanced_accuracy
        else:
            self.scoring_function = scoring_function

    def fit(self, features, classes, feature_names=None):
        """Fits a machine learning pipeline that maximizes classification accuracy on the provided data
        
        Uses genetic programming to optimize a machine learning pipeline that
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
            # Store the training features and classes for later use
            self._training_features = features
            self._training_classes = classes

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

            # Default guess: the most frequent class in the training data
            most_frequent_class = Counter(training_testing_data.loc[training_indices, 'class'].values).most_common(1)[0][0]
            training_testing_data.loc[:, 'guess'] = most_frequent_class

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

            stats = tools.Statistics(lambda ind: ind.fitness.values[1])
            stats.register('Minimum score', np.min)
            stats.register('Average score', np.mean)
            stats.register('Maximum score', np.max)

            verbose = (self.verbosity == 2)

            pop, _ = algorithms.eaSimple(population=pop, toolbox=self._toolbox, cxpb=self.crossover_rate,
                                         mutpb=self.mutation_rate, ngen=self.generations,
                                         stats=stats, halloffame=self.hof, verbose=verbose)

            # Store the pipeline with the highest internal testing accuracy
            top_score = 0.
            for pipeline in self.hof:
                pipeline_score = self._evaluate_individual(pipeline, training_testing_data)[1]
                if pipeline_score > top_score:
                    top_score = pipeline_score
                    self._optimized_pipeline = pipeline

            if self.verbosity == 2:
                print('')

            if self.verbosity >= 1:
                print('Best pipeline: {}'.format(self._optimized_pipeline))

        # Store the best pipeline if the optimization process is ended prematurely
        except KeyboardInterrupt:
            top_score = 0.
            for pipeline in self.hof:
                pipeline_score = self._evaluate_individual(pipeline, training_testing_data)[1]
                if pipeline_score > top_score:
                    top_score = pipeline_score
                    self._optimized_pipeline = pipeline

            if self.verbosity == 2:
                print('')

            if self.verbosity >= 1:
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
        most_frequent_class = Counter(self._training_classes).most_common(1)[0][0]
        training_testing_data.loc[:, 'guess'] = most_frequent_class

        new_col_names = {}
        for column in training_testing_data.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        training_testing_data.rename(columns=new_col_names, inplace=True)

        # Transform the tree expression in a callable function
        func = self._toolbox.compile(expr=self._optimized_pipeline)

        result = func(training_testing_data)
        
        return result.loc[result['group'] == 'testing', 'guess'].values

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
        most_frequent_class = Counter(self._training_classes).most_common(1)[0][0]
        training_testing_data.loc[:, 'guess'] = most_frequent_class

        for column in training_testing_data.columns.values:
            if type(column) != str:
                training_testing_data.rename(columns={column: str(column).zfill(10)}, inplace=True)

        return self._evaluate_individual(self._optimized_pipeline, training_testing_data)[1]


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

        # Replace all of the mathematical operators with their results. Check export_utils.py for details.
        exported_pipeline = replace_mathematical_operators(exported_pipeline)

        # Unroll the nested function calls into serial code. Check export_utils.py for details.
        exported_pipeline, pipeline_list = unroll_nested_fuction_calls(exported_pipeline)

        # Have the exported code import all of the necessary modules and functions
        pipeline_text = generate_import_code(pipeline_list)

        # Replace the function calls with their corresponding Python code. Check export_utils.py for details.
        pipeline_text += replace_function_calls(pipeline_list)

        with open(output_file_name, 'w') as output_file:
            output_file.write(pipeline_text)

    def _decision_tree(self, input_df, max_features, max_depth):
        """Fits a decision tree classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the decision tree
        max_features: int
            Number of features used to fit the decision tree; must be a positive value
        max_depth: int
            Maximum depth of the decision tree; must be a positive value

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        if max_features < 1:
            max_features = 'auto'
        elif max_features == 1:
            max_features = None
        elif max_features > len(input_df.columns) - 3:
            max_features = len(input_df.columns) - 3

        if max_depth < 1:
            max_depth = None

        return self._train_model_and_predict(input_df, DecisionTreeClassifier, max_features=max_features, max_depth=max_depth, random_state=42)

    def _random_forest(self, input_df, n_estimators, max_features):
        """Fits a random forest classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the random forest
        n_estimators: int
            Number of trees in the random forest; must be a positive value
        max_features: int
            Number of features used to fit the decision tree; must be a positive value

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        if n_estimators < 1:
            n_estimators = 1
        elif n_estimators > 500:
            n_estimators = 500

        if max_features < 1:
            max_features = 'auto'
        elif max_features == 1:
            max_features = None
        elif max_features > len(input_df.columns) - 3:
            max_features = len(input_df.columns) - 3

        return self._train_model_and_predict(input_df, RandomForestClassifier, n_estimators=n_estimators, max_features=max_features, random_state=42, n_jobs=-1)

    def _logistic_regression(self, input_df, C):
        """Fits a logistic regression classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the logistic regression classifier
        C: float
            Inverse of regularization strength; must be a positive value. Like in support vector machines, smaller values specify stronger regularization.

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        if C <= 0.:
            C = 0.0001

        return self._train_model_and_predict(input_df, LogisticRegression, C=C, random_state=42)

    def _svc(self, input_df, C):
        """Fits a C-support vector classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the C-support vector classifier
        C: float
            Penalty parameter C of the error term; must be a positive value

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        if C <= 0.:
            C = 0.0001

        return self._train_model_and_predict(input_df, SVC, C=C, random_state=42)


    def _knnc(self, input_df, n_neighbors):
        """Fits a k-nearest neighbor classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the k-nearest neighbor classifier
        n_neighbors: int
            Number of neighbors to use by default for k_neighbors queries; must be a positive value

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        training_set_size = len(input_df.loc[input_df['group'] == 'training'])

        if n_neighbors < 2:
            n_neighbors = 2
        elif n_neighbors >= training_set_size:
            n_neighbors = training_set_size - 1

        return self._train_model_and_predict(input_df, KNeighborsClassifier, n_neighbors=n_neighbors)

    def _xgradient_boosting(self, input_df, learning_rate, n_estimators, max_depth):
        """Fits the dmlc eXtreme gradient boosting classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the XGBoost classifier
        learning_rate: float
            Shrinks the contribution of each tree by learning_rate
        n_estimators: int
            The number of boosting stages to perform
        max_depth: int
            Maximum depth of the individual estimators; the maximum depth limits the number of nodes in the tree

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        if learning_rate <= 0.:
            learning_rate = 0.0001

        if n_estimators < 1:
            n_estimators = 1
        elif n_estimators > 500:
            n_estimators = 500

        if max_depth < 1:
            max_depth = None

        return self._train_model_and_predict(input_df, XGBClassifier, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, seed=42)

    def _train_model_and_predict(self, input_df, model, **kwargs):
        """Fits an arbitrary sklearn classifier model with a set of keyword parameters

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the k-neares
        model: sklearn classifier
            Input model to fit and predict on input_df
        kwargs: unpacked parameters
            Input parameters to pass to the model's constructor, does not need to be a dictionary

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        #Validate input
        #If there are no features left (i.e., only 'class', 'group', and 'guess' remain in the DF), then there is nothing to do
        if len(input_df.columns) == 3:
            return input_df

        input_df = input_df.copy()

        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1).values
        training_classes = input_df.loc[input_df['group'] == 'training', 'class'].values

        # Try to seed the random_state parameter if the model accepts it.
        try:
            clf = model(random_state=42,**kwargs)
            clf.fit(training_features, training_classes)
        except TypeError:
            clf = model(**kwargs)
            clf.fit(training_features, training_classes)

        all_features = input_df.drop(['class', 'group', 'guess'], axis=1).values
        input_df.loc[:, 'guess'] = clf.predict(all_features)

        # Also store the guesses as a synthetic feature
        sf_hash = '-'.join(sorted(input_df.columns.values))
        #Use the classifier object's class name in the synthetic feature
        sf_hash += '{}'.format(clf.__class__)
        sf_hash += '-'.join(kwargs)
        sf_identifier = 'SyntheticFeature-{}'.format(hashlib.sha224(sf_hash.encode('UTF-8')).hexdigest())
        input_df.loc[:, sf_identifier] = input_df['guess'].values

        return input_df

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

    def _rfe(self, input_df, num_features, step):
        """Uses scikit-learn's Recursive Feature Elimination to learn the subset of features that have the highest weights according to the estimator

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to perform feature selection on
        num_features: int
            The number of features to select
        step: float
            The percentage of features to drop each iteration

        Returns
        -------
        subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the `num_features` best features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values

        if step < 0.1:
            step = 0.1
        elif step >= 1.:
            step = 0.99
        if num_features < 1:
            num_features = 1
        elif num_features > len(training_features.columns):
            num_features = len(training_features.columns)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        estimator = SVC(kernel='linear')
        selector = RFE(estimator, n_features_to_select=num_features, step=step)
        try:
            selector.fit(training_features, training_class_vals)
            mask = selector.get_support(True)
            mask_cols = list(training_features.iloc[:, mask].columns) + ['guess', 'class', 'group']
            return input_df[mask_cols].copy()
        except ValueError:
            return input_df[['guess', 'class', 'group']].copy()

    def _select_percentile(self, input_df, percentile):
        """Uses scikit-learn's SelectPercentile feature selection to learn the subset of features that belong in the highest `percentile`

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to perform feature selection on
        percentile: int
            The features that belong in the top percentile to keep from the original set of features in the training data

        Returns
        -------
        subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the best features in the given `percentile`

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values

        if percentile < 0:
            percentile = 0
        elif percentile > 100:
            percentile = 100

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        with warnings.catch_warnings():
            # Ignore warnings about constant features
            warnings.simplefilter('ignore', category=UserWarning)

            selector = SelectPercentile(f_classif, percentile=percentile)
            selector.fit(training_features, training_class_vals)
            mask = selector.get_support(True)

        mask_cols = list(training_features.iloc[:, mask].columns) + ['guess', 'class', 'group']
        return input_df[mask_cols].copy()

    def _select_kbest(self, input_df, k):
        """Uses scikit-learn's SelectKBest feature selection to learn the subset of features that have the highest score according to some scoring function

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to perform feature selection on
        k: int
            The top k features to keep from the original set of features in the training data

        Returns
        -------
        subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the `k` best features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values

        if k < 1:
            k = 1
        elif k >= len(training_features.columns):
            k = 'all'

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        with warnings.catch_warnings():
            # Ignore warnings about constant features
            warnings.simplefilter('ignore', category=UserWarning)

            selector = SelectKBest(f_classif, k=k)
            selector.fit(training_features, training_class_vals)
            mask = selector.get_support(True)

        mask_cols = list(training_features.iloc[:, mask].columns) + ['guess', 'class', 'group']
        return input_df[mask_cols].copy()

    def _select_fwe(self, input_df, alpha):
        """Uses scikit-learn's SelectFwe feature selection to subset the features according to p-values corresponding to family-wise error rate

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to perform feature selection on
        alpha: float in the range [0.001, 0.05]
            The highest uncorrected p-value for features to keep

        Returns
        -------
        subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the 'best' features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values

        # forcing  0.001 <= alpha <= 0.05
        if alpha > 0.05:
            alpha = 0.05
        elif alpha <= 0.001:
            alpha = 0.001


        if len(training_features.columns.values) == 0:
            return input_df.copy()

        with warnings.catch_warnings():
            # Ignore warnings about constant features
            warnings.simplefilter('ignore', category=UserWarning)

            selector = SelectFwe(f_classif, alpha=alpha)
            selector.fit(training_features, training_class_vals)
            mask = selector.get_support(True)

        mask_cols = list(training_features.iloc[:, mask].columns) + ['guess', 'class', 'group']
        return input_df[mask_cols].copy()

    def _variance_threshold(self, input_df, threshold):
        """Uses scikit-learn's VarianceThreshold feature selection to learn the subset of features that pass the threshold

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to perform feature selection on
        threshold: float
            The variance threshold that removes features that fall under the threshold

        Returns
        -------
        subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the features that are above the variance threshold

        """

        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)

        selector = VarianceThreshold(threshold=threshold)
        try:
            selector.fit(training_features)
        except ValueError:
            # None features are above the variance threshold
            return input_df[['guess', 'class', 'group']].copy()

        mask = selector.get_support(True)
        mask_cols = list(training_features.iloc[:, mask].columns) + ['guess', 'class', 'group']
        return input_df[mask_cols].copy()

    def _standard_scaler(self, input_df):
        """Uses scikit-learn's StandardScaler to scale the features by removing their mean and scaling to unit variance

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale

        Returns
        -------
        scaled_df: pandas.DataFrame {n_samples, n_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the scaled features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # The scaler must be fit on only the training data
        scaler = StandardScaler()
        scaler.fit(training_features.values.astype(np.float64))
        scaled_features = scaler.transform(input_df.drop(['class', 'group', 'guess'], axis=1).values.astype(np.float64))

        for col_num, column in enumerate(input_df.drop(['class', 'group', 'guess'], axis=1).columns.values):
            input_df.loc[:, column] = scaled_features[:, col_num]

        return input_df.copy()

    def _robust_scaler(self, input_df):
        """Uses scikit-learn's RobustScaler to scale the features using statistics that are robust to outliers

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale

        Returns
        -------
        scaled_df: pandas.DataFrame {n_samples, n_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the scaled features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # The scaler must be fit on only the training data
        scaler = RobustScaler()
        scaler.fit(training_features.values.astype(np.float64))
        scaled_features = scaler.transform(input_df.drop(['class', 'group', 'guess'], axis=1).values.astype(np.float64))

        for col_num, column in enumerate(input_df.drop(['class', 'group', 'guess'], axis=1).columns.values):
            input_df.loc[:, column] = scaled_features[:, col_num]

        return input_df.copy()

    def _polynomial_features(self, input_df):
        """Uses scikit-learn's PolynomialFeatures to construct new degree-2 polynomial features from the existing feature set

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_constructed_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the constructed features

        """

        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()
        elif len(training_features.columns.values) > 700:
            # Too many features to produce - skip this operator
            return input_df.copy()

        # The feature constructor must be fit on only the training data
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly.fit(training_features.values.astype(np.float64))
        constructed_features = poly.transform(input_df.drop(['class', 'group', 'guess'], axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=constructed_features)
        modified_df['class'] = input_df['class'].values
        modified_df['group'] = input_df['group'].values
        modified_df['guess'] = input_df['guess'].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()

    def _pca(self, input_df, n_components, iterated_power):
        """Uses scikit-learn's RandomizedPCA to transform the feature set

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale
        n_components: int
            The number of components to keep
        iterated_power: int
            Number of iterations for the power method. [1, 10]


        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_components + ['guess', 'group', 'class']}
            Returns a DataFrame containing the transformed features

        """

        if n_components < 1:
            n_components = 1
        elif n_components >= len(input_df.columns.values) - 3:
            n_components = None

        #Thresholding iterated_power [1,10]
        if iterated_power < 1:
            iterated_power = 1
        elif iterated_power > 10:
            iterated_power = 10


        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # PCA must be fit on only the training data
        pca = RandomizedPCA(n_components=n_components, iterated_power=iterated_power)
        pca.fit(training_features.values.astype(np.float64))
        transformed_features = pca.transform(input_df.drop(['class', 'group', 'guess'], axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=transformed_features)
        modified_df['class'] = input_df['class'].values
        modified_df['group'] = input_df['group'].values
        modified_df['guess'] = input_df['guess'].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()

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
            func = self._toolbox.compile(expr=individual)

            # Count the number of pipeline operators as a measure of pipeline complexity
            operator_count = 0
            for i in range(len(individual)):
                node = individual[i]
                if type(node) is deap.gp.Terminal:
                    continue
                if type(node) is deap.gp.Primitive and node.name in ['add', 'sub', 'mul', '_div', '_combine_dfs']:
                    continue

                operator_count += 1

            result = func(training_testing_data)
            result = result[result['group'] == 'testing']
            resulting_score = self.scoring_function(result)

        except MemoryError:
            # Throw out GP expressions that are too large to be compiled in Python
            return 5000., 0.
        except Exception:
            # Catch-all: Do not allow one pipeline that crashes to cause TPOT to crash
            # Instead, assign the crashing pipeline a poor fitness
            return 5000., 0.

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
            this_class_accuracy = len(result[(result['guess'] == this_class) \
                & (result['class'] == this_class)])\
                / float(len(result[result['class'] == this_class]))
            all_class_accuracies.append(this_class_accuracy)

        balanced_accuracy = np.mean(all_class_accuracies)

        return balanced_accuracy

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
        roll = random.random()
        if roll <= 0.333333:
            return gp.mutUniform(individual, expr=self._toolbox.expr_mut, pset=self._pset)
        elif roll <= 0.666666:
            return gp.mutInsert(individual, pset=self._pset)
        else:
            return gp.mutShrink(individual)

def main():
    """Main function that is called when TPOT is run on the command line"""
    from _version import __version__

    parser = argparse.ArgumentParser(description='A Python tool that'
            ' automatically creates and optimizes machine learning pipelines'
            ' using genetic programming.')

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
            raise argparse.ArgumentTypeError('invalid int value: \'{}\''.format(value))
        if value < 0:
            raise argparse.ArgumentTypeError('invalid positive int value: \'{}\''.format(value))
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
            raise argparse.ArgumentTypeError('invalid float value: \'{}\''.format(value))
        if value < 0.0 or value > 1.0:
            raise argparse.ArgumentTypeError('invalid float value: \'{}\''.format(value))
        return value

    parser.add_argument('INPUT_FILE', type=str, help='Data file to optimize the pipeline on; ensure that the class column is labeled as "class"')

    parser.add_argument('-is', action='store', dest='INPUT_SEPARATOR', default='\t',
                        type=str, help='Character used to separate columns in the input file')

    parser.add_argument('-o', action='store', dest='OUTPUT_FILE', default='',
                        type=str, help='File to export the final optimized pipeline')

    parser.add_argument('-g', action='store', dest='GENERATIONS', default=100,
                        type=positive_integer, help='Number of generations to run pipeline optimization')

    parser.add_argument('-p', action='store', dest='POPULATION_SIZE', default=100,
                        type=positive_integer, help='Number of individuals in the GP population')

    parser.add_argument('-mr', action='store', dest='MUTATION_RATE', default=0.9,
                        type=float_range, help='GP mutation rate in the range [0.0, 1.0]')

    parser.add_argument('-xr', action='store', dest='CROSSOVER_RATE', default=0.05,
                        type=float_range, help='GP crossover rate in the range [0.0, 1.0]')

    parser.add_argument('-s', action='store', dest='RANDOM_STATE', default=0,
                        type=int, help='Random number generator seed for reproducibility')

    parser.add_argument('-v', action='store', dest='VERBOSITY', default=1, choices=[0, 1, 2],
                        type=int, help='How much information TPOT communicates while it is running; 0 = none, 1 = minimal, 2 = all')

    parser.add_argument('--version', action='version', version='TPOT v{version}'.format(version=__version__))

    args = parser.parse_args()

    if args.VERBOSITY >= 2:
        print('\nTPOT settings:')
        for arg in sorted(args.__dict__):
            print('{}\t=\t{}'.format(arg, args.__dict__[arg]))
        print('')

    input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR)

    if 'Class' in input_data.columns.values:
        input_data.rename(columns={'Class': 'class'}, inplace=True)

    if args.RANDOM_STATE > 0:
        RANDOM_STATE = args.RANDOM_STATE
    else:
        RANDOM_STATE = None

    training_indices, testing_indices = next(iter(StratifiedShuffleSplit(input_data['class'].values,
                                                                         n_iter=1,
                                                                         train_size=0.75,
                                                                         test_size=0.25,
                                                                         random_state=RANDOM_STATE)))

    training_features = input_data.loc[training_indices].drop('class', axis=1).values
    training_classes = input_data.loc[training_indices, 'class'].values

    testing_features = input_data.loc[testing_indices].drop('class', axis=1).values
    testing_classes = input_data.loc[testing_indices, 'class'].values

    tpot = TPOT(generations=args.GENERATIONS, population_size=args.POPULATION_SIZE,
                mutation_rate=args.MUTATION_RATE, crossover_rate=args.CROSSOVER_RATE,
                random_state=args.RANDOM_STATE, verbosity=args.VERBOSITY)

    tpot.fit(training_features, training_classes)

    if args.VERBOSITY >= 1:
        print('\nTraining accuracy: {}'.format(tpot.score(training_features, training_classes)))
        print('Testing accuracy: {}'.format(tpot.score(testing_features, testing_classes)))

    if args.OUTPUT_FILE != '':
        tpot.export(args.OUTPUT_FILE)


if __name__ == '__main__':
    main()
