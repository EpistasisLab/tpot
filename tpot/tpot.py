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
import hashlib
import inspect
import sys
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import FeatureAgglomeration
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, RFE, SelectFwe, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures, Binarizer
from sklearn.decomposition import RandomizedPCA, FastICA
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.cross_validation import train_test_split

import warnings
from update_checker import update_check

from ._version import __version__
from .export_utils import unroll_nested_fuction_calls, generate_import_code, replace_function_calls
from .decorators import _gp_new_generation

import deap
from deap import algorithms, base, creator, tools, gp

from tqdm import tqdm


class Bool(object):
    """Boolean class used for deap due to deap's poor handling of ints and booleans"""
    pass


class TPOT(object):
    """TPOT automatically creates and optimizes machine learning pipelines using genetic programming."""

    update_checked = False

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

        # Do not prompt the user to update during this session if they ever disabled the update check
        if disable_update_check:
            TPOT.update_checked = True

        # Prompt the user if their version is out of date
        if not disable_update_check and not TPOT.update_checked:
            update_check('tpot', __version__)
            TPOT.update_checked = True

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

        # Columns to always ignore when in an operator
        self.non_feature_columns = ['class', 'group', 'guess']

        if random_state > 0:
            random.seed(random_state)
            np.random.seed(random_state)

        self._pset = gp.PrimitiveSetTyped('MAIN', [pd.DataFrame], pd.DataFrame)

        # Rename pipeline input to "input_df"
        self._pset.renameArguments(ARG0='input_df')

        # Machine learning model operators
        self._pset.addPrimitive(self._decision_tree, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._random_forest, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._ada_boost, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._logistic_regression, [pd.DataFrame, float, int, Bool], pd.DataFrame)
        self._pset.addPrimitive(self._knnc, [pd.DataFrame, int, int], pd.DataFrame)
        self._pset.addPrimitive(self._gradient_boosting, [pd.DataFrame, float, float, float], pd.DataFrame)
        self._pset.addPrimitive(self._bernoulli_nb, [pd.DataFrame, float, float], pd.DataFrame)
        self._pset.addPrimitive(self._extra_trees, [pd.DataFrame, int, float, float], pd.DataFrame)
        self._pset.addPrimitive(self._gaussian_nb, [pd.DataFrame], pd.DataFrame)
        self._pset.addPrimitive(self._multinomial_nb, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._linear_svc, [pd.DataFrame, float, int, Bool], pd.DataFrame)
        self._pset.addPrimitive(self._passive_aggressive, [pd.DataFrame, float, int], pd.DataFrame)

        # Feature preprocessing operators
        self._pset.addPrimitive(self._combine_dfs, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
        self._pset.addPrimitive(self._variance_threshold, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._standard_scaler, [pd.DataFrame], pd.DataFrame)
        self._pset.addPrimitive(self._robust_scaler, [pd.DataFrame], pd.DataFrame)
        self._pset.addPrimitive(self._min_max_scaler, [pd.DataFrame], pd.DataFrame)
        self._pset.addPrimitive(self._max_abs_scaler, [pd.DataFrame], pd.DataFrame)
        self._pset.addPrimitive(self._binarizer, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._polynomial_features, [pd.DataFrame], pd.DataFrame)
        self._pset.addPrimitive(self._pca, [pd.DataFrame, int], pd.DataFrame)
        self._pset.addPrimitive(self._rbf, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._fast_ica, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._feat_agg, [pd.DataFrame, int, int, int], pd.DataFrame)
        self._pset.addPrimitive(self._nystroem, [pd.DataFrame, int, float, int], pd.DataFrame)
        self._pset.addPrimitive(self._zero_count, [pd.DataFrame], pd.DataFrame)

        # Feature selection operators
        self._pset.addPrimitive(self._select_kbest, [pd.DataFrame, int], pd.DataFrame)
        self._pset.addPrimitive(self._select_fwe, [pd.DataFrame, float], pd.DataFrame)
        self._pset.addPrimitive(self._select_percentile, [pd.DataFrame, int], pd.DataFrame)
        self._pset.addPrimitive(self._rfe, [pd.DataFrame, int, float], pd.DataFrame)

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

        self.hof = None

        if not scoring_function:
            self.scoring_function = self._balanced_accuracy
        else:
            self.scoring_function = scoring_function

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

    def _decision_tree(self, input_df, min_weight):
        """Fits a decision tree classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the decision tree
        min_weight_fraction_leaf: float
            The minimum weighted fraction of the input samples required to be at a leaf node.

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """

        min_weight = min(0.5, max(0., min_weight))

        return self._train_model_and_predict(input_df, DecisionTreeClassifier,
            min_weight_fraction_leaf=min_weight, random_state=42)

    def _random_forest(self, input_df, min_weight):
        """Fits a random forest classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the random forest
        min_weight_fraction_leaf: float
            The minimum weighted fraction of the input samples required to be at a leaf node.

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """

        min_weight = min(0.5, max(0., min_weight))

        return self._train_model_and_predict(input_df, RandomForestClassifier,
            min_weight_fraction_leaf=min_weight, n_estimators=500, random_state=42, n_jobs=-1)

    def _ada_boost(self, input_df, learning_rate):
        """Fits an AdaBoost classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the classifier
        learning_rate: float
            Learning rate shrinks the contribution of each classifier by learning_rate.

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        learning_rate = min(1., max(0.0001, learning_rate))

        return self._train_model_and_predict(input_df, AdaBoostClassifier,
            learning_rate=learning_rate, n_estimators=500, random_state=42)

    def _bernoulli_nb(self, input_df, alpha, binarize):
        """Fits a Bernoulli Naive Bayes classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the classifier
        alpha: float
            Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        binarize: float
            Threshold for binarizing (mapping to booleans) of sample features.

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """

        return self._train_model_and_predict(input_df, BernoulliNB, alpha=alpha,
            binarize=binarize, fit_prior=True)

    def _extra_trees(self, input_df, criterion, max_features, min_weight):
        """Fits an Extra Trees Classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the classifier
        criterion: int
            Integer that is used to select from the list of valid criteria,
            either 'gini', or 'entropy'
        max_features: float
            The number of features to consider when looking for the best split
        min_weight_fraction_leaf: float
            The minimum weighted fraction of the input samples required to be at a leaf node.

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        # Select criterion string from list of valid parameters
        criterion_values = ['gini', 'entropy']
        criterion_selection = criterion_values[criterion % len(criterion_values)]

        min_weight = min(0.5, max(0., min_weight))
        max_features = min(1., max(0., max_features))

        return self._train_model_and_predict(input_df, ExtraTreesClassifier,
            criterion=criterion_selection, max_features=max_features, min_weight_fraction_leaf=min_weight,
            n_estimators=500, random_state=42)

    def _gaussian_nb(self, input_df):
        """Fits a Gaussian Naive Bayes Classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the classifier

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        return self._train_model_and_predict(input_df, GaussianNB)

    def _multinomial_nb(self, input_df, alpha):
        """Fits a Naive Bayes classifier for multinomial models

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the classifier
        alpha: float
            Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        return self._train_model_and_predict(input_df, MultinomialNB, alpha=alpha, fit_prior=True)

    def _linear_svc(self, input_df, C, penalty, dual):
        """Fits a Linear Support Vector Classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the classifier
        C: float
            Penalty parameter C of the error term.
        penalty: int
            Integer used to specify the norm used in the penalization (l1 or l2)
        dual: bool
            Select the algorithm to either solve the dual or primal optimization problem.

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        penalty_values = ['l1', 'l2']
        penalty_selection = penalty_values[penalty % len(penalty_values)]

        C = min(25., max(0.0001, C))

        if penalty_selection == 'l1':
            dual = False

        return self._train_model_and_predict(input_df, LinearSVC, C=C,
            penalty=penalty_selection, dual=dual, random_state=42)

    def _passive_aggressive(self, input_df, C, loss):
        """Fits a Linear Support Vector Classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the classifier
        C: float
            Penalty parameter C of the error term.
        loss: int
            Integer used to determine the loss function (either 'hinge' or 'squared_hinge')

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        loss_values = ['hinge', 'squared_hinge']
        loss_selection = loss_values[loss % len(loss_values)]

        C = min(1., max(0.0001, C))

        return self._train_model_and_predict(input_df, PassiveAggressiveClassifier,
            C=C, loss=loss_selection, fit_intercept=True, random_state=42)

    def _logistic_regression(self, input_df, C, penalty, dual):
        """Fits a logistic regression classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the logistic regression classifier
        C: float
            Inverse of regularization strength; must be a positive value. Like in support vector machines, smaller values specify stronger regularization.
        penalty: int
            Integer used to specify the norm used in the penalization (l1 or l2)
        dual: bool
            Select the algorithm to either solve the dual or primal optimization problem.

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        C = min(50., max(0.0001, C))

        penalty_values = ['l1', 'l2']
        penalty_selection = penalty_values[penalty % len(penalty_values)]

        if penalty_selection == 'l1':
            dual = False

        return self._train_model_and_predict(input_df, LogisticRegression, C=C,
            penalty=penalty_selection, dual=dual, random_state=42)

    def _knnc(self, input_df, n_neighbors, weights):
        """Fits a k-nearest neighbor classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the k-nearest neighbor classifier
        n_neighbors: int
            Number of neighbors to use by default for k_neighbors queries; must be a positive value
        weights: int
            Selects a value from the list: ['uniform', 'distance']

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        training_set_size = len(input_df.loc[input_df['group'] == 'training'])
        n_neighbors = max(min(training_set_size - 1, n_neighbors), 2)

        weights_values = ['uniform', 'distance']
        weights_selection = weights_values[weights % len(weights_values)]

        return self._train_model_and_predict(input_df, KNeighborsClassifier,
            n_neighbors=n_neighbors, weights=weights_selection)

    def _gradient_boosting(self, input_df, learning_rate, max_features, min_weight):
        """Fits the sklearn GradientBoostingClassifier classifier

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame for fitting the XGBoost classifier
        learning_rate: float
            Shrinks the contribution of each tree by learning_rate
        max_features: float
            Maximum number of features to use (proportion of total features)
        min_weight_fraction_leaf: float
            The minimum weighted fraction of the input samples required to be at a leaf node.

        Returns
        -------
        input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
            Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
            Also adds the classifiers's predictions as a 'SyntheticFeature' column.

        """
        learning_rate = min(1., max(learning_rate, 0.0001))
        max_features = min(1., max(0., learning_rate))
        min_weight = min(0.5, max(0., min_weight))

        return self._train_model_and_predict(input_df, GradientBoostingClassifier,
            learning_rate=learning_rate, n_estimators=500,
            max_features=max_features, random_state=42, min_weight_fraction_leaf=min_weight)

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
        # If there are no features left (i.e., only 'class', 'group', and 'guess' remain in the DF), then there is nothing to do
        if len(input_df.columns) == 3:
            return input_df

        input_df = input_df.copy()

        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1).values
        training_classes = input_df.loc[input_df['group'] == 'training', 'class'].values

        # Try to seed the random_state parameter if the model accepts it.
        try:
            clf = model(random_state=42, **kwargs)
            clf.fit(training_features, training_classes)
        except TypeError:
            clf = model(**kwargs)
            clf.fit(training_features, training_classes)

        all_features = input_df.drop(self.non_feature_columns, axis=1).values
        input_df.loc[:, 'guess'] = clf.predict(all_features)

        # Also store the guesses as a synthetic feature
        sf_hash = '-'.join(sorted(input_df.columns.values))
        # Use the classifier object's class name in the synthetic feature
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
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values

        step = max(min(0.99, step), 0.1)

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
            mask_cols = list(training_features.iloc[:, mask].columns) + self.non_feature_columns
            return input_df[mask_cols].copy()
        except ValueError:
            return input_df[self.non_feature_columns].copy()

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
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values

        percentile = max(min(100, percentile), 0)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        with warnings.catch_warnings():
            # Ignore warnings about constant features
            warnings.simplefilter('ignore', category=UserWarning)

            selector = SelectPercentile(f_classif, percentile=percentile)
            selector.fit(training_features, training_class_vals)
            mask = selector.get_support(True)

        mask_cols = list(training_features.iloc[:, mask].columns) + self.non_feature_columns
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
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)
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

        mask_cols = list(training_features.iloc[:, mask].columns) + self.non_feature_columns
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
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values

        # Clamp alpha in the range [0.001, 0.05]
        alpha = max(min(0.05, alpha), 0.001)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        with warnings.catch_warnings():
            # Ignore warnings about constant features
            warnings.simplefilter('ignore', category=UserWarning)

            selector = SelectFwe(f_classif, alpha=alpha)
            selector.fit(training_features, training_class_vals)
            mask = selector.get_support(True)

        mask_cols = list(training_features.iloc[:, mask].columns) + self.non_feature_columns
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
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        selector = VarianceThreshold(threshold=threshold)
        try:
            selector.fit(training_features)
        except ValueError:
            # None features are above the variance threshold
            return input_df[['guess', 'class', 'group']].copy()

        mask = selector.get_support(True)
        mask_cols = list(training_features.iloc[:, mask].columns) + self.non_feature_columns
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
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # The scaler must be fit on only the training data
        scaler = StandardScaler(copy=False)
        scaler.fit(training_features.values.astype(np.float64))
        scaled_features = scaler.transform(input_df.drop(self.non_feature_columns, axis=1).values.astype(np.float64))

        for col_num, column in enumerate(input_df.drop(self.non_feature_columns, axis=1).columns.values):
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
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # The scaler must be fit on only the training data
        scaler = RobustScaler(copy=False)
        scaler.fit(training_features.values.astype(np.float64))
        scaled_features = scaler.transform(input_df.drop(self.non_feature_columns, axis=1).values.astype(np.float64))

        for col_num, column in enumerate(input_df.drop(self.non_feature_columns, axis=1).columns.values):
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
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()
        elif len(training_features.columns.values) > 700:
            # Too many features to produce - skip this operator
            return input_df.copy()

        # The feature constructor must be fit on only the training data
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly.fit(training_features.values.astype(np.float64))
        constructed_features = poly.transform(input_df.drop(self.non_feature_columns, axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=constructed_features)

        for non_feature_column in self.non_feature_columns:
            modified_df[non_feature_column] = input_df[non_feature_column].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()

    def _min_max_scaler(self, input_df):
        """Uses scikit-learn's MinMaxScaler to transform all of the features by scaling them to the range [0, 1]

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the scaled features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # The feature scaler must be fit on only the training data
        mm_scaler = MinMaxScaler(copy=False)
        mm_scaler.fit(training_features.values.astype(np.float64))
        scaled_features = mm_scaler.transform(input_df.drop(self.non_feature_columns, axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=scaled_features)

        for non_feature_column in self.non_feature_columns:
            modified_df[non_feature_column] = input_df[non_feature_column].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()

    def _max_abs_scaler(self, input_df):
        """Uses scikit-learn's MaxAbsScaler to transform all of the features by scaling them to [0, 1] relative to the feature's maximum value

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the scaled features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # The feature scaler must be fit on only the training data
        ma_scaler = MaxAbsScaler(copy=False)
        ma_scaler.fit(training_features.values.astype(np.float64))
        scaled_features = ma_scaler.transform(input_df.drop(self.non_feature_columns, axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=scaled_features)

        for non_feature_column in self.non_feature_columns:
            modified_df[non_feature_column] = input_df[non_feature_column].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()

    def _binarizer(self, input_df, threshold):
        """Uses scikit-learn's Binarizer to binarize all of the features, setting any feature >`threshold` to 1 and all others to 0

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale
        threshold: float
            Feature values below or equal to this value are replaced by 0, above it by 1

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the binarized features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # The binarizer must be fit on only the training data
        binarizer = Binarizer(copy=False, threshold=threshold)
        binarizer.fit(training_features.values.astype(np.float64))
        binarized_features = binarizer.transform(input_df.drop(self.non_feature_columns, axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=binarized_features)

        for non_feature_column in self.non_feature_columns:
            modified_df[non_feature_column] = input_df[non_feature_column].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()

    def _pca(self, input_df, iterated_power):
        """Uses scikit-learn's RandomizedPCA to transform the feature set

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale
        iterated_power: int
            Number of iterations for the power method. [1, 10]

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_components + ['guess', 'group', 'class']}
            Returns a DataFrame containing the transformed features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # Thresholding iterated_power [1, 10]
        iterated_power = min(10, max(1, iterated_power))

        # PCA must be fit on only the training data
        pca = RandomizedPCA(iterated_power=iterated_power, copy=False)
        pca.fit(training_features.values.astype(np.float64))
        transformed_features = pca.transform(input_df.drop(self.non_feature_columns, axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=transformed_features)

        for non_feature_column in self.non_feature_columns:
            modified_df[non_feature_column] = input_df[non_feature_column].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()

    def _rbf(self, input_df, gamma):
        """Uses scikit-learn's RBFSampler to transform the feature set

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale
        gamma: float
            Parameter of RBF kernel: exp(-gamma * x^2)

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_components + ['guess', 'group', 'class']}
            Returns a DataFrame containing the transformed features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # RBF must be fit on only the training data
        rbf = RBFSampler(gamma=gamma)
        rbf.fit(training_features.values.astype(np.float64))
        transformed_features = rbf.transform(input_df.drop(self.non_feature_columns, axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=transformed_features)

        for non_feature_column in self.non_feature_columns:
            modified_df[non_feature_column] = input_df[non_feature_column].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()

    def _fast_ica(self, input_df, tol):
        """Uses scikit-learn's FastICA to transform the feature set

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale
        tol: float
            Tolerance on update at each iteration.

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_components + ['guess', 'group', 'class']}
            Returns a DataFrame containing the transformed features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # Ensure that tol does not get to be too small
        tol = max(tol, 0.0001)

        ica = FastICA(tol=tol, random_state=42)

        # Ignore convergence warnings during GP
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)

            ica.fit(training_features.values.astype(np.float64))

        transformed_features = ica.transform(input_df.drop(self.non_feature_columns, axis=1).values.astype(np.float64))
        modified_df = pd.DataFrame(data=transformed_features)

        for non_feature_column in self.non_feature_columns:
            modified_df[non_feature_column] = input_df[non_feature_column].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()

    def _feat_agg(self, input_df, n_clusters, affinity, linkage):
        """Uses scikit-learn's FeatureAgglomeration to transform the feature set

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale
        n_clusters: int
            The number of clusters to find.
        affinity: int
            Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
            "manhattan", "cosine", or "precomputed". If linkage is "ward", only
            "euclidean" is accepted.
            Input integer is used to select one of the above strings.
        linkage: int
            Can be one of the following values:
                "ward", "complete", "average"
            Input integer is used to select one of the above strings.

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_clusters + ['guess', 'group', 'class']}
            Returns a DataFrame containing the transformed features
        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        if n_clusters < 1:
            n_clusters = 1

        affinity_types = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
        linkage_types = ['ward', 'complete', 'average']

        linkage_name = linkage_types[linkage % len(linkage_types)]

        if linkage_name == 'ward':
            affinity_name = 'euclidean'
        else:
            affinity_name = affinity_types[affinity % len(affinity_types)]

        fa = FeatureAgglomeration(n_clusters=n_clusters, affinity=affinity_name, linkage=linkage_name)
        fa.fit(training_features.values.astype(np.float64))

        clustered_features = fa.transform(input_df.drop(self.non_feature_columns, axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=clustered_features)

        for non_feature_column in self.non_feature_columns:
            modified_df[non_feature_column] = input_df[non_feature_column].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()

    def _nystroem(self, input_df, kernel, gamma, n_components):
        """
        Uses scikit-learn's Nystroem to approximate a kernel map using a subset
        of the training data. Constructs an approximate feature map for an
        arbitrary kernel using a subset of the data as basis.

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale
        kernel: int
            Kernel type is selected from scikit-learn's provided types:
                'sigmoid', 'polynomial', 'additive_chi2', 'poly', 'laplacian', 'cosine', 'linear', 'rbf', 'chi2'

            Input integer is used to select one of the above strings.
        gamma: float
            Gamma parameter for the kernels.
        n_components: int
            The number of components to keep

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_components + ['guess', 'group', 'class']}
            Returns a DataFrame containing the transformed features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(self.non_feature_columns, axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        if n_components < 1:
            n_components = 1
        else:
            n_components = min(n_components, len(training_features.columns.values), len(training_features))

        # Pulled from sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS
        kernel_types = ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid']
        kernel_name = kernel_types[kernel % len(kernel_types)]

        nys = Nystroem(kernel=kernel_name, gamma=gamma, n_components=n_components)
        nys.fit(training_features.values.astype(np.float64))
        transformed_features = nys.transform(input_df.drop(self.non_feature_columns, axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=transformed_features)

        for non_feature_column in self.non_feature_columns:
            modified_df[non_feature_column] = input_df[non_feature_column].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()

    def _zero_count(self, input_df):
        """Adds virtual features for the number of zeros per row, and number of non-zeros per row.

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Returns a DataFrame containing the new virtual features

        """
        feature_cols_only = input_df.drop(self.non_feature_columns, axis=1)
        num_features = len(feature_cols_only.columns.values)

        if num_features == 0:
            return input_df.copy()

        modified_df = input_df.copy()
        modified_df['non_zero'] = feature_cols_only.apply(lambda row: np.count_nonzero(row), axis=1).astype(np.float64)
        modified_df['zero_col'] = feature_cols_only.apply(lambda row: (num_features - np.count_nonzero(row)), axis=1).astype(np.float64)

        return modified_df.copy()

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
