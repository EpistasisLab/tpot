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

import numpy as np
import argparse
from sklearn.model_selection import train_test_split

from .tpot import TPOTClassifier, TPOTRegressor
from ._version import __version__


def positive_integer(value):
    """Ensures that the provided value is a positive integer. Throws an exception otherwise.

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
    """Ensures that the provided value is a float integer in the range [0., 1.]. Throws an exception otherwise.

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
    parser = argparse.ArgumentParser(description='A Python tool that '
        'automatically creates and optimizes machine learning pipelines using '
        'genetic programming.', add_help=False)

    parser.add_argument('INPUT_FILE', type=str, help='Data file to optimize the '
        'pipeline on; ensure that the class label column is labeled as "class".')

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    parser.add_argument('-is', action='store', dest='INPUT_SEPARATOR', default='\t',
        type=str, help='Character used to separate columns in the input file.')

    parser.add_argument('-target', action='store', dest='TARGET_NAME', default='class',
        type=str, help='Name of the target column in the input file.')

    parser.add_argument('-mode', action='store', dest='TPOT_MODE',
        choices=['classification', 'regression'], default='classification', type=str,
        help='Whether TPOT is being used for a classification or regression problem.')

    parser.add_argument('-o', action='store', dest='OUTPUT_FILE', default='',
        type=str, help='File to export the final optimized pipeline.')

    parser.add_argument('-g', action='store', dest='GENERATIONS', default=100,
        type=positive_integer, help='Number of generations to run pipeline '
        'optimization over.\nGenerally, TPOT will work better when '
        'you give it more generations (and therefore time) to optimize over. '
        'TPOT will evaluate GENERATIONS x POPULATION_SIZE number of pipelines in total.')

    parser.add_argument('-p', action='store', dest='POPULATION_SIZE', default=100,
        type=positive_integer, help='Number of individuals in the GP population.\n'
        'Generally, TPOT will work better when you give it more individuals '
        '(and therefore time) to optimize over. TPOT will evaluate '
        'GENERATIONS x POPULATION_SIZE number of pipelines in total.')

    parser.add_argument('-mr', action='store', dest='MUTATION_RATE', default=0.9,
        type=float_range, help='GP mutation rate in the range [0.0, 1.0]. We '
        'recommend using the default parameter unless you '
        'understand how the mutation rate affects GP algorithms.')

    parser.add_argument('-xr', action='store', dest='CROSSOVER_RATE', default=0.05,
        type=float_range, help='GP crossover rate in the range [0.0, 1.0]. We '
        'recommend using the default parameter unless you '
        'understand how the crossover rate affects GP algorithms.')

    parser.add_argument('-cv', action='store', dest='NUM_CV_FOLDS', default=3,
        type=int, help='The number of folds to evaluate each pipeline over in '
        'k-fold cross-validation during the TPOT pipeline optimization process.')

    parser.add_argument('-scoring', action='store', dest='SCORING_FN', default=None,
        type=str, help='Function used to evaluate the quality of a given pipeline for '
        'the problem. By default, balanced accuracy is used for classification and mean '
        'squared error is used for regression. '
        'TPOT assumes that any function with "error" or "loss" in the name is meant to '
        'be minimized, whereas any other functions will be maximized. '
        'Offers the same options as cross_val_score: '
        '"accuracy", "adjusted_rand_score", "average_precision", "f1", "f1_macro", '
        '"f1_micro", "f1_samples", "f1_weighted", "log_loss", "mean_absolute_error", '
        '"mean_squared_error", "median_absolute_error", "precision", "precision_macro", '
        '"precision_micro", "precision_samples", "precision_weighted", "r2", "recall", '
        '"recall_macro", "recall_micro", "recall_samples", "recall_weighted", "roc_auc"')

    parser.add_argument('-maxtime', action='store', dest='MAX_TIME_MINS', default=None,
        type=int, help='How many minutes TPOT has to optimize the pipeline. This '
        'setting will override the GENERATIONS parameter '
        'and allow TPOT to run until it runs out of time.')

    parser.add_argument('-maxeval', action='store', dest='MAX_EVAL_MINS', default=5,
        type=float, help='How many minutes TPOT has to evaluate a single pipeline. '
        'Setting this parameter to higher values will allow TPOT to explore more complex '
        'pipelines but will also allow TPOT to run longer.')

    parser.add_argument('-s', action='store', dest='RANDOM_STATE', default=None,
        type=int, help='Random number generator seed for reproducibility. Set '
        'this seed if you want your TPOT run to be reproducible with the same '
        'seed and data set in the future.')

    parser.add_argument('-v', action='store', dest='VERBOSITY', default=1,
        choices=[0, 1, 2, 3], type=int, help='How much information TPOT '
        'communicates while it is running: 0 = none, 1 = minimal, 2 = high, 3 = all.')

    parser.add_argument('--no-update-check', action='store_true',
        dest='DISABLE_UPDATE_CHECK', default=False,
        help='Flag indicating whether the TPOT version checker should be disabled.')

    parser.add_argument('--version', action='version',
        version='TPOT {version}'.format(version=__version__),
        help='Show TPOT\'s version number and exit.')

    args = parser.parse_args()

    if args.VERBOSITY >= 2:
        print('\nTPOT settings:')
        for arg in sorted(args.__dict__):
            arg_val = args.__dict__[arg]
            if arg == 'DISABLE_UPDATE_CHECK':
                continue
            elif arg == 'SCORING_FN' and args.__dict__[arg] is None:
                if args.TPOT_MODE == 'classification':
                    arg_val = 'balanced_accuracy'
                else:
                    arg_val = 'mean_squared_error'
            print('{}\t=\t{}'.format(arg, arg_val))
        print('')

    input_data = np.recfromcsv(args.INPUT_FILE, delimiter=args.INPUT_SEPARATOR, dtype=np.float64, case_sensitive=True)
    if args.TARGET_NAME not in input_data.dtype.names:
        raise ValueError('The provided data file does not seem to have a target column. '
                         'Please make sure to specify the target column using the -target parameter.')

    features = np.delete(input_data.view(np.float64).reshape(input_data.size, -1),
                         input_data.dtype.names.index(args.TARGET_NAME), axis=1)

    training_features, testing_features, training_classes, testing_classes = \
        train_test_split(features, input_data[args.TARGET_NAME], random_state=args.RANDOM_STATE)

    if args.TPOT_MODE == 'classification':
        tpot_type = TPOTClassifier
    else:
        tpot_type = TPOTRegressor

    tpot = tpot_type(generations=args.GENERATIONS, population_size=args.POPULATION_SIZE,
                mutation_rate=args.MUTATION_RATE, crossover_rate=args.CROSSOVER_RATE,
                num_cv_folds=args.NUM_CV_FOLDS, scoring=args.SCORING_FN,
                max_time_mins=args.MAX_TIME_MINS, max_eval_time_mins=args.MAX_EVAL_MINS,
                random_state=args.RANDOM_STATE, verbosity=args.VERBOSITY,
                disable_update_check=args.DISABLE_UPDATE_CHECK)

    tpot.fit(training_features, training_classes)

    if args.VERBOSITY in [1, 2] and tpot._optimized_pipeline:
        training_score = max([tpot._hof.keys[x].wvalues[1] for x in range(len(tpot._hof.keys))])
        print('\nTraining score: {}'.format(abs(training_score)))
        print('Holdout score: {}'.format(tpot.score(testing_features, testing_classes)))

    elif args.VERBOSITY >= 3 and tpot._hof:
        print('Final Pareto front testing scores:')

        for pipeline, pipeline_scores in zip(tpot._hof.items, reversed(tpot._hof.keys)):
            tpot._fitted_pipeline = tpot._hof_fitted_pipelines[str(pipeline)]
            print('{}\t{}\t{}'.format(int(abs(pipeline_scores.wvalues[0])),
                                      tpot.score(testing_features, testing_classes),
                                      pipeline))

    if args.OUTPUT_FILE != '':
        tpot.export(args.OUTPUT_FILE)


if __name__ == '__main__':
    main()
