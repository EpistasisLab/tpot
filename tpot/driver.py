# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

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

import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# for manual scoring function, see load_scoring_function
import sys
import os
from importlib import import_module

from .tpot import TPOTClassifier, TPOTRegressor
from ._version import __version__


def positive_integer(value):
    """Ensure that the provided value is a positive integer.

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
    """Ensure that the provided value is a float integer in the range [0., 1.].

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
    except Exception:
        raise argparse.ArgumentTypeError('Invalid float value: \'{}\''.format(value))
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError('Invalid float value: \'{}\''.format(value))
    return value


def _get_arg_parser():
    """Main function that is called when TPOT is run on the command line."""
    parser = argparse.ArgumentParser(
        description=(
            'A Python tool that automatically creates and optimizes machine '
            'learning pipelines using genetic programming.'
        ),
        add_help=False
    )

    parser.add_argument(
        'INPUT_FILE',
        type=str,
        help=(
            'Data file to use in the TPOT optimization process. Ensure that '
            'the class label column is labeled as "class".'
        )
    )

    parser.add_argument(
        '-h',
        '--help',
        action='help',
        help='Show this help message and exit.'
    )

    parser.add_argument(
        '-is',
        action='store',
        dest='INPUT_SEPARATOR',
        default='\t',
        type=str,
        help='Character used to separate columns in the input file.'
    )

    parser.add_argument(
        '-target',
        action='store',
        dest='TARGET_NAME',
        default='class',
        type=str,
        help='Name of the target column in the input file.'
    )

    parser.add_argument(
        '-mode',
        action='store',
        dest='TPOT_MODE',
        choices=['classification', 'regression'],
        default='classification',
        type=str,
        help=(
            'Whether TPOT is being used for a supervised classification or '
            'regression problem.'
        )
    )

    parser.add_argument(
        '-o',
        action='store',
        dest='OUTPUT_FILE',
        default=None,
        type=str,
        help='File to export the code for the final optimized pipeline.'
    )

    parser.add_argument(
        '-g',
        action='store',
        dest='GENERATIONS',
        default=100,
        type=positive_integer,
        help=(
            'Number of iterations to run the pipeline optimization process. '
            'Generally, TPOT will work better when you give it more '
            'generations (and therefore time) to optimize the pipeline. TPOT '
            'will evaluate POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE '
            'pipelines in total.'
        )
    )

    parser.add_argument(
        '-p',
        action='store',
        dest='POPULATION_SIZE',
        default=100,
        type=positive_integer,
        help=(
            'Number of individuals to retain in the GP population every '
            'generation. Generally, TPOT will work better when you give it '
            'more individuals (and therefore time) to optimize the pipeline. '
            'TPOT will evaluate POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE '
            'pipelines in total.'
        )
    )

    parser.add_argument(
        '-os',
        action='store',
        dest='OFFSPRING_SIZE',
        default=None,
        type=positive_integer,
        help=(
            'Number of offspring to produce in each GP generation. By default,'
            'OFFSPRING_SIZE = POPULATION_SIZE.'
        )
    )

    parser.add_argument(
        '-mr',
        action='store',
        dest='MUTATION_RATE',
        default=0.9,
        type=float_range,
        help=(
            'GP mutation rate in the range [0.0, 1.0]. This tells the GP '
            'algorithm how many pipelines to apply random changes to every '
            'generation. We recommend using the default parameter unless you '
            'understand how the mutation rate affects GP algorithms.'
        )
    )

    parser.add_argument(
        '-xr',
        action='store',
        dest='CROSSOVER_RATE',
        default=0.1,
        type=float_range,
        help=(
            'GP crossover rate in the range [0.0, 1.0]. This tells the GP '
            'algorithm how many pipelines to "breed" every generation. We '
            'recommend using the default parameter unless you understand how '
            'the crossover rate affects GP algorithms.'
        )
    )

    parser.add_argument(
        '-scoring',
        action='store',
        dest='SCORING_FN',
        default=None,
        type=str,
        help=(
            'Function used to evaluate the quality of a given pipeline for the '
            'problem. By default, accuracy is used for classification problems '
            'and mean squared error (mse) is used for regression problems. '

            'Note: If you wrote your own function, set this argument to mymodule.myfunction'
            'and TPOT will import your module and take the function from there.'
            'TPOT will assume the module can be imported from the current workdir.'

            'TPOT assumes that any function with "error" or "loss" in the name '
            'is meant to be minimized, whereas any other functions will be '
            'maximized. Offers the same options as cross_val_score: '
            'accuracy, '
            'adjusted_rand_score, '
            'average_precision, '
            'f1, '
            'f1_macro, '
            'f1_micro, '
            'f1_samples, '
            'f1_weighted, '
            'neg_log_loss, '
            'neg_mean_absolute_error, '
            'neg_mean_squared_error, '
            'neg_median_absolute_error, '
            'precision, '
            'precision_macro, '
            'precision_micro, '
            'precision_samples, '
            'precision_weighted, '
            'r2, '
            'recall, '
            'recall_macro, '
            'recall_micro, '
            'recall_samples, '
            'recall_weighted, '
            'roc_auc'
        )
    )

    parser.add_argument(
        '-cv',
        action='store',
        dest='NUM_CV_FOLDS',
        default=5,
        type=int,
        help=(
            'Number of folds to evaluate each pipeline over in stratified k-fold '
            'cross-validation during the TPOT optimization process.'
        )
    )

    parser.add_argument(
        '-sub',
        action='store',
        dest='SUBSAMPLE',
        default=1.0,
        type=float,
        help=(
            'Subsample ratio of the training instance. Setting it to 0.5 means that TPOT '
            'use a random subsample of half of training data for the pipeline optimization process.'
        )
    )


    parser.add_argument(
        '-njobs',
        action='store',
        dest='NUM_JOBS',
        default=1,
        type=int,
        help=(
            'Number of CPUs for evaluating pipelines in parallel during the '
            'TPOT optimization process. Assigning this to -1 will use as many '
            'cores as available on the computer.'
        )
    )

    parser.add_argument(
        '-maxtime',
        action='store',
        dest='MAX_TIME_MINS',
        default=None,
        type=int,
        help=(
            'How many minutes TPOT has to optimize the pipeline. This setting '
            'will override the GENERATIONS parameter and allow TPOT to run '
            'until it runs out of time.'
        )
    )

    parser.add_argument(
        '-maxeval',
        action='store',
        dest='MAX_EVAL_MINS',
        default=5,
        type=float,
        help=(
            'How many minutes TPOT has to evaluate a single pipeline. Setting '
            'this parameter to higher values will allow TPOT to explore more '
            'complex pipelines but will also allow TPOT to run longer.'
        )
    )

    parser.add_argument(
        '-s',
        action='store',
        dest='RANDOM_STATE',
        default=None,
        type=int,
        help=(
            'Random number generator seed for reproducibility. Set this seed '
            'if you want your TPOT run to be reproducible with the same seed '
            'and data set in the future.'
        )
    )


    parser.add_argument(
        '-config',
        action='store',
        dest='CONFIG_FILE',
        default=None,
        type=str,
        help=(
            'Configuration file for customizing the operators and parameters '
            'that TPOT uses in the optimization process. Must be a Python '
            'module containing a dict export named "tpot_config" or the name of '
            'built-in configuration.'
        )
    )


    parser.add_argument(
        '-memory',
        action='store',
        dest='MEMORY',
        default=None,
        type=str,
        help=(
            'Path of a directory for pipeline caching or \"auto\" for using a temporary '
            'caching directory during the optimization process. If supplied, pipelines will '
            'cache each transformer after fitting them. This feature is used to avoid '
            'repeated computation by transformers within a pipeline if the parameters and '
            'input data are identical with another fitted pipeline during optimization process.'
        )
    )


    parser.add_argument(
        '-cf',
        action='store',
        dest='CHECKPOINT_FOLDER',
        default=None,
        type=str,
        help=('If supplied, a folder in which tpot will periodically '
        'save the best pipeline so far while optimizing. '
        'This is useful in multiple cases: '
        'sudden death before tpot could save an optimized pipeline, '
        'progress tracking, '
        "grabbing a pipeline while it's still optimizing etc."
        )
    )

    parser.add_argument(
        '-es',
        action='store',
        dest='EARLY_STOP',
        default=None,
        type=int,
        help=(
            'How many generations TPOT checks whether there is no improvement '
            'in optimization process. End optimization process if there is no improvement '
            'in the set number of generations.'
        )
    )

    parser.add_argument(
        '-v',
        action='store',
        dest='VERBOSITY',
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help=(
            'How much information TPOT communicates while it is running: '
            '0 = none, 1 = minimal, 2 = high, 3 = all. A setting of 2 or '
            'higher will add a progress bar during the optimization procedure.'
        )
    )

    parser.add_argument(
        '--no-update-check',
        action='store_true',
        dest='DISABLE_UPDATE_CHECK',
        default=False,
        help='Flag indicating whether the TPOT version checker should be disabled.'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='TPOT {version}'.format(version=__version__),
        help='Show the TPOT version number and exit.'
    )

    return parser


def _print_args(args):
    print('\nTPOT settings:')
    for arg, arg_val in sorted(args.__dict__.items()):
        if arg == 'DISABLE_UPDATE_CHECK':
            continue
        elif arg == 'SCORING_FN' and arg_val is None:
            if args.TPOT_MODE == 'classification':
                arg_val = 'accuracy'
            else:
                arg_val = 'neg_mean_squared_error'
        elif arg == 'OFFSPRING_SIZE' and arg_val is None:
            arg_val = args.__dict__['POPULATION_SIZE']
        else:
            arg_val = args.__dict__[arg]

        # Pad the outputs with an even amount of space
        arg = (arg + (' ') * 100)[:20]
        arg_val = ((' ') * 5 + str(arg_val))
        print('{}={}'.format(arg, arg_val))
    print('')


def _read_data_file(args):
    input_data = pd.read_csv(
        args.INPUT_FILE,
        sep=args.INPUT_SEPARATOR,
        dtype=np.float64,
    )

    if args.TARGET_NAME not in input_data.columns.values:
        raise ValueError(
            'The provided data file does not seem to have a target column. '
            'Please make sure to specify the target column using the -target '
            'parameter.'
        )

    return input_data


def load_scoring_function(scoring_func):
    """
    converts mymodule.myfunc in the myfunc
    object itself so tpot receives a scoring function
    """
    if scoring_func and ("." in scoring_func):
        try:
            module_name, func_name = scoring_func.rsplit('.', 1)

            module_path = os.getcwd()
            sys.path.insert(0, module_path)
            scoring_func = getattr(import_module(module_name), func_name)
            sys.path.pop(0)

            print('manual scoring function: {}'.format(scoring_func))
            print('taken from module: {}'.format(module_name))
        except Exception as e:
            print('failed importing custom scoring function, error: {}'.format(str(e)))
            raise ValueError(e)

    return scoring_func


def tpot_driver(args):
    """Perform a TPOT run."""
    if args.VERBOSITY >= 2:
        _print_args(args)

    input_data = _read_data_file(args)
    features = input_data.drop(args.TARGET_NAME, axis=1).values

    training_features, testing_features, training_target, testing_target = \
        train_test_split(features, input_data[args.TARGET_NAME].values, random_state=args.RANDOM_STATE)

    tpot_type = TPOTClassifier if args.TPOT_MODE == 'classification' else TPOTRegressor

    scoring_func = load_scoring_function(args.SCORING_FN)

    tpot_obj = tpot_type(
        generations=args.GENERATIONS,
        population_size=args.POPULATION_SIZE,
        offspring_size=args.OFFSPRING_SIZE,
        mutation_rate=args.MUTATION_RATE,
        crossover_rate=args.CROSSOVER_RATE,
        cv=args.NUM_CV_FOLDS,
        subsample=args.SUBSAMPLE,
        n_jobs=args.NUM_JOBS,
        scoring=scoring_func,
        max_time_mins=args.MAX_TIME_MINS,
        max_eval_time_mins=args.MAX_EVAL_MINS,
        random_state=args.RANDOM_STATE,
        config_dict=args.CONFIG_FILE,
        memory=args.MEMORY,
        periodic_checkpoint_folder=args.CHECKPOINT_FOLDER,
        early_stop=args.EARLY_STOP,
        verbosity=args.VERBOSITY,
        disable_update_check=args.DISABLE_UPDATE_CHECK
    )

    tpot_obj.fit(training_features, training_target)

    if args.VERBOSITY in [1, 2] and tpot_obj._optimized_pipeline:
        training_score = max([x.wvalues[1] for x in tpot_obj._pareto_front.keys])
        print('\nTraining score: {}'.format(abs(training_score)))
        print('Holdout score: {}'.format(tpot_obj.score(testing_features, testing_target)))

    elif args.VERBOSITY >= 3 and tpot_obj._pareto_front:
        print('Final Pareto front testing scores:')
        pipelines = zip(tpot_obj._pareto_front.items, reversed(tpot_obj._pareto_front.keys))
        for pipeline, pipeline_scores in pipelines:
            tpot_obj._fitted_pipeline = tpot_obj.pareto_front_fitted_pipelines_[str(pipeline)]
            print('{TRAIN_SCORE}\t{TEST_SCORE}\t{PIPELINE}'.format(
                    TRAIN_SCORE=int(abs(pipeline_scores.wvalues[0])),
                    TEST_SCORE=tpot_obj.score(testing_features, testing_target),
                    PIPELINE=pipeline
                )
            )

    if args.OUTPUT_FILE:
        tpot_obj.export(args.OUTPUT_FILE)

def main():
    args = _get_arg_parser().parse_args()
    tpot_driver(args)

if __name__ == '__main__':
    main()
