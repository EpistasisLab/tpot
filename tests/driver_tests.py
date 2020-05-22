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

import subprocess
import sys
from os import remove, path
from contextlib import contextmanager
from distutils.version import LooseVersion
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import nose
import numpy as np
import pandas as pd
import sklearn

from tpot.driver import positive_integer, float_range, _get_arg_parser, \
    _print_args, _read_data_file, load_scoring_function, tpot_driver, \
    positive_integer_or_none
from nose.tools import assert_raises, assert_equal, assert_in
from unittest import TestCase



@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_scoring_function_argument():
    with captured_output() as (out, err):
        # regular argument returns regular string
        assert_equal(load_scoring_function("roc_auc"), "roc_auc")

        # bad function returns exception
        assert_raises(Exception, load_scoring_function, scoring_func="tests.__fake_BAD_FUNC_NAME")

        # manual function loads the function
        assert_equal(load_scoring_function('driver_tests.test_scoring_function_argument').__name__, test_scoring_function_argument.__name__)

        # installed-module function test
        assert_equal(load_scoring_function('sklearn.metrics.auc').__name__, "auc")

        out, err = out.getvalue(), err.getvalue()

    assert_in("failed importing custom scoring function", out)
    assert_in("manual scoring function: <function auc", out)
    assert_in("taken from module: sklearn.metrics", out)
    assert_in("manual scoring function: <function test_scoring_function_argument", out)
    assert_in("taken from module: driver_tests", out)
    assert_equal(err, "")


def test_driver():
    """Assert that the TPOT driver outputs normal result in mode mode."""
    batcmd = "python -m tpot.driver tests/tests.csv -is , -target class -g 1 -p 2 -os 4 -cv 5 -s 45 -v 1"
    ret_stdout = subprocess.check_output(batcmd, shell=True)
    try:
        ret_val = float(ret_stdout.decode('UTF-8').split('\n')[-2].split(': ')[-1])

    except Exception as e:
        ret_val = -float('inf')
    assert ret_val > 0.0


def test_driver_2():
    """Assert that the tpot_driver() in TPOT driver outputs normal result with verbosity = 1."""
    args_list = [
                'tests/tests.csv',
                '-is', ',',
                '-target', 'class',
                '-g', '1',
                '-p', '2',
                '-cv', '2',
                '-s',' 45',
                '-config', 'TPOT light',
                '-v', '1'
                ]
    args = _get_arg_parser().parse_args(args_list)
    with captured_output() as (out, err):
        tpot_driver(args)
    ret_stdout = out.getvalue()

    assert "TPOT settings" not in ret_stdout
    assert "Final Pareto front testing scores" not in ret_stdout
    try:
        ret_val = float(ret_stdout.split('\n')[-2].split(': ')[-1])
    except Exception:
        ret_val = -float('inf')
    assert ret_val > 0.0


def test_driver_3():
    """Assert that the tpot_driver() in TPOT driver outputs normal result with verbosity = 2."""
    args_list = [
                'tests/tests.csv',
                '-is', ',',
                '-target', 'class',
                '-g', '1',
                '-p', '2',
                '-cv', '3',
                '-s',' 45',
                '-config', 'TPOT light',
                '-v', '2'
                ]
    args = _get_arg_parser().parse_args(args_list)
    with captured_output() as (out, err):
        tpot_driver(args)
    ret_stdout = out.getvalue()
    assert "TPOT settings" in ret_stdout
    assert "Final Pareto front testing scores" not in ret_stdout
    try:
        ret_val = float(ret_stdout.split('\n')[-2].split(': ')[-1])
    except Exception:
        ret_val = -float('inf')
    assert ret_val > 0.0


def test_driver_4():
    """Assert that the tpot_driver() in TPOT driver outputs normal result with verbosity = 3."""
    args_list = [
                'tests/tests.csv',
                '-is', ',',
                '-target', 'class',
                '-g', '1',
                '-p', '2',
                '-cv', '3',
                '-s', '42',
                '-config', 'TPOT light',
                '-v', '3'
                ]
    args = _get_arg_parser().parse_args(args_list)
    with captured_output() as (out, err):
        tpot_driver(args)
    ret_stdout = out.getvalue()

    assert "TPOT settings" in ret_stdout
    assert "Final Pareto front testing scores" in ret_stdout
    try:
        ret_val = float(ret_stdout.split('\n')[-2].split('\t')[1])
    except Exception:
        ret_val = -float('inf')
    assert ret_val > 0.0


def test_driver_5():
    """Assert that the tpot_driver() in TPOT driver outputs normal result with exported python file and verbosity = 0."""

    # Catch FutureWarning https://github.com/scikit-learn/scikit-learn/issues/11785
    if (np.__version__ >= LooseVersion("1.15.0") and
            sklearn.__version__ <= LooseVersion("0.20.0")):
        raise nose.SkipTest("Warning raised by scikit-learn")

    args_list = [
                'tests/tests.csv',
                '-is', ',',
                '-target', 'class',
                '-o', 'test_export.py',
                '-g', '1',
                '-p', '2',
                '-cv', '3',
                '-s', '42',
                '-config', 'TPOT light',
                '-v', '0'
                ]
    args = _get_arg_parser().parse_args(args_list)
    with captured_output() as (out, err):
        tpot_driver(args)
    ret_stdout = out.getvalue()

    assert ret_stdout == ""
    assert path.isfile("test_export.py")
    remove("test_export.py") # clean up exported file


def test_read_data_file():
    """Assert that _read_data_file raises ValueError when the targe column is missing."""
    # Mis-spelled target
    args_list = [
        'tests/tests.csv',
        '-is', ',',
        '-target', 'clas'   # typo for right target 'class'
    ]
    args = _get_arg_parser().parse_args(args_list)
    assert_raises(ValueError, _read_data_file, args=args)

    # Correctly spelled target
    args_list = [
        'tests/tests.csv',
        '-is', ',',
        '-target', 'class'
    ]
    args = _get_arg_parser().parse_args(args_list)
    input_data = _read_data_file(args)

    assert isinstance(input_data, pd.DataFrame)


class ParserTest(TestCase):
    def setUp(self):
        self.parser = _get_arg_parser()


    def test_default_param(self):
        """Assert that the TPOT driver stores correct default values for all parameters."""
        args = self.parser.parse_args(['tests/tests.csv'])
        self.assertEqual(args.CONFIG_FILE, None)
        self.assertEqual(args.CROSSOVER_RATE, 0.1)
        self.assertEqual(args.EARLY_STOP, None)
        self.assertEqual(args.DISABLE_UPDATE_CHECK, False)
        self.assertEqual(args.GENERATIONS, 100)
        self.assertEqual(args.INPUT_FILE, 'tests/tests.csv')
        self.assertEqual(args.INPUT_SEPARATOR, '\t')
        self.assertEqual(args.MAX_EVAL_MINS, 5)
        self.assertEqual(args.MAX_TIME_MINS, None)
        self.assertEqual(args.MEMORY, None)
        self.assertEqual(args.MUTATION_RATE, 0.9)
        self.assertEqual(args.NUM_CV_FOLDS, 5)
        self.assertEqual(args.NUM_JOBS, 1)
        self.assertEqual(args.OFFSPRING_SIZE, None)
        self.assertEqual(args.OUTPUT_FILE, None)
        self.assertEqual(args.POPULATION_SIZE, 100)
        self.assertEqual(args.RANDOM_STATE, None)
        self.assertEqual(args.SUBSAMPLE, 1.0)
        self.assertEqual(args.SCORING_FN, None)
        self.assertEqual(args.TARGET_NAME, 'class')
        self.assertEqual(args.TEMPLATE, None)
        self.assertEqual(args.TPOT_MODE, 'classification')
        self.assertEqual(args.VERBOSITY, 1)


    def test_print_args(self):
        """Assert that _print_args prints correct values for all parameters in default settings."""
        args_list = [
            'tests/tests.csv',
            '-is', ','
        ]
        args = self.parser.parse_args(args_list)
        with captured_output() as (out, err):
            _print_args(args)
        output = out.getvalue()
        expected_output = """
TPOT settings:
CHECKPOINT_FOLDER   =     None
CONFIG_FILE         =     None
CROSSOVER_RATE      =     0.1
EARLY_STOP          =     None
GENERATIONS         =     100
INPUT_FILE          =     tests/tests.csv
INPUT_SEPARATOR     =     ,
LOG                 =     None
MAX_EVAL_MINS       =     5
MAX_TIME_MINS       =     None
MEMORY              =     None
MUTATION_RATE       =     0.9
NUM_CV_FOLDS        =     5
NUM_JOBS            =     1
OFFSPRING_SIZE      =     100
OUTPUT_FILE         =     None
POPULATION_SIZE     =     100
RANDOM_STATE        =     None
SCORING_FN          =     accuracy
SUBSAMPLE           =     1.0
TARGET_NAME         =     class
TEMPLATE            =     None
TPOT_MODE           =     classification
VERBOSITY           =     1

"""
        self.assertEqual(_sort_lines(expected_output), _sort_lines(output))


    def test_print_args_2(self):
        """Assert that _print_args prints correct values for all parameters in regression mode."""
        args_list = [
            'tests/tests.csv',
            '-mode', 'regression',
            '-is', ','
        ]
        args = self.parser.parse_args(args_list)
        with captured_output() as (out, err):
            _print_args(args)
        output = out.getvalue()
        expected_output = """
TPOT settings:
CHECKPOINT_FOLDER   =     None
CONFIG_FILE         =     None
CROSSOVER_RATE      =     0.1
EARLY_STOP          =     None
GENERATIONS         =     100
INPUT_FILE          =     tests/tests.csv
INPUT_SEPARATOR     =     ,
LOG                 =     None
MAX_EVAL_MINS       =     5
MAX_TIME_MINS       =     None
MEMORY              =     None
MUTATION_RATE       =     0.9
NUM_CV_FOLDS        =     5
NUM_JOBS            =     1
OFFSPRING_SIZE      =     100
OUTPUT_FILE         =     None
POPULATION_SIZE     =     100
RANDOM_STATE        =     None
SCORING_FN          =     neg_mean_squared_error
SUBSAMPLE           =     1.0
TARGET_NAME         =     class
TEMPLATE            =     None
TPOT_MODE           =     regression
VERBOSITY           =     1

"""

        self.assertEqual(_sort_lines(expected_output), _sort_lines(output))


def _sort_lines(text):
    return '\n'.join(sorted(text.split('\n')))


def test_positive_integer():
    """Assert that the TPOT CLI interface's integer parsing throws an exception when n < 0."""
    assert_raises(Exception, positive_integer, '-1')


def test_positive_integer_2():
    """Assert that the TPOT CLI interface's integer parsing returns the integer value of a string encoded integer when n > 0."""
    assert 1 == positive_integer('1')


def test_positive_integer_3():
    """Assert that the TPOT CLI interface's integer parsing throws an exception when n is not an integer."""
    assert_raises(Exception, positive_integer, 'foobar')

def test_positive_integer_or_none():
    """Assert that the TPOT CLI interface's positive_integer_or_none parsing throws an exception when n < 0."""
    assert_raises(Exception, positive_integer_or_none, '-1')


def test_positive_integer_or_none_2():
    """Assert that the TPOT CLI interface's positive_integer_or_none parsing returns the integer value of a string encoded integer when n > 0."""
    assert 1 == positive_integer_or_none('1')


def test_positive_integer_or_none_3():
    """Assert that the TPOT CLI interface's positive_integer_or_none parsing throws an exception when n is not an integer and not None."""
    assert_raises(Exception, positive_integer_or_none, 'foobar')


def test_positive_integer_or_none_4():
    """Assert that the TPOT CLI interface's positive_integer_or_none parsing return None when value is string 'None' or 'none'."""
    assert positive_integer_or_none('none') is None
    assert positive_integer_or_none('None') is None


def test_float_range():
    """Assert that the TPOT CLI interface's float range returns a float with input is in 0. - 1.0."""
    assert 0.5 == float_range('0.5')


def test_float_range_2():
    """Assert that the TPOT CLI interface's float range throws an exception when input it out of range."""
    assert_raises(Exception, float_range, '2.0')


def test_float_range_3():
    """Assert that the TPOT CLI interface's float range throws an exception when input is not a float."""
    assert_raises(Exception, float_range, 'foobar')
