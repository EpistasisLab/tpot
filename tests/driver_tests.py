# -*- coding: utf-8 -*-

"""Copyright 2015-Present Randal S. Olson.

This file is part of the TPOT library.

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

from tpot.driver import positive_integer, float_range
from nose.tools import assert_raises


def test_driver():
    """Assert that the TPOT driver output normal result."""
    batcmd = "python -m tpot.driver tests/tests.csv -is , -target class -g 2 -p 2 -os 4 -cv 5 -s 45 -v 1"
    ret_stdout = subprocess.check_output(batcmd, shell=True)

    try:
        ret_val = float(ret_stdout.decode('UTF-8').split('\n')[-2].split(': ')[-1])
    except Exception:
        ret_val = -float('inf')

    assert ret_val > 0.0


def test_positive_integer():
    """Assert that the TPOT CLI interface's integer parsing throws an exception when n < 0."""
    assert_raises(Exception, positive_integer, '-1')


def test_positive_integer_2():
    """Assert that the TPOT CLI interface's integer parsing returns the integer value of a string encoded integer when n > 0."""
    assert 1 == positive_integer('1')


def test_positive_integer_3():
    """Assert that the TPOT CLI interface's integer parsing throws an exception when n is not an integer."""
    assert_raises(Exception, positive_integer, 'foobar')


def test_float_range():
    """Assert that the TPOT CLI interface's float range returns a float with input is in 0. - 1.0."""
    assert 0.5 == float_range('0.5')


def test_float_range_2():
    """Assert that the TPOT CLI interface's float range throws an exception when input it out of range."""
    assert_raises(Exception, float_range, '2.0')


def test_float_range_3():
    """Assert that the TPOT CLI interface's float range throws an exception when input is not a float."""
    assert_raises(Exception, float_range, 'foobar')
