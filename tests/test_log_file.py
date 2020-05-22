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

from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from nose.tools import assert_equal
import os

data = load_iris()
X = data['data']
y = data['target']

def test_log_file_verbosity_1():
  """ Set verbosity as 1. Assert log_file parameter to generate log file. """
  file_name = "progress_verbose_1.log"
  tracking_progress_file = open(file_name, "w")
  tpot_obj = TPOTClassifier(
                population_size=10,
                generations=10,
                verbosity=1, 
                log_file=tracking_progress_file
            )
  tpot_obj.fit(X, y)
  assert_equal(os.path.getsize(file_name), 0)

def test_log_file_verbosity_2():
  """ Set verbosity as 2. Assert log_file parameter to generate log file. """
  file_name = "progress_verbose_2.log"
  tracking_progress_file = open(file_name, "w")
  tpot_obj = TPOTClassifier(
                population_size=10,
                generations=10,
                verbosity=2, 
                log_file=tracking_progress_file
            )
  tpot_obj.fit(X, y)
  assert_equal(os.path.getsize(file_name) > 0,  True)

def test_log_file_verbose_3():
  """ Set verbosity as 3. Assert log_file parameter to generate log file. """
  file_name = "progress_verbosity_3.log"
  tracking_progress_file = open(file_name, "w")
  tpot_obj = TPOTClassifier(
                population_size=10,
                generations=10,
                verbosity=3, 
                log_file=tracking_progress_file
            )
  tpot_obj.fit(X, y)
  assert_equal(os.path.getsize(file_name) > 0,  True)
