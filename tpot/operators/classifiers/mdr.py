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

from .base import Classifier
from mdr import MDR


class TPOTMDR(Classifier):
    """Fits a MDR Classifier

    Parameters
    ----------
    tie_break: int
        Default label in case there's a tie in a set of feature pair values 
    default_label: int
        Default label in case there's no data for a set of feature pair values

    """
    import_hash = {'mdr': ['MDR']}
    sklearn_class = MDR
    arg_types = (int, int)

    def __init__(self):
        pass

    def preprocess_args(self, tie_break, default_label):
        tie_break = max(0, min(1, tie_break))
        default_label = max(0, min(1, tie_break))

        return {
            'tie_break': tie_break,
            'default_label': default_label
        }
