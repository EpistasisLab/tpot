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

from .base import *
# Temporarily remove the RFE operator. In many cases it seems to be slow and causes TPOT to freeze.
# TODO: Dig into the freezing issue with RFE and see if we can add it back under certain constraints.
#from .rfe import *
from .select_fwe import *
from .select_kbest import *
from .select_percentile import *
from .variance_threshold import *
from .select_from_model import *
from .select_from_model_r import *
