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

# This configuration only includes the text extractors. 
# These are appended to any config selected if the input is indicated as an text.
# This config is NOT meant to be used by itself. 
# It is selected/appended to any config used if the argument
# input_type='text' is passed to the TPOT object on instantiation

config_textfeatureextract = {

    #TODO: Add text feature extractor(s) here
    'tpot.builtins.TfidfVectorizerTextExtractor': {
        'analyzer': ["word", "char", "char_wb"],
        'ngram_range': [(1,1), (1,2), (2,2)],
        'max_df': [0.75, 0.9, 1.0],
        'min_df': [1, 0.1, 0.25],
        'binary': [True, False],
        'norm': ['l1', 'l2'],
        'use_idf': [True, False],
        'sublinear_tf': [True, False]
    },

    'tpot.builtins.CountVectorizerTextExtractor': {
        'analyzer': ["word", "char", "char_wb"],
        'ngram_range': [(1,1), (1,2), (2,2)],
        'max_df': [0.75, 0.9, 1.0],
        'min_df': [1, 0.1, 0.25],
        'binary': [True, False]
    },

}
